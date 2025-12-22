"""
HOPE (Hierarchical Optimization for Predictive Encoding) 实现

本模块实现了使用嵌套优化循环进行记忆和注意力机制的 HOPE 模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def hope_scan(k, v, q, initial_state, eta, beta):
    """
    HOPE 记忆更新的融合内核。
    
    k, v, q: (Batch, Seq, Head_Dim)
    initial_state: (Batch, Head_Dim, Head_Dim)
    eta: 内部循环的学习率
    beta: 遗忘因子
    """
    b, t, d = k.shape
    
    # 初始化状态
    state = initial_state  # (B, D, D)
    outputs = torch.jit.annotate(torch.Tensor, torch.zeros(b, t, d, device=k.device))
    
    # 遍历序列
    for i in range(t):
        k_t = k[:, i].unsqueeze(2)  # (B, D, 1)
        v_t = v[:, i].unsqueeze(2)
        q_t = q[:, i].unsqueeze(2)
        
        # 1. 查询记忆（更新前预测）
        # y = M * q
        y_t = torch.bmm(state, q_t)
        outputs[:, i] = y_t.squeeze(2)
        
        # 2. 更新记忆状态（内部优化步骤）
        # 预测：v_pred = M * k
        v_pred = torch.bmm(state, k_t)
        
        # 误差：e = v - v_pred
        error = v_t - v_pred
        
        # 梯度/增量：delta = error * k.T
        delta = torch.bmm(error, k_t.transpose(1, 2))
        
        # 更新规则：M_new = Beta * M_old + Eta * Delta
        state = (beta * state) + (eta * delta)
    
    return outputs


def hope_update_rule(W, x, grad_loss, eta):
    """
    实现 HOPE 优化器更新规则（论文中的公式 28/29）。
    
    W_{t+1} = W_t(I - x_t x_t^T) - eta * grad_L
    
    参数:
        W: 当前权重状态 (Batch, Out_Dim, In_Dim)
        x: 输入向量 (Batch, In_Dim, 1)
        grad_loss: 损失关于 W 的梯度（或简化：error * x.T）
        eta: 学习率
    """
    # 项 1: W_t (I - x x^T) -> 投影/遗忘项
    # x @ x.transpose: (B, In, 1) @ (B, 1, In) -> (B, In, In)
    # I - xxT: (B, In, In)
    I = torch.eye(W.shape[-1], device=W.device).unsqueeze(0).expand(W.shape[0], -1, -1)
    
    # 论文建议 x 应该归一化以在此更新中保持稳定性
    x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
    projection_matrix = I - torch.bmm(x_norm, x_norm.transpose(1, 2))
    
    term_1 = torch.bmm(W, projection_matrix)
    
    # 项 2: 梯度更新
    term_2 = eta * grad_loss
    
    W_new = term_1 - term_2
    return W_new


class HOPERecurrentMemory(nn.Module):
    """
    HOPE 的"工作记忆"。
    它学习使用嵌套更新规则将上下文（Keys -> Values）压缩到权重矩阵 M_t 中，
    然后回答 Queries。
    """
    def __init__(self, dim, head_dim, learning_rate=0.1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.lr = learning_rate
        
        # 投影（标准 Transformer 风格）
        self.W_q = nn.Linear(dim, head_dim, bias=False)
        self.W_k = nn.Linear(dim, head_dim, bias=False)
        self.W_v = nn.Linear(dim, head_dim, bias=False)
        self.W_o = nn.Linear(head_dim, dim, bias=False)
        
        # 归一化以在循环更新中保持稳定性
        self.ln_k = nn.LayerNorm(head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.W_q(x) # (B, T, H)
        k = self.ln_k(self.W_k(x)) # (B, T, H)
        v = self.W_v(x) # (B, T, H)
        
        # 初始化记忆状态 M_0 (Batch, Head, Head)
        # M 映射 Keys (Head) -> Values (Head)
        M = torch.zeros(batch_size, self.head_dim, self.head_dim, device=x.device)
        
        outputs = []
        
        # 顺序处理（线性注意力的循环视图）
        # 注意：在生产环境中，可以通过分块/CUDA 并行化（如 Titans/Mamba）
        for t in range(seq_len):
            k_t = k[:, t, :].unsqueeze(2) # (B, H, 1)
            v_t = v[:, t, :].unsqueeze(2) # (B, H, 1)
            q_t = q[:, t, :].unsqueeze(2) # (B, H, 1)
            
            # 1. 预测/损失计算（隐式）
            # 我们希望 M_t 将 k_t 映射到 v_t。
            # 误差信号：(M_t * k_t - v_t)
            pred_v = torch.bmm(M, k_t)
            error = pred_v - v_t
            
            # MSE 的梯度 = 2 * error * k_t.T
            grad_loss = torch.bmm(error, k_t.transpose(1, 2))
            
            # 2. 使用 HOPE 规则更新记忆 M_t -> M_{t+1}（公式 29）
            M = hope_update_rule(M, k_t, grad_loss, self.lr)
            
            # 3. 使用更新后的记忆计算输出
            # y_t = M_{t+1} * q_t
            y_t = torch.bmm(M, q_t).squeeze(2)
            outputs.append(y_t)
            
        y = torch.stack(outputs, dim=1) # (B, T, H)
        return self.W_o(y)


class CMSLayer(nn.Module):
    """
    Continuum Memory System 的单个层级（用于 HOPE）。
    概念：它是一个保持状态并根据特定频率的局部惊喜信号更新自身权重的 MLP。
    """
    def __init__(self, dim, hidden_dim, update_freq=1, learning_rate=0.01):
        super().__init__()
        self.dim = dim
        self.update_freq = update_freq
        self.lr = learning_rate
        
        # 我们将权重作为缓冲区保存，以便在需要时可以手动按批次更新它们，
        # 但在这个演示中，我们将它们作为通过自定义逻辑更新的参数。
        # 模拟每个样本的"快速权重"很昂贵，因此我们在批次间共享权重
        # 但在前向传播期间按时间更新它们。
        self.w1 = nn.Parameter(torch.randn(hidden_dim, dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(dim, hidden_dim) * 0.02)
        
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward_computation(self, x, w1, w2):
        # x: (B, D)
        h = F.linear(x, w1)
        h = self.act(h)
        return F.linear(h, w2)

    def forward(self, x, global_step=0):
        """
        x: (Batch, Seq, Dim)
        """
        batch_size, seq_len, dim = x.shape
        outputs = []
        
        # 克隆当前权重用于临时修改（可塑性）
        # 在完整实现中，这些将是每个批次的状态。
        curr_w1 = self.w1.clone()
        curr_w2 = self.w2.clone()
        
        for t in range(seq_len):
            input_t = x[:, t, :]
            
            # 标准 FFN 前向传播
            out_t = self.forward_computation(input_t, curr_w1, curr_w2)
            outputs.append(out_t)
            
            # CMS 更新逻辑（公式 31）
            # 检查频率
            current_time = global_step + t
            if current_time % self.update_freq == 0:
                # 计算"惊喜"/梯度
                # 这里我们使用自监督重建代理：
                # 理想情况下，CMS 尝试预测 input_t 或重建特征。
                # 在这个演示中，我们将 FFN 输出视为输入的预测（类似自编码器）
                # 以生成梯度信号。
                
                loss = F.mse_loss(out_t, input_t) # 简单的局部目标
                
                # 手动梯度（为演示速度近似）
                grad_w1 = torch.autograd.grad(loss, curr_w1, retain_graph=True)[0]
                grad_w2 = torch.autograd.grad(loss, curr_w2, retain_graph=True)[0]
                
                # 应用更新（SGD）
                curr_w1 = curr_w1 - self.lr * grad_w1
                curr_w2 = curr_w2 - self.lr * grad_w2
                
                # 分离以防止 VRAM 爆炸（相当于截断 BPTT）
                curr_w1 = curr_w1.detach().requires_grad_(True)
                curr_w2 = curr_w2.detach().requires_grad_(True)

        return torch.stack(outputs, dim=1) + x # 残差连接


class ContinuumMemorySystem(nn.Module):
    """
    公式 30：具有不同频率的嵌套 MLP。
    输入 -> 快速 MLP -> 中等 MLP -> 慢速 MLP -> 输出
    """
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden = dim * expansion_factor
        
        # 层级 1：快速（每步更新）
        self.fast_mlp = CMSLayer(dim, hidden, update_freq=1, learning_rate=0.1)
        
        # 层级 2：慢速（每 16 步更新）
        # 代表"长期"知识存储
        self.slow_mlp = CMSLayer(dim, hidden, update_freq=16, learning_rate=0.01)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 思维链/计算链
        h = self.fast_mlp(x)
        out = self.slow_mlp(h)
        return self.norm(out)


class HOPEBlock(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.memory = HOPERecurrentMemory(dim, head_dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cms = ContinuumMemorySystem(dim)
        
    def forward(self, x):
        # 1. 工作记忆（类似注意力）
        h = x + self.memory(self.norm1(x))
        
        # 2. 连续记忆（类似 FFN）
        out = h + self.cms(self.norm2(h))
        return out


class HOPEModel(nn.Module):
    def __init__(self, vocab_size, dim, depth, head_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            HOPEBlock(dim, head_dim) for _ in range(depth)
        ])
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits


@torch.jit.script
def hope_cross_scan(k_enc, v_enc, q_dec, initial_state, eta, beta):
    """
    HOPE cross-attention 记忆更新的融合内核。
    
    k_enc, v_enc: (Batch, Seq_Enc, Head_Dim) - 来自 encoder
    q_dec: (Batch, Seq_Dec, Head_Dim) - 来自 decoder
    initial_state: (Batch, Head_Dim, Head_Dim)
    eta: 内部循环的学习率
    beta: 遗忘因子
    """
    b, t_enc, d = k_enc.shape
    t_dec = q_dec.shape[1]
    
    # 初始化状态（使用 encoder 的 k, v 建立记忆）
    state = initial_state  # (B, D, D)
    outputs = torch.jit.annotate(torch.Tensor, torch.zeros(b, t_dec, d, device=q_dec.device))
    
    # 第一步：用 encoder 的 k, v 建立记忆
    for i in range(t_enc):
        k_t = k_enc[:, i].unsqueeze(2)  # (B, D, 1)
        v_t = v_enc[:, i].unsqueeze(2)
        
        # 预测：v_pred = M * k
        v_pred = torch.bmm(state, k_t)
        
        # 误差：e = v - v_pred
        error = v_t - v_pred
        
        # 梯度/增量：delta = error * k.T
        delta = torch.bmm(error, k_t.transpose(1, 2))
        
        # 更新规则：M_new = Beta * M_old + Eta * Delta
        state = (beta * state) + (eta * delta)
    
    # 第二步：用 decoder 的 query 查询记忆
    for j in range(t_dec):
        q_t = q_dec[:, j].unsqueeze(2)  # (B, D, 1)
        # y = M * q
        y_t = torch.bmm(state, q_t)
        outputs[:, j] = y_t.squeeze(2)
    
    return outputs


class HOPECrossAttention(nn.Module):
    """
    HOPE cross-attention：query 来自 decoder，key/value 来自 encoder
    使用嵌套优化循环建立 encoder 的记忆，然后用 decoder query 查询
    """
    def __init__(self, dim, head_dim, learning_rate=0.1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.lr = learning_rate
        
        # Query 投影（来自 decoder）
        self.W_q = nn.Linear(dim, dim, bias=False)
        # Key/Value 投影（来自 encoder）
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # 可学习的内部循环参数
        self.eta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.5)
        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.95)
        
    def forward(self, q_dec, k_enc, v_enc, mask=None):
        """
        q_dec: [B, L_dec, D] - decoder 输入
        k_enc, v_enc: [B, L_enc, D] - encoder 输出
        mask: [B, 1, L_dec, L_enc] 或 [B, L_dec, L_enc] - cross-attention mask
        """
        B, L_dec, _ = q_dec.shape
        _, L_enc, _ = k_enc.shape
        
        # 归一化
        q_norm = self.norm_q(q_dec)
        k_norm = self.norm_kv(k_enc)
        v_norm = self.norm_kv(v_enc)
        
        # 投影并重塑头
        q = self.W_q(q_norm).view(B, L_dec, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_dec, D]
        k = self.W_k(k_norm).view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_enc, D]
        v = self.W_v(v_norm).view(B, L_enc, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_enc, D]
        
        # 归一化 K 以稳定外积投影
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        
        # 重塑以进行并行扫描：合并批次和头 -> (B*Heads, L, D)
        q_flat = q.reshape(B * self.num_heads, L_dec, self.head_dim)
        k_flat = k.reshape(B * self.num_heads, L_enc, self.head_dim)
        v_flat = v.reshape(B * self.num_heads, L_enc, self.head_dim)
        
        # 初始状态 M_0
        initial_state = torch.zeros(B * self.num_heads, self.head_dim, self.head_dim, device=q_dec.device)
        
        # 扩展元参数
        eta_flat = self.eta.repeat(B, 1, 1)
        beta_flat = self.beta.repeat(B, 1, 1)
        
        # 运行 JIT cross-attention 扫描
        out_flat = hope_cross_scan(k_flat, v_flat, q_flat, initial_state, eta_flat, beta_flat)
        
        # 重塑回 (B, L_dec, C)
        out = out_flat.view(B, self.num_heads, L_dec, self.head_dim).transpose(1, 2).reshape(B, L_dec, self.dim)
        
        # 应用 mask（如果需要）
        # 注意：HOPE 的记忆更新是顺序的，mask 主要用于 padding，可以在输出后应用
        # 这里我们简单返回，mask 的处理可以在外层完成
        
        return self.W_o(out)


class HOPEAttentionWithMask(nn.Module):
    """
    支持 mask 的 HOPE self-attention（用于 encoder 和 decoder self-attention）
    """
    def __init__(self, dim, head_dim, learning_rate=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.lr = learning_rate
        
        # 多头投影
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm = nn.LayerNorm(dim)
        
        # 可学习的内部循环参数
        self.eta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.5)
        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.95)
        
    def forward(self, x, mask=None):
        """
        x: [B, T, C]
        mask: [B, 1, T, T] 或 [B, T, T]，1=屏蔽，0=保留
        """
        B, T, C = x.shape
        x_norm = self.norm(x)
        
        # 投影并重塑头
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 归一化 K 以稳定外积投影
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        
        # 处理 mask：如果提供了 mask，需要在记忆更新时跳过被屏蔽的位置
        # 简化处理：在顺序扫描时检查 mask
        if mask is not None:
            # 确保 mask 是 [B, H, T, T] 格式
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
            if mask.size(1) == 1:
                mask = mask.expand(B, self.num_heads, T, T)  # [B, H, T, T]
            mask = mask.bool()  # 转换为 bool
        
        # 重塑以进行并行扫描：合并批次和头 -> (B*Heads, T, D)
        q_flat = q.reshape(B * self.num_heads, T, self.head_dim)
        k_flat = k.reshape(B * self.num_heads, T, self.head_dim)
        v_flat = v.reshape(B * self.num_heads, T, self.head_dim)
        
        # 初始状态 M_0
        initial_state = torch.zeros(B * self.num_heads, self.head_dim, self.head_dim, device=x.device)
        
        # 扩展元参数
        eta_flat = self.eta.repeat(B, 1, 1)
        beta_flat = self.beta.repeat(B, 1, 1)
        
        # 如果有 mask，使用简化的顺序处理
        # 为了内存效率，我们限制只处理当前可以访问的位置
        if mask is not None:
            # 简化处理：对于有 mask 的情况，按顺序处理，但只更新可访问的位置
            # 这避免了双重循环，减少内存占用
            outputs = []
            state = initial_state
            
            # 处理 mask 格式
            if mask.dim() == 3:
                mask_expanded = mask.unsqueeze(1).expand(B, self.num_heads, T, T)
            elif mask.dim() == 4:
                if mask.size(1) == 1:
                    mask_expanded = mask.expand(B, self.num_heads, T, T)
                else:
                    mask_expanded = mask
            else:
                mask_expanded = None
            
            # 顺序处理每个位置
            for i in range(T):
                # 查询记忆
                q_t = q_flat[:, i].unsqueeze(2)  # [B*H, D, 1]
                y_t = torch.bmm(state, q_t)
                outputs.append(y_t.squeeze(2))
                
                # 更新记忆：只使用当前可以访问的位置
                # 对于 look-ahead mask，只能访问 j <= i 的位置
                # 对于 padding mask，跳过被屏蔽的位置
                if mask_expanded is not None:
                    # 获取当前可以访问的位置（mask == False 表示可以访问）
                    # 只考虑 j <= i 的位置（look-ahead）
                    valid_positions = ~mask_expanded[:, :, i, :i+1].any(dim=0) if i > 0 else torch.ones(self.num_heads, 1, dtype=torch.bool, device=mask.device)
                    valid_positions = valid_positions.reshape(B * self.num_heads, i+1)
                else:
                    valid_positions = torch.ones(B * self.num_heads, i+1, dtype=torch.bool, device=x.device)
                
                # 批量更新所有有效位置
                if valid_positions.any():
                    # 只处理最后一个位置（简化版本，避免内存问题）
                    # 对于完整的实现，可以批量处理所有有效位置
                    j = i  # 只更新当前位置
                    k_t = k_flat[:, j].unsqueeze(2)  # [B*H, D, 1]
                    v_t = v_flat[:, j].unsqueeze(2)
                    
                    v_pred = torch.bmm(state, k_t)
                    error = v_t - v_pred
                    delta = torch.bmm(error, k_t.transpose(1, 2))
                    
                    # 更新状态
                    state = (beta_flat * state) + (eta_flat * delta)
            
            out_flat = torch.stack(outputs, dim=1)
        else:
            # 无 mask，使用 JIT 扫描（更高效）
            out_flat = hope_scan(k_flat, v_flat, q_flat, initial_state, eta_flat, beta_flat)
        
        # 重塑回 (B, T, C)
        out = out_flat.view(B, self.num_heads, T, self.head_dim).transpose(1, 2).reshape(B, T, C)
        
        return self.out_proj(out) + x


def toy_training_loop():
    """HOPE 模型的玩具训练示例。"""
    # 配置
    vocab_size = 100
    dim = 64
    depth = 2
    head_dim = 16
    seq_len = 32
    batch_size = 4
    
    # 初始化模型
    model = HOPEModel(vocab_size, dim, depth, head_dim)
    
    # HOPE 引入了内部优化循环。
    # 外部优化器（元学习器）训练初始化
    # 和内部模块的学习率。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Running HOPE Toy Task...")
    
    # 虚拟数据：复制任务（输入：A B C ... -> 目标：A B C ...）
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.roll(inputs, shifts=-1, dims=1) # 下一个 token 预测
    
    model.train()
    
    for step in range(100):
        optimizer.zero_grad()
        
        # 前向传播运行内部嵌套学习（内部优化器）
        logits = model(inputs)
        
        # 损失计算（标准交叉熵）
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")
        
    print("Done. Model runs successfully with nested optimization logic.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualization import plot_training_curves
    
    print("=" * 60)
    print("HOPE 模型演示")
    print("=" * 60)
    
    # 配置
    vocab_size = 100
    dim = 64
    depth = 2
    head_dim = 16
    seq_len = 32
    batch_size = 4
    num_steps = 50
    
    print(f"\n模型配置:")
    print(f"  - 词汇表大小: {vocab_size}")
    print(f"  - 模型维度: {dim}")
    print(f"  - 深度: {depth}")
    print(f"  - 头维度: {head_dim}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 批次大小: {batch_size}")
    
    # 初始化模型
    model = HOPEModel(vocab_size, dim, depth, head_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters())/1e3:.2f}K")
    
    # 生成数据
    torch.manual_seed(42)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.roll(inputs, shifts=-1, dims=1)
    
    print("\n开始训练...")
    model.train()
    loss_history = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"  步骤 {step+1:3d}/{num_steps}: 损失 = {loss.item():.4f}")
    
    print("\n训练完成！")
    
    # 可视化训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves({'训练损失': loss_history})
    
    # 演示 HOPE 更新规则
    print("\n" + "-" * 60)
    print("演示 HOPE 更新规则")
    print("-" * 60)
    
    # 创建一个简单的记忆状态
    head_dim = 8
    W = torch.randn(1, head_dim, head_dim) * 0.1
    x = torch.randn(1, head_dim, 1)
    grad_loss = torch.randn(1, head_dim, head_dim) * 0.01
    eta = 0.1
    
    W_old = W.clone()
    W_new = hope_update_rule(W, x, grad_loss, eta)
    
    print(f"  记忆矩阵形状: {W.shape}")
    print(f"  更新前记忆范数: {torch.norm(W_old).item():.4f}")
    print(f"  更新后记忆范数: {torch.norm(W_new).item():.4f}")
    print(f"  更新幅度: {torch.norm(W_new - W_old).item():.4f}")
    
    print("\n演示完成！")


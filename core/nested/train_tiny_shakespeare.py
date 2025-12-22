"""
Tiny Shakespeare 训练脚本

本脚本在 Tiny Shakespeare 数据集上训练 PracticalHOPE 模型。
包括数据加载、训练循环、文本生成和可视化工具。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns


# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 设置随机种子以确保可重现性
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)


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
    state = initial_state # (B, D, D)
    outputs = torch.jit.annotate(torch.Tensor, torch.zeros(b, t, d, device=k.device))
    
    # 遍历序列
    for i in range(t):
        k_t = k[:, i].unsqueeze(2) # (B, D, 1)
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


class HOPEAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        
        # 多头投影
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm = nn.LayerNorm(dim)
        
        # 可学习的内部循环参数（可广播）
        self.eta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.5)
        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 0.95)
        
    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.norm(x)
        
        # 投影并重塑头
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 归一化 K 以稳定外积投影
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        
        # 重塑以进行并行扫描：合并批次和头 -> (B*Heads, T, D)
        q_flat = q.reshape(B * self.num_heads, T, self.head_dim)
        k_flat = k.reshape(B * self.num_heads, T, self.head_dim)
        v_flat = v.reshape(B * self.num_heads, T, self.head_dim)
        
        # 初始状态 M_0（零初始化效果良好）
        initial_state = torch.zeros(B * self.num_heads, self.head_dim, self.head_dim, device=x.device)
        
        # 扩展元参数以进行扫描
        eta_flat = self.eta.repeat(B, 1, 1)
        beta_flat = self.beta.repeat(B, 1, 1)
        
        # 运行 JIT 扫描
        out_flat = hope_scan(k_flat, v_flat, q_flat, initial_state, eta_flat, beta_flat)
        
        # 重塑回 (B, T, C)
        out = out_flat.view(B, self.num_heads, T, self.head_dim).transpose(1, 2).reshape(B, T, C)
        
        return self.out_proj(out) + x


class CMSBlock(nn.Module):
    """
    简化的连续记忆系统。
    使用门控来模拟"快速"与"慢速"信息流的选择。
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # 门控机制控制信息流
        gate = torch.sigmoid(self.gate(x))
        h = self.act(self.w1(x))
        out = self.w2(h)
        
        return residual + (out * gate)


class PracticalHOPE(nn.Module):
    def __init__(self, vocab_size, dim=256, depth=4, head_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim) # 简单的位置编码
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                HOPEAttention(dim, head_dim),
                CMSBlock(dim, dim * 4)
            ])
            for _ in range(depth)
        ])
        
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 嵌入
        tok_emb = self.emb(idx)
        # 确保如果 T > 512，pos_emb 不会出错（在此演示中使用模运算以确保安全）
        safe_T = torch.arange(T, device=idx.device) % 512
        pos_emb = self.pos_emb(safe_T)
        x = tok_emb + pos_emb
        
        for attn, cms in self.layers:
            x = attn(x)
            x = cms(x)
            
        x = self.norm_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # 生成循环
        for _ in range(max_new_tokens):
            # 裁剪到上下文大小以防止索引错误
            idx_cond = idx[:, -256:] 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_tiny_shakespeare(file_path='datasets/tinyshakespeare.txt'):
    """下载并加载 Tiny Shakespeare 数据集。"""
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(file_path, 'r') as f:
        text = f.read()

    # 创建词汇表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocab Size: {vocab_size}")
    print(f"Total Characters: {len(text)}")

    # 映射
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 转换为张量
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, encode, decode, stoi


def get_batch(train_data, val_data, split, batch_size, block_size, device):
    """获取一批数据。"""
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))
    x = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=50):
    """估计训练集和验证集上的损失。"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def visualize_surprise(model, text_str, stoi, encode):
    """可视化 HOPE 模型的'惊喜'信号。"""
    model.eval()
    
    # 1. 准备输入
    device = next(model.parameters()).device
    
    # 检查不在词汇表中的字符以防止崩溃
    # （Tiny Shakespeare 可能没有某些特殊字符）
    valid_chars = [c for c in text_str if c in stoi]
    if len(valid_chars) < len(text_str):
        print("Warning: Some characters were not in the vocabulary and were skipped.")
    
    encoded_ids = encode("".join(valid_chars))
    idx = torch.tensor(encoded_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. 钩入模型以捕获"惊喜"（误差）
    target_layer = model.layers[0][0] 
    
    with torch.no_grad():
        # --- 修复：仅解包批次和时间 (B, T) ---
        B, T = idx.shape 
        
        # 嵌入
        tok_emb = model.emb(idx)
        safe_T = torch.arange(T, device=device) % 512
        pos_emb = model.pos_emb(safe_T)
        x = tok_emb + pos_emb
        x_norm = target_layer.norm(x)
        
        # 投影
        q = target_layer.q_proj(x_norm).view(B, T, target_layer.num_heads, target_layer.head_dim).transpose(1, 2)
        k = target_layer.k_proj(x_norm).view(B, T, target_layer.num_heads, target_layer.head_dim).transpose(1, 2)
        v = target_layer.v_proj(x_norm).view(B, T, target_layer.num_heads, target_layer.head_dim).transpose(1, 2)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)

        # Python 端模拟扫描以捕获'误差'
        head_idx = 0
        k_h = k[0, head_idx] # (T, D)
        v_h = v[0, head_idx]
        q_h = q[0, head_idx]
        
        state = torch.zeros(target_layer.head_dim, target_layer.head_dim, device=device)
        surprises = []
        
        eta = target_layer.eta[head_idx, 0, 0]
        beta = target_layer.beta[head_idx, 0, 0]
        
        for t in range(T):
            kt = k_h[t].unsqueeze(1)
            vt = v_h[t].unsqueeze(1)
            
            # 预测
            v_pred = torch.mm(state, kt)
            
            # 计算误差幅度（惊喜）
            error = vt - v_pred
            surprise_mag = torch.norm(error).item()
            surprises.append(surprise_mag)
            
            # 更新（复制 JIT 内核逻辑）
            delta = torch.mm(error, kt.t())
            state = (beta * state) + (eta * delta)

    # 3. 绘图
    chars = list("".join(valid_chars))
    plt.figure(figsize=(15, 3))
    sns.heatmap([surprises], xticklabels=chars, yticklabels=['Surprise'], cmap="viridis", cbar=True)
    plt.title(f"Nested Learning 'Surprise' Signal (Layer 0, Head 0)")
    plt.show()


def main():
    """主训练函数。"""
    # 超参数
    BATCH_SIZE = 64
    BLOCK_SIZE = 128  # 上下文长度
    MAX_ITERS = 100  # 训练步数
    EVAL_INTERVAL = 200
    LEARNING_RATE = 3e-4
    DIM = 192
    DEPTH = 6
    HEAD_DIM = 32

    # 加载数据
    train_data, val_data, vocab_size, encode, decode, stoi = load_tiny_shakespeare()

    # 初始化模型
    model = PracticalHOPE(vocab_size, dim=DIM, depth=DEPTH, head_dim=HEAD_DIM)
    model.to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    start_time = time.time()
    print("Starting training...")

    for iter in range(MAX_ITERS):
        # 每隔几步，评估训练集和验证集上的损失
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, BATCH_SIZE, BLOCK_SIZE, device)
            print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            
        # 采样一批数据
        xb, yb = get_batch(train_data, val_data, 'train', BATCH_SIZE, BLOCK_SIZE, device)
        
        # 评估损失
        _, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 裁剪梯度（对循环/HOPE 模型很重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 生成文本
    print("--- Generating Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out_ids = model.generate(context, max_new_tokens=500)
    print(decode(out_ids[0].tolist()))

    # 可视化惊喜信号
    print("\n--- Visualizing Surprise Signal ---")
    sample_text = "ROMEO: \nI surely hope this works."
    visualize_surprise(model, sample_text, stoi, encode)


if __name__ == "__main__":
    main()


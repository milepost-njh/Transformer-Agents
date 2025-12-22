"""
CMS (Continuum Memory System) 实现

本模块实现了 Nested Learning 论文中描述的 Continuum Memory System。
CMS 使用多个具有不同更新频率的记忆层级来捕获不同时间尺度的信息。
"""

import torch
import torch.nn as nn


class CMSLayer(nn.Module):
    """
    单个记忆层级:
      mem_t = alpha * mem_{t-1} + (1 - alpha) * MLP(x_t)
    """
    def __init__(self, dim, alpha):
        super().__init__()
        self.alpha = alpha
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, mem):
        # 计算新的更新值
        new_val = self.mlp(x)      # [B, D]

        # 快速初始化（第一个token）
        if mem is None:
            mem = new_val
        else:
            mem = self.alpha * mem + (1 - self.alpha) * new_val

        return mem


class CMS(nn.Module):
    """
    具有 L 个记忆层级的 CMS。每个层级都有自己的 alpha 和 MLP。
    输出：记忆层级的平均值（论文默认设置）。
    """
    def __init__(self, dim, levels=3, alphas=None):
        super().__init__()
        if alphas is None:
            # 论文默认设置：最慢的在前
            # 例如：[0.9, 0.7, 0.3]
            alphas = [0.2 + 0.6 * (i/(levels-1)) for i in range(levels)]
            alphas = list(reversed(alphas))

        assert len(alphas) == levels

        self.levels = nn.ModuleList([
            CMSLayer(dim, alpha=alphas[i]) for i in range(levels)
        ])

    def forward(self, x, memories):
        """
        x:          [B, D]
        memories:   list of length L (None or tensors [B, D])
        Return: (output, new_memories)
        """
        new_mems = []
        outs = []

        for i, layer in enumerate(self.levels):
            mem_i = None if memories is None else memories[i]
            new_mem = layer(x, mem_i)
            new_mems.append(new_mem)
            outs.append(new_mem)

        # 论文：对输出求平均
        final = sum(outs) / len(outs)
        return final, new_mems


class CMSForSequence(nn.Module):
    """
    CMS 的序列版本，用于 Transformer 的 FFN 层
    优化版本：使用批量处理减少内存占用
    """
    def __init__(self, dim, levels=3, alphas=None, expansion_factor=4):
        super().__init__()
        self.dim = dim
        self.levels = levels
        
        # 使用简化的 CMS：直接使用 MLP 而不是逐时间步的 CMS
        # 这样可以避免内存问题，同时保持类似的效果
        hidden_dim = dim * expansion_factor
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 可选的 CMS 层级（如果内存允许）
        self.use_cms = levels > 0
        if self.use_cms:
            self.cms = CMS(dim, levels=min(levels, 2), alphas=alphas)  # 限制层级数以减少内存
        
    def forward(self, x):
        """
        x: [B, L, D] - 序列输入
        返回: [B, L, D] - 序列输出
        """
        B, L, D = x.shape
        
        # 先通过标准 MLP
        mlp_out = self.mlp(x)  # [B, L, D]
        
        # 如果使用 CMS，逐时间步处理（但限制内存使用）
        if self.use_cms and L <= 128:  # 只对短序列使用 CMS
            memories = None
            outputs = []
            
            # 逐时间步处理
            for t in range(L):
                x_t = mlp_out[:, t, :]  # [B, D]
                cms_out, memories = self.cms(x_t, memories)
                outputs.append(cms_out)
            
            out = torch.stack(outputs, dim=1)  # [B, L, D]
        else:
            # 对于长序列，直接使用 MLP 输出
            out = mlp_out
        
        # 残差连接
        return out + x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualization import plot_cms_memory_evolution
    
    print("=" * 60)
    print("CMS (Continuum Memory System) 演示")
    print("=" * 60)
    
    # 配置
    dim = 32
    levels = 3
    seq_len = 50
    batch_size = 1
    
    # 创建 CMS 模型
    cms = CMS(dim=dim, levels=levels)
    print(f"\n创建 CMS 模型:")
    print(f"  - 维度: {dim}")
    print(f"  - 层级数: {levels}")
    alphas = [layer.alpha for layer in cms.levels]
    print(f"  - Alpha 值: {alphas}")
    
    # 生成模拟输入序列
    torch.manual_seed(42)
    inputs = torch.randn(seq_len, batch_size, dim)
    
    # 运行 CMS 并记录记忆状态
    memories = None
    memory_history = []
    
    print(f"\n处理 {seq_len} 个时间步的输入...")
    for t in range(seq_len):
        x_t = inputs[t]  # [B, D]
        output, memories = cms(x_t, memories)
        
        # 记录当前记忆状态
        mem_state = torch.stack([mem.detach() for mem in memories])  # [levels, B, D]
        memory_history.append(mem_state.squeeze(1))  # [levels, D]
    
    print("处理完成！")
    
    # 可视化记忆演化
    print("\n绘制记忆演化图...")
    plot_cms_memory_evolution(memory_history, alphas)
    
    # 打印一些统计信息
    print("\n记忆统计信息:")
    final_memories = memory_history[-1]  # [levels, dim]
    for level in range(levels):
        mem_norm = torch.norm(final_memories[level]).item()
        print(f"  层级 {level} (α={alphas[level]:.2f}): 最终记忆强度 = {mem_norm:.4f}")
    
    print("\n演示完成！")


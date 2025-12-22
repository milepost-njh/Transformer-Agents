"""
自修改 MLP 实现

本模块实现了可以根据快速上下文信号修改自身权重的 MLP，
如 Nested Learning 论文中所述。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModifyingMLP(nn.Module):
    """
    Implements W' = W + u(h) @ v(h)^T (rank-1 update)
    x:        [B, D]
    fast_h:   [B, D]   (fast context from CMS or attention)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # base slow weight
        self.base = nn.Linear(dim, dim)

        # fast update projections
        self.to_u = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x, fast_h):
        """
        x:        [B, D]
        fast_h:   [B, D]   (the fast context signal)
        """
        # 计算低秩更新向量
        u = self.to_u(fast_h)      # [B, D]
        v = self.to_v(fast_h)      # [B, D]

        # 跨批次平均更新（简单、稳定）
        u_mean = u.mean(dim=0)     # [D]
        v_mean = v.mean(dim=0)     # [D]

        # 外积得到 ΔW（rank-1）
        delta = torch.ger(u_mean, v_mean)   # [D, D]

        # 应用修改后的权重：W' = W + ΔW
        W_mod = self.base.weight + delta

        out = F.linear(x, W_mod, self.base.bias)
        return out, delta.detach()   # 返回 ΔW 用于检查


class SelfModifyingMLP_RankK(nn.Module):
    """
    Rank-k self modifying MLP:
    ΔW = sum_i u_i ⊗ v_i
    """
    def __init__(self, dim, rank=4):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.base = nn.Linear(dim, dim)

        # 投影到 rank*k 向量 → 重塑为 (k, dim)
        self.to_u = nn.Linear(dim, dim * rank, bias=False)
        self.to_v = nn.Linear(dim, dim * rank, bias=False)

    def forward(self, x, fast_h):
        B = x.size(0)

        u = self.to_u(fast_h).view(B, self.rank, self.dim)
        v = self.to_v(fast_h).view(B, self.rank, self.dim)

        # 批次平均
        u_mean = u.mean(dim=0)    # [k, D]
        v_mean = v.mean(dim=0)    # [k, D]

        # 计算 ΔW
        delta = torch.einsum("kd,ke->de", u_mean, v_mean)

        W_mod = self.base.weight + delta
        return F.linear(x, W_mod, self.base.bias), delta.detach()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from visualization import plot_weight_update_magnitude
    
    print("=" * 60)
    print("自修改 MLP 演示")
    print("=" * 60)
    
    # 配置
    dim = 32
    batch_size = 4
    num_updates = 20
    
    # 创建模型
    print(f"\n创建自修改 MLP 模型:")
    print(f"  - 维度: {dim}")
    print(f"  - 批次大小: {batch_size}")
    
    # 测试 rank-1 版本
    print("\n" + "-" * 60)
    print("测试 Rank-1 版本")
    print("-" * 60)
    model_rank1 = SelfModifyingMLP(dim=dim)
    
    # 记录权重更新历史
    delta_history_rank1 = []
    
    torch.manual_seed(42)
    for i in range(num_updates):
        # 生成随机输入和快速上下文信号
        x = torch.randn(batch_size, dim)
        fast_h = torch.randn(batch_size, dim)
        
        # 前向传播
        output, delta = model_rank1(x, fast_h)
        delta_history_rank1.append(delta)
        
        if i < 3 or i == num_updates - 1:
            delta_norm = torch.norm(delta).item()
            print(f"  更新 {i+1:2d}: ΔW 幅度 = {delta_norm:.4f}")
    
    # 测试 rank-k 版本
    print("\n" + "-" * 60)
    print("测试 Rank-K 版本 (k=4)")
    print("-" * 60)
    model_rankk = SelfModifyingMLP_RankK(dim=dim, rank=4)
    
    delta_history_rankk = []
    
    torch.manual_seed(42)
    for i in range(num_updates):
        x = torch.randn(batch_size, dim)
        fast_h = torch.randn(batch_size, dim)
        
        output, delta = model_rankk(x, fast_h)
        delta_history_rankk.append(delta)
        
        if i < 3 or i == num_updates - 1:
            delta_norm = torch.norm(delta).item()
            print(f"  更新 {i+1:2d}: ΔW 幅度 = {delta_norm:.4f}")
    
    # 可视化权重更新幅度
    print("\n绘制权重更新幅度变化...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Rank-1
    magnitudes_rank1 = [torch.norm(d).item() for d in delta_history_rank1]
    axes[0].plot(magnitudes_rank1, 'b-', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('更新次数', fontsize=12)
    axes[0].set_ylabel('权重更新幅度', fontsize=12)
    axes[0].set_title('Rank-1 权重更新幅度', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Rank-K
    magnitudes_rankk = [torch.norm(d).item() for d in delta_history_rankk]
    axes[1].plot(magnitudes_rankk, 'r-', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('更新次数', fontsize=12)
    axes[1].set_ylabel('权重更新幅度', fontsize=12)
    axes[1].set_title('Rank-K 权重更新幅度', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 比较两种方法
    print("\n对比分析:")
    print(f"  Rank-1 平均更新幅度: {np.mean(magnitudes_rank1):.4f}")
    print(f"  Rank-K 平均更新幅度: {np.mean(magnitudes_rankk):.4f}")
    
    print("\n演示完成！")

"""
CMS 交互式学习版本

这个文件提供了更详细的调试和可视化功能，帮助理解 CMS 的工作原理。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from cms import CMS, CMSLayer
from visualization import plot_cms_memory_evolution


def visualize_cms_step_by_step():
    """逐步可视化 CMS 的工作过程"""
    print("=" * 60)
    print("CMS 逐步可视化演示")
    print("=" * 60)
    
    # 配置
    dim = 8  # 使用较小的维度便于可视化
    levels = 3
    seq_len = 10
    
    # 创建模型
    cms = CMS(dim=dim, levels=levels)
    alphas = [layer.alpha for layer in cms.levels]
    
    print(f"\n模型配置:")
    print(f"  维度: {dim}")
    print(f"  层级数: {levels}")
    print(f"  Alpha 值: {alphas}")
    
    # 生成输入
    torch.manual_seed(42)
    inputs = torch.randn(seq_len, 1, dim)
    
    memories = None
    memory_history = []
    
    print(f"\n逐步处理 {seq_len} 个时间步:")
    print("-" * 60)
    
    for t in range(seq_len):
        x_t = inputs[t]
        print(f"\n时间步 {t}:")
        print(f"  输入 x_t 形状: {x_t.shape}")
        print(f"  输入 x_t 范数: {torch.norm(x_t).item():.4f}")
        
        if memories is not None:
            print(f"  当前记忆状态:")
            for i, mem in enumerate(memories):
                mem_norm = torch.norm(mem).item()
                print(f"    Level {i} (α={alphas[i]:.2f}): 范数={mem_norm:.4f}")
        
        # 前向传播
        output, memories = cms(x_t, memories)
        
        # 记录状态
        mem_state = torch.stack([mem.detach() for mem in memories])
        memory_history.append(mem_state.squeeze(1))
        
        print(f"  输出形状: {output.shape}")
        print(f"  输出范数: {torch.norm(output).item():.4f}")
        
        if t < 3:  # 只详细显示前3步
            print(f"  更新后的记忆状态:")
            for i, mem in enumerate(memories):
                mem_norm = torch.norm(mem).item()
                print(f"    Level {i}: 范数={mem_norm:.4f}")
    
    # 可视化
    print("\n" + "=" * 60)
    print("绘制记忆演化图...")
    plot_cms_memory_evolution(memory_history, alphas)
    
    return cms, memory_history, alphas


def visualize_cms_architecture():
    """可视化 CMS 架构"""
    print("\n" + "=" * 60)
    print("CMS 架构可视化")
    print("=" * 60)
    
    cms = CMS(dim=32, levels=3)
    alphas = [layer.alpha for layer in cms.levels]
    
    print("\nCMS 架构:")
    print("  CMS")
    print("  ├── Level 0 (α={:.2f}) - 慢速记忆".format(alphas[0]))
    print("  │   └── MLP: Linear -> ReLU -> Linear")
    print("  ├── Level 1 (α={:.2f}) - 中速记忆".format(alphas[1]))
    print("  │   └── MLP: Linear -> ReLU -> Linear")
    print("  └── Level 2 (α={:.2f}) - 快速记忆".format(alphas[2]))
    print("      └── MLP: Linear -> ReLU -> Linear")
    print("\n输出: 所有层级记忆的平均值")
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in cms.parameters())
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    for i, layer in enumerate(cms.levels):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  Level {i}: {layer_params:,} 参数")


def compare_alpha_effects():
    """比较不同 alpha 值的效果"""
    print("\n" + "=" * 60)
    print("Alpha 值效果对比")
    print("=" * 60)
    
    dim = 16
    seq_len = 20
    
    # 测试不同的 alpha 值
    alpha_configs = [
        ([0.9, 0.7, 0.5], "高 alpha（慢速更新）"),
        ([0.5, 0.3, 0.1], "低 alpha（快速更新）"),
        ([0.8, 0.5, 0.2], "混合 alpha（默认）"),
    ]
    
    torch.manual_seed(42)
    inputs = torch.randn(seq_len, 1, dim)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (alphas, title) in enumerate(alpha_configs):
        cms = CMS(dim=dim, levels=len(alphas), alphas=alphas)
        memories = None
        memory_history = []
        
        for t in range(seq_len):
            x_t = inputs[t]
            output, memories = cms(x_t, memories)
            mem_state = torch.stack([mem.detach() for mem in memories])
            memory_history.append(mem_state.squeeze(1))
        
        # 计算记忆强度
        mem_array = torch.stack(memory_history).cpu().numpy()
        memory_norms = np.linalg.norm(mem_array, axis=2)
        
        ax = axes[idx]
        for level in range(len(alphas)):
            ax.plot(memory_norms[:, level], 
                   label=f'Level {level} (α={alphas[level]:.2f})',
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('记忆强度', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 逐步可视化
    cms, memory_history, alphas = visualize_cms_step_by_step()
    
    # 2. 架构可视化
    visualize_cms_architecture()
    
    # 3. Alpha 值效果对比
    compare_alpha_effects()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


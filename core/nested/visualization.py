"""
可视化工具模块

提供各种可视化函数用于展示 Nested Learning 各组件的动态行为。
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict, Optional
import platform
import warnings

# 配置 matplotlib 支持中文显示
def _setup_chinese_font():
    """配置 matplotlib 以支持中文显示"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS 系统字体列表（按优先级）
        fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Helvetica Neue']
    elif system == 'Windows':  # Windows
        # Windows 系统字体
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    else:  # Linux 或其他
        # Linux 常用中文字体
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 尝试设置字体
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 静默测试字体是否可用（不显示警告）
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=10)
            plt.close(fig)
    except Exception:
        # 如果测试失败，使用默认字体（可能会显示方框，但不会崩溃）
        pass

# 初始化时配置字体
_setup_chinese_font()


def plot_cms_memory_evolution(memory_history: List[torch.Tensor], alphas: List[float], 
                               save_path: Optional[str] = None):
    """
    可视化 CMS 多个记忆层级的演化过程。
    
    参数:
        memory_history: 每个时间步的记忆状态列表，每个元素是 [levels, dim] 的张量
        alphas: 每个层级的 alpha 值列表
        save_path: 保存路径（可选）
    """
    if not memory_history:
        print("记忆历史为空，无法可视化")
        return
    
    # 转换为 numpy
    mem_array = torch.stack(memory_history).cpu().numpy()  # [time_steps, levels, dim]
    time_steps, levels, dim = mem_array.shape
    
    # 计算每个层级的平均记忆强度（L2 范数）
    memory_norms = np.linalg.norm(mem_array, axis=2)  # [time_steps, levels]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制每个层级的记忆强度
    for level in range(levels):
        ax.plot(memory_norms[:, level], label=f'Level {level} (α={alphas[level]:.2f})', 
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('记忆强度 (L2 范数)', fontsize=12)
    ax.set_title('CMS 多层级记忆演化', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def animate_cms_memory(memory_history: List[torch.Tensor], alphas: List[float], 
                       interval: int = 100):
    """
    动态动画展示 CMS 记忆演化。
    
    参数:
        memory_history: 每个时间步的记忆状态列表
        alphas: 每个层级的 alpha 值列表
        interval: 动画帧间隔（毫秒）
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("警告: 需要 sklearn 库来运行动画功能。请安装: pip install scikit-learn")
        print("将使用简化的静态可视化代替。")
        plot_cms_memory_evolution(memory_history, alphas)
        return None
    
    if not memory_history:
        print("记忆历史为空，无法创建动画")
        return None
    
    mem_array = torch.stack(memory_history).cpu().numpy()
    time_steps, levels, dim = mem_array.shape
    
    # 使用 PCA 降维到 2D 以便可视化
    pca = PCA(n_components=2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 为每个层级准备颜色
    colors = plt.cm.viridis(np.linspace(0, 1, levels))
    
    # 初始化散点图
    scatters = []
    for level in range(levels):
        scatter = ax.scatter([], [], c=[colors[level]], s=50, alpha=0.6,
                             label=f'Level {level} (α={alphas[level]:.2f})')
        scatters.append(scatter)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('PCA 第一主成分', fontsize=12)
    ax.set_ylabel('PCA 第二主成分', fontsize=12)
    ax.set_title('CMS 记忆演化动画', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 对所有数据进行 PCA 拟合
    all_data = mem_array.reshape(-1, dim)
    pca.fit(all_data)
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('PCA 第一主成分', fontsize=12)
        ax.set_ylabel('PCA 第二主成分', fontsize=12)
        ax.set_title(f'CMS 记忆演化动画 (时间步: {frame}/{time_steps-1})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 绘制当前时间步的所有层级
        for level in range(levels):
            mem_2d = pca.transform(mem_array[frame, level:level+1])
            ax.scatter(mem_2d[0, 0], mem_2d[0, 1], c=[colors[level]], s=100, 
                      alpha=0.8, label=f'Level {level} (α={alphas[level]:.2f})')
        
        # 绘制历史轨迹
        for level in range(levels):
            history_2d = pca.transform(mem_array[:frame+1, level])
            ax.plot(history_2d[:, 0], history_2d[:, 1], c=colors[level], 
                   alpha=0.3, linewidth=1)
        
        ax.legend()
    
    anim = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                   interval=interval, repeat=True)
    plt.show()
    return anim


def plot_optimizer_convergence(param_history: Dict[str, List[float]], 
                               loss_history: Optional[List[float]] = None,
                               save_path: Optional[str] = None):
    """
    可视化优化器的参数收敛过程。
    
    参数:
        param_history: 参数字典，键为优化器名称，值为参数值历史列表
        loss_history: 损失历史（可选）
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2 if loss_history else 1, figsize=(15, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # 绘制参数收敛
    ax = axes[0]
    for name, history in param_history.items():
        ax.plot(history, label=name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('参数值', fontsize=12)
    ax.set_title('优化器参数收敛对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制损失（如果有）
    if loss_history:
        ax2 = axes[1]
        ax2.plot(loss_history, 'r-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('损失值', fontsize=12)
        ax2.set_title('损失变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_weight_update_magnitude(delta_history: List[torch.Tensor], 
                                save_path: Optional[str] = None):
    """
    可视化自修改 MLP 的权重更新幅度。
    
    参数:
        delta_history: 权重更新历史列表，每个元素是 ΔW 张量
        save_path: 保存路径（可选）
    """
    if not delta_history:
        print("更新历史为空，无法可视化")
        return
    
    # 计算每次更新的幅度（Frobenius 范数）
    magnitudes = [torch.norm(delta).item() for delta in delta_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(magnitudes, 'b-', linewidth=2, alpha=0.8)
    ax.set_xlabel('更新次数', fontsize=12)
    ax.set_ylabel('权重更新幅度 (Frobenius 范数)', fontsize=12)
    ax.set_title('自修改 MLP 权重更新幅度变化', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_hope_memory_state(memory_states: List[torch.Tensor], 
                          save_path: Optional[str] = None):
    """
    可视化 HOPE 记忆状态矩阵。
    
    参数:
        memory_states: 记忆状态列表，每个元素是 [head_dim, head_dim] 的矩阵
        save_path: 保存路径（可选）
    """
    if not memory_states:
        print("记忆状态为空，无法可视化")
        return
    
    num_states = len(memory_states)
    cols = min(5, num_states)
    rows = (num_states + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, mem_state in enumerate(memory_states):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        mem_np = mem_state.cpu().numpy()
        im = ax.imshow(mem_np, cmap='viridis', aspect='auto')
        ax.set_title(f'时间步 {idx}', fontsize=10)
        plt.colorbar(im, ax=ax)
    
    # 隐藏多余的子图
    for idx in range(num_states, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle('HOPE 记忆状态矩阵演化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_curves(losses: Dict[str, List[float]], 
                        save_path: Optional[str] = None):
    """
    绘制训练曲线。
    
    参数:
        losses: 损失字典，键为数据集名称（如 'train', 'val'），值为损失列表
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, loss_list in losses.items():
        ax.plot(loss_list, label=name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title('训练曲线', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


"""
嵌套优化器实现

本模块实现了各种被视为嵌套学习系统的优化器，
其中优化器被重新解释为具有不同更新规则的记忆系统。
"""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Dict


class GDMemory:
    """将梯度下降视为简单的记忆系统。"""
    def __init__(self, lr: float = 1e-2):
        self.lr = float(lr)

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return param - self.lr * grad


class MomentumMemory:
    """将动量视为关联记忆（每个参数的状态）。"""
    def __init__(self, lr: float = 1e-2, beta: float = 0.9, device=None):
        self.lr = float(lr)
        self.beta = float(beta)
        self.state: Dict[int, torch.Tensor] = {}
        self.device = device

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        pid = id(param)
        if pid not in self.state:
            self.state[pid] = torch.zeros_like(grad, device=grad.device)
        m = self.state[pid]
        m = self.beta * m + (1.0 - self.beta) * grad
        self.state[pid] = m
        return param - self.lr * m


class AdamMemory:
    """将 Adam 重新解释为每个参数的多级记忆 (m, v)。"""
    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}
        self.t: Dict[int, int] = {}

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        pid = id(param)
        if pid not in self.state:
            self.state[pid] = {
                'm': torch.zeros_like(grad, device=grad.device),
                'v': torch.zeros_like(grad, device=grad.device)
            }
            self.t[pid] = 0
        s = self.state[pid]
        self.t[pid] += 1
        s['m'] = self.b1 * s['m'] + (1 - self.b1) * grad
        s['v'] = self.b2 * s['v'] + (1 - self.b2) * (grad * grad)
        t = self.t[pid]
        m_hat = s['m'] / (1 - self.b1 ** t)
        v_hat = s['v'] / (1 - self.b2 ** t)
        update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return param - update


class DeepMomentumMemory(nn.Module):
    """
    深度动量：动量记忆是一个 MLP，它将传入的梯度（或梯度特征）
    映射到类似动量的更新。这增加了表示能力。
    此类是一个 Module，因此如果需要可以联合训练（需要接入更高级的训练循环）。
    """
    def __init__(self, dim, hidden=64, lr=1e-2, beta=0.9):
        super().__init__()
        self.lr = float(lr)
        self.beta = float(beta)
        # 小型 MLP 用于将梯度"压缩"成学习到的动量信号
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        # 每个参数的运行记忆将存储在由 id(param) 作为键的 Python 字典中
        self.state: Dict[int, torch.Tensor] = {}

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        grad 的形状必须是 (N,) 展平或与 param 形状相同；为简单起见，我们展平 param 和 grad，
        通过 mlp（因此 mlp 的维度必须匹配展平后的大小）或者可以对每个元素应用 mlp（1D 卷积）。
        这里我们操作 param.view(-1) 用于演示（仅适用于小参数）。
        """
        pid = id(param)
        flat_grad = grad.view(-1)
        # 如果 mlp 输入/输出维度不匹配，我们回退到逐元素变换
        if flat_grad.shape[0] != self.mlp[0].in_features:
            # 安全回退：逐元素单层学习缩放器（相同形状的向量）
            # 创建或获取作为状态一部分存储的每个参数的线性缩放向量（不理想但简单）
            if pid not in self.state:
                self.state[pid] = torch.zeros_like(flat_grad, device=flat_grad.device)
            # 通过简单的非线性变换（逐元素 tanh）将梯度映射到动量
            learned = torch.tanh(flat_grad)  # 占位符压缩
            m = self.beta * self.state[pid] + (1.0 - self.beta) * learned
            self.state[pid] = m
            new_flat = flat_grad - self.lr * m
            return new_flat.view_as(param)
        else:
            # MLP 路径：产生学习到的动量向量
            g_in = flat_grad.unsqueeze(0)  # [1, N]
            m_pred = self.mlp(g_in).squeeze(0)  # [N]
            if pid not in self.state:
                self.state[pid] = torch.zeros_like(m_pred, device=m_pred.device)
            m = self.beta * self.state[pid] + (1.0 - self.beta) * m_pred
            self.state[pid] = m
            new_flat = flat_grad - self.lr * m
            return new_flat.view_as(param)


class PreconditionedMomentumMemory:
    """
    在存储之前对梯度应用可学习或提供的预条件器 P 的动量记忆。
    为简单起见，我们展示对角预条件器（向量），但也可以是矩阵或低秩。
    """
    def __init__(self, lr=1e-2, beta=0.9, preconditioner: Optional[Iterable[float]]=None):
        self.lr = float(lr)
        self.beta = float(beta)
        self.state = {}
        # 预条件器可以是标量或每个参数的函数；这里我们存储可调用对象或标量
        self.pre = preconditioner

    def _apply_pre(self, grad: torch.Tensor) -> torch.Tensor:
        if self.pre is None:
            return grad
        # 如果 pre 是可调用的，用 grad 调用；如果是标量，则相乘
        if callable(self.pre):
            return self.pre(grad)
        else:
            return grad * float(self.pre)

    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        grad_pre = self._apply_pre(grad)
        pid = id(param)
        if pid not in self.state:
            self.state[pid] = torch.zeros_like(grad_pre)
        m = self.state[pid]
        m = self.beta * m + (1.0 - self.beta) * grad_pre
        self.state[pid] = m
        return param - self.lr * m


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from visualization import plot_optimizer_convergence
    
    print("=" * 60)
    print("嵌套优化器演示")
    print("=" * 60)
    
    # 定义一个简单的优化问题：最小化 f(x) = x^2
    def loss_fn(x):
        return x ** 2
    
    def grad_fn(x):
        return 2 * x
    
    # 初始化参数
    x0 = torch.tensor(5.0, requires_grad=True)
    target = 0.0  # 最优解
    
    # 创建不同的优化器
    optimizers = {
        'GD': GDMemory(lr=0.1),
        'Momentum': MomentumMemory(lr=0.1, beta=0.9),
        'Adam': AdamMemory(lr=0.1),
    }
    
    # 运行优化并记录历史
    num_iterations = 50
    param_history = {name: [] for name in optimizers.keys()}
    loss_history = []
    
    print(f"\n优化问题: f(x) = x², 初始值 x0 = {x0.item():.2f}")
    print(f"运行 {num_iterations} 次迭代...\n")
    
    for name, opt in optimizers.items():
        x = x0.clone()
        param_history[name] = []
        
        for i in range(num_iterations):
            # 计算梯度
            loss = loss_fn(x)
            grad = grad_fn(x)
            
            # 更新参数
            x_new = opt.step(x, grad)
            x = x_new.detach().requires_grad_(True)
            
            # 记录历史
            param_history[name].append(x.item())
            
            if i == 0:
                print(f"{name:12s}: 初始损失 = {loss.item():.4f}, x = {x.item():.4f}")
    
    # 打印最终结果
    print("\n最终结果:")
    for name in optimizers.keys():
        final_x = param_history[name][-1]
        final_loss = final_x ** 2
        print(f"{name:12s}: x = {final_x:.6f}, 损失 = {final_loss:.6f}")
    
    # 可视化收敛过程
    print("\n绘制收敛曲线...")
    plot_optimizer_convergence(param_history)
    
    print("\n演示完成！")

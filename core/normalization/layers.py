# -*- coding: utf-8 -*-
"""
归一化层实现

包含各种归一化层的实现，如 RMSNorm、LayerNorm 等
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm 实现，替代 LayerNorm 以获得更好的训练稳定性
    
    RMSNorm 使用均方根归一化，相比 LayerNorm 具有以下优势：
    - 更好的数值稳定性
    - 更快的计算速度
    - 更适合大型模型训练
    
    Args:
        hidden_size (int): 隐藏层维度
        eps (float): 防止除零的小常数，默认为 1e-6
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        前向传播
        
        Args:
            hidden_states (torch.Tensor): 输入张量，形状为 [..., hidden_size]
            
        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    """
    标准的 LayerNorm 实现
    
    保留此实现以兼容现有代码或作为对比
    
    Args:
        hidden_size (int): 隐藏层维度
        eps (float): 防止除零的小常数，默认为 1e-6
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        前向传播
        
        Args:
            hidden_states (torch.Tensor): 输入张量，形状为 [..., hidden_size]
            
        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states + self.bias).to(input_dtype)


def get_normalization_layer(norm_type: str, hidden_size: int, eps: float = 1e-6):
    """
    工厂函数，根据类型创建归一化层
    
    Args:
        norm_type (str): 归一化类型，支持 'rmsnorm' 和 'layernorm'
        hidden_size (int): 隐藏层维度
        eps (float): 防止除零的小常数，默认为 1e-6
        
    Returns:
        nn.Module: 归一化层实例
        
    Raises:
        ValueError: 当 norm_type 不支持时抛出
    """
    norm_type = norm_type.lower()
    if norm_type == 'rmsnorm':
        return RMSNorm(hidden_size, eps)
    elif norm_type == 'layernorm':
        return LayerNorm(hidden_size, eps)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}. "
                        f"Supported types: 'rmsnorm', 'layernorm'")


__all__ = ['RMSNorm', 'LayerNorm', 'get_normalization_layer']

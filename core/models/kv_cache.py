#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KV-Cache 实现
支持动态和静态两种Cache类型，同时支持标准注意力和MLA
"""

import torch
from typing import List, Tuple, Optional, Dict, Any


class Cache:
    """
    KV-Cache 抽象基类
    参考 HuggingFace Transformers 的 Cache 设计
    """
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新指定层的KV-cache
        
        Args:
            key_states: 新的key状态 [B, H, new_seq_len, d_h]
            value_states: 新的value状态 [B, H, new_seq_len, d_h]
            layer_idx: 层索引
            cache_kwargs: 额外参数（如cache_position）
        
        Returns:
            更新后的 (key_states, value_states)
        """
        raise NotImplementedError("Subclasses must implement the update method")
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """获取缓存的序列长度"""
        raise NotImplementedError("Subclasses must implement the get_seq_length method")
    
    def get_max_length(self) -> Optional[int]:
        """获取最大缓存长度（仅StaticCache有效）"""
        return None
    
    def reset(self):
        """重置cache"""
        raise NotImplementedError("Subclasses must implement the reset method")


class DynamicCache(Cache):
    """
    动态增长的KV-Cache
    适用于推理场景，每步自动扩展
    
    存储格式：
    - key_cache: List[torch.Tensor], 每层一个 [B, H, S, d_h]
    - value_cache: List[torch.Tensor], 每层一个 [B, H, S, d_h]
    """
    
    def __init__(self):
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # 已处理的token数
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV-cache
        
        Args:
            key_states: [B, H, new_seq_len, d_h]
            value_states: [B, H, new_seq_len, d_h]
            layer_idx: 层索引
            cache_kwargs: 可选的额外参数
        
        Returns:
            完整的 (key_states, value_states) 包含历史+新token
        """
        # 如果这是新层，初始化
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接新的key/value到历史cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        
        # 更新seen_tokens（仅在第一层时更新）
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """获取指定层的序列长度"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """获取可用的缓存长度"""
        return self.get_seq_length(layer_idx)
    
    def reset(self):
        """清空cache"""
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
    
    def __len__(self):
        """返回缓存的层数"""
        return len(self.key_cache)
    
    def __repr__(self):
        return f"DynamicCache(num_layers={len(self)}, seq_length={self.get_seq_length()})"


class StaticCache(Cache):
    """
    静态固定大小的KV-Cache
    适用于 torch.compile() 优化
    
    预分配固定大小的显存，使用 index_copy_ 进行高效更新
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 1,
        max_cache_len: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            num_layers: 模型层数
            num_heads: 注意力头数
            head_dim: 每个头的维度
            max_batch_size: 最大批次大小
            max_cache_len: 最大缓存序列长度
            device: 设备
            dtype: 数据类型
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # 预分配cache空间
        cache_shape = (max_batch_size, num_heads, max_cache_len, head_dim)
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        for _ in range(num_layers):
            self.key_cache.append(torch.zeros(cache_shape, dtype=dtype, device=self.device))
            self.value_cache.append(torch.zeros(cache_shape, dtype=dtype, device=self.device))
        
        self._seen_tokens = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新StaticCache
        使用 index_copy_ 进行高效原地更新
        
        Args:
            key_states: [B, H, new_seq_len, d_h]
            value_states: [B, H, new_seq_len, d_h]
            layer_idx: 层索引
            cache_kwargs: 必须包含 'cache_position' 指示写入位置
        
        Returns:
            当前有效的 (key_states, value_states)
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        
        if cache_position is None:
            # 没有提供position，直接覆盖
            k_out[:, :, :key_states.shape[2], :] = key_states
            v_out[:, :, :value_states.shape[2], :] = value_states
        else:
            # 使用cache_position指定位置
            # cache_position应该是一维tensor，如 torch.arange(start, end)
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # MPS设备可能不支持index_copy_，使用切片赋值
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states
        
        # 更新seen_tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]
        
        # 返回当前有效长度的cache
        current_length = min(self._seen_tokens, self.max_cache_len)
        return k_out[:, :, :current_length, :], v_out[:, :, :current_length, :]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """获取当前序列长度"""
        return min(self._seen_tokens, self.max_cache_len)
    
    def get_max_length(self) -> int:
        """获取最大缓存长度"""
        return self.max_cache_len
    
    def reset(self):
        """重置cache（清零并重置计数器）"""
        for i in range(self.num_layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
        self._seen_tokens = 0
    
    def __len__(self):
        """返回缓存的层数"""
        return self.num_layers
    
    def __repr__(self):
        return (f"StaticCache(num_layers={self.num_layers}, "
                f"max_cache_len={self.max_cache_len}, "
                f"current_length={self.get_seq_length()})")


class MLACache(Cache):
    """
    MLA (Multi-head Latent Attention) 专用的压缩Cache
    
    存储格式：
    - compressed_kv: List[torch.Tensor], 每层一个压缩的KV表示
    - 相比标准注意力节省约34%显存
    """
    
    def __init__(
        self,
        num_layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_heads: int,
        v_head_dim: int,
        max_batch_size: int = 1,
        max_cache_len: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        dynamic: bool = True,
    ):
        """
        Args:
            num_layers: 模型层数
            kv_lora_rank: KV的低秩维度
            qk_rope_head_dim: RoPE部分的维度
            num_heads: 注意力头数
            v_head_dim: Value的头维度
            max_batch_size: 最大批次大小
            max_cache_len: 最大缓存序列长度
            device: 设备
            dtype: 数据类型
            dynamic: 是否使用动态cache
        """
        super().__init__()
        self.num_layers = num_layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.dynamic = dynamic
        
        # MLA的压缩KV格式：
        # - compressed_k: [B, 1, S, kv_lora_rank + qk_rope_head_dim]
        # - compressed_v: [B, H, S, v_head_dim]
        
        if dynamic:
            self.compressed_k_cache: List[torch.Tensor] = []
            self.compressed_v_cache: List[torch.Tensor] = []
        else:
            # 静态模式：预分配
            k_shape = (max_batch_size, 1, max_cache_len, kv_lora_rank + qk_rope_head_dim)
            v_shape = (max_batch_size, num_heads, max_cache_len, v_head_dim)
            
            self.compressed_k_cache = [
                torch.zeros(k_shape, dtype=dtype, device=self.device)
                for _ in range(num_layers)
            ]
            self.compressed_v_cache = [
                torch.zeros(v_shape, dtype=dtype, device=self.device)
                for _ in range(num_layers)
            ]
        
        self._seen_tokens = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新MLA压缩cache
        
        Args:
            key_states: 压缩的key [B, 1, new_seq_len, kv_lora_rank + qk_rope_head_dim]
            value_states: value [B, H, new_seq_len, v_head_dim]
            layer_idx: 层索引
            cache_kwargs: 额外参数
        
        Returns:
            完整的 (key_states, value_states)
        """
        if self.dynamic:
            # 动态模式
            if len(self.compressed_k_cache) <= layer_idx:
                self.compressed_k_cache.append(key_states)
                self.compressed_v_cache.append(value_states)
            else:
                self.compressed_k_cache[layer_idx] = torch.cat(
                    [self.compressed_k_cache[layer_idx], key_states], dim=2
                )
                self.compressed_v_cache[layer_idx] = torch.cat(
                    [self.compressed_v_cache[layer_idx], value_states], dim=2
                )
            
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]
            
            return self.compressed_k_cache[layer_idx], self.compressed_v_cache[layer_idx]
        else:
            # 静态模式
            cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
            
            k_out = self.compressed_k_cache[layer_idx]
            v_out = self.compressed_v_cache[layer_idx]
            
            if cache_position is None:
                k_out[:, :, :key_states.shape[2], :] = key_states
                v_out[:, :, :value_states.shape[2], :] = value_states
            else:
                try:
                    k_out.index_copy_(2, cache_position, key_states)
                    v_out.index_copy_(2, cache_position, value_states)
                except NotImplementedError:
                    k_out[:, :, cache_position] = key_states
                    v_out[:, :, cache_position] = value_states
            
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]
            
            current_length = min(self._seen_tokens, self.max_cache_len)
            return k_out[:, :, :current_length, :], v_out[:, :, :current_length, :]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """获取序列长度"""
        if self.dynamic:
            if len(self.compressed_k_cache) <= layer_idx:
                return 0
            return self.compressed_k_cache[layer_idx].shape[2]
        else:
            return min(self._seen_tokens, self.max_cache_len)
    
    def get_max_length(self) -> Optional[int]:
        """获取最大缓存长度"""
        return self.max_cache_len if not self.dynamic else None
    
    def reset(self):
        """重置cache"""
        if self.dynamic:
            self.compressed_k_cache = []
            self.compressed_v_cache = []
        else:
            for i in range(self.num_layers):
                self.compressed_k_cache[i].zero_()
                self.compressed_v_cache[i].zero_()
        self._seen_tokens = 0
    
    def __len__(self):
        """返回缓存的层数"""
        return len(self.compressed_k_cache)
    
    def __repr__(self):
        cache_type = "Dynamic" if self.dynamic else "Static"
        return (f"MLACache({cache_type}, num_layers={len(self)}, "
                f"seq_length={self.get_seq_length()})")


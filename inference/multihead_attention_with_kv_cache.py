#!/usr/bin/env python3
"""
支持KV-cache的MultiHeadAttention实现
用于测试MLA的KV-cache压缩效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

class MultiHeadAttentionWithKVCache(nn.Module):
    """
    支持KV-cache的MultiHeadAttention
    基于train_tmp.py中的实现，添加KV-cache支持
    """
    
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False,
                 use_mla: bool = False, q_lora_rank: int = None, kv_lora_rank: int = None,
                 qk_rope_head_dim: int = None, qk_nope_head_dim: int = None, v_head_dim: int = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度 Dh
        self.use_rope = use_rope
        self.use_mla = use_mla

        if use_mla:
            # MLA 模式：使用低秩投影
            self.q_lora_rank = q_lora_rank if q_lora_rank is not None else d_model // 2
            self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else 4 * self.depth
            self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else self.depth // 2
            self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else self.depth // 2
            self.v_head_dim = v_head_dim if v_head_dim is not None else self.depth
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

            # Q 投影：低秩分解
            self.q_a_proj = nn.Linear(d_model, self.q_lora_rank, bias=True)
            self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(self.q_lora_rank, num_heads * self.q_head_dim, bias=False)

            # KV 投影：压缩的 KV 投影
            self.kv_a_proj_with_mqa = nn.Linear(d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=True)
            self.kv_a_layernorm = nn.LayerNorm(self.kv_lora_rank, eps=1e-6)
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False
            )

            self.out_proj = nn.Linear(num_heads * self.v_head_dim, d_model, bias=True)
        else:
            # 标准模式
            self.WQ = nn.Linear(d_model, d_model, bias=True)
            self.WK = nn.Linear(d_model, d_model, bias=True)
            self.WV = nn.Linear(d_model, d_model, bias=True)
            self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _rope_get_cos_sin(self, seq_len: int, head_dim: int, device):
        half_dim = head_dim // 2
        inv_freq = torch.pow(10000, -2 * torch.arange(half_dim, device=device, dtype=torch.float32) / head_dim)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('l,d->ld', positions, inv_freq)  # [L, half_dim]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin  # [L, half_dim]

    def _rope_apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x: [B, H, L, Dh]; cos/sin: [L, Dh/2]
        B, H, L, Dh = x.shape
        x_ = x.view(B, H, L, Dh // 2, 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]
        cos = cos.view(1, 1, L, Dh // 2)
        sin = sin.view(1, 1, L, Dh // 2)
        rot0 = x1 * cos - x2 * sin
        rot1 = x1 * sin + x2 * cos
        out = torch.stack([rot0, rot1], dim=-1).view(B, H, L, Dh)
        return out

    def _split_heads(self, x: torch.Tensor):
        """
        x: [B, L, d_model] -> [B, num_heads, L, depth]
        """
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.depth)  # [B, L, H, Dh]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, L, Dh]
        return x

    def _combine_heads(self, x: torch.Tensor):
        """
        x: [B, num_heads, L, depth] -> [B, L, d_model]
        """
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, Dh]
        x = x.view(B, L, H * Dh)  # [B, L, d_model]
        return x

    def forward(self, q, k, v, mask=None, return_attn: bool = True, kv_cache=None, layer_idx: int = 0):
        """
        支持KV-cache的前向传播
        Args:
            q, k, v: [B, Lq/Lk/Lv, d_model]
            mask: 注意力掩码
            return_attn: 是否返回注意力权重
            kv_cache: KV-cache对象
            layer_idx: 当前层索引
        """
        B = q.size(0)

        if self.use_mla:
            # MLA 模式：借鉴 DeepseekV3Attention
            bsz, q_len, _ = q.size()
            k_len = k.size(1)  # 获取 k 的实际序列长度

            # Q 投影：通过低秩分解
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(q)))
            q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [B, H, Lq, q_head_dim]
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            # KV 投影：压缩的 KV
            compressed_kv = self.kv_a_proj_with_mqa(k)  # 注意这里使用 k 作为输入
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe.view(bsz, k_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # [B, 1, Lk, rope_dim]

            kv = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
                .view(bsz, k_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)  # [B, H, Lk, nope_dim + v_dim]
            )

            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # 应用 RoPE 到 q_pe 和 k_pe
            if self.use_rope:
                Lq = q_pe.size(2)
                Lk = k_pe.size(2)
                cos_q, sin_q = self._rope_get_cos_sin(Lq, self.qk_rope_head_dim, q.device)
                cos_k, sin_k = self._rope_get_cos_sin(Lk, self.qk_rope_head_dim, k.device)
                q_pe = self._rope_apply(q_pe, cos_q, sin_q)
                k_pe = self._rope_apply(k_pe, cos_k, sin_k)

            # 组合 query_states 和 key_states
            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

            # k_pe 需要广播到所有头
            k_pe_expanded = k_pe.expand(bsz, self.num_heads, k_len, self.qk_rope_head_dim)
            key_states = k_pe.new_empty(bsz, self.num_heads, k_len, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe_expanded

            # 使用 value_states 作为 v
            v_states = value_states

            # 处理 mask
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                elif mask.dim() == 4 and mask.size(1) == 1:
                    pass
                else:
                    raise ValueError("mask 形状需为 [B, Lq, Lk] 或 [B, 1, Lq, Lk]")
                mask = mask.expand(B, self.num_heads, mask.size(-2), mask.size(-1))

            # 注意力计算
            attn_out, attn_weights = self._scaled_dot_product_attention(
                query_states, key_states, v_states, mask
            )  # [B, H, Lq, v_head_dim]

            # 合并头
            attn_out = attn_out.transpose(1, 2).contiguous()  # [B, Lq, H, v_head_dim]
            attn_out = attn_out.reshape(bsz, q_len, self.num_heads * self.v_head_dim)  # [B, Lq, H*v_head_dim]

            # 输出投影
            output = self.out_proj(attn_out)  # [B, Lq, d_model]
            
            # 更新KV-cache（MLA模式）
            if kv_cache is not None:
                # 对于MLA，我们缓存压缩的KV
                kv_cache.update_cache(layer_idx, compressed_kv, k_pe)
                
        else:
            # 标准模式
            # 线性映射
            q = self.WQ(q)  # [B, Lq, d_model]
            k = self.WK(k)  # [B, Lk, d_model]
            v = self.WV(v)  # [B, Lv, d_model]

            # 分头
            q = self._split_heads(q)  # [B, H, Lq, Dh]
            k = self._split_heads(k)  # [B, H, Lk, Dh]
            v = self._split_heads(v)  # [B, H, Lv, Dh]

            # 应用 RoPE 到 q,k（不作用于 v）
            if self.use_rope:
                Lq = q.size(2)
                Lk = k.size(2)
                cos_q, sin_q = self._rope_get_cos_sin(Lq, self.depth, q.device)
                cos_k, sin_k = self._rope_get_cos_sin(Lk, self.depth, k.device)
                q = self._rope_apply(q, cos_q, sin_q)
                k = self._rope_apply(k, cos_k, sin_k)

            # 处理 mask：广播到 [B, H, Lq, Lk]
            if mask is not None:
                # 允许 [B, 1, Lq, Lk] 或 [B, Lq, Lk]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # [B,1,Lq,Lk]
                elif mask.dim() == 4 and mask.size(1) == 1:
                    pass  # 已是 [B,1,Lq,Lk]
                else:
                    raise ValueError("mask 形状需为 [B, Lq, Lk] 或 [B, 1, Lq, Lk]")
                mask = mask.expand(B, self.num_heads, mask.size(-2), mask.size(-1))

            # 注意力
            attn_out, attn_weights = self._scaled_dot_product_attention(q, k, v, mask)  # [B,H,Lq,Dh], [B,H,Lq,Lk]

            # 合并头
            attn_out = self._combine_heads(attn_out)  # [B, Lq, d_model]

            # 输出线性层
            output = self.out_proj(attn_out)  # [B, Lq, d_model]
            
            # 更新KV-cache（标准模式）
            if kv_cache is not None:
                kv_cache.update_cache(layer_idx, k, v)

        if return_attn:
            return output, attn_weights
        return output

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力
        Args:
            q: (..., seq_len_q, depth)
            k: (..., seq_len_k, depth)
            v: (..., seq_len_v, depth_v)  (seq_len_k == seq_len_v)
            mask: (..., seq_len_q, seq_len_k)，
                  mask里1表示要忽略的位置，0表示保留。

        Returns:
            output: (..., seq_len_q, depth_v) 加权和
            attention_weights: (..., seq_len_q, seq_len_k) 注意力权重
        """
        # (..., seq_len_q, seq_len_k)
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        # 缩放
        dk = q.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))

        # 加上 mask
        if mask is not None:
            # 在 mask==1 的位置加上 -1e9，使 softmax 后趋近于0
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, -1e9)

        # softmax 得到注意力权重
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # 加权求和
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import logging
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]"""
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
        t: Tensor,
        freqs: Tensor,
        rotary_interleaved: bool = False,
        multi_latent_attention: bool = False,
        mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T."""
    rot_dim = freqs.shape[-1]
    input_dtype = t.dtype
    t = t.to(freqs.dtype)
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale)
    sin_ = (torch.sin(freqs) * mscale)

    # 在这一行设置断点！
    tmp = _rotate_half(t, rotary_interleaved)
    t = (t * cos_) + (tmp * sin_)
    return torch.cat((t, t_pass), dim=-1).to(input_dtype)

def _rope_apply_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """参考实现：你提供的 _rope_apply 方法（interleaved模式）"""
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


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE 等价性验证测试")
    print("=" * 60)

    # 设置随机种子以便复现
    torch.manual_seed(42)

    # 定义测试参数
    B = 2  # batch size
    H = 4  # num heads
    L = 8  # sequence length
    Dh = 64  # head dimension

    # 创建示例输入
    print(f"\n输入参数: B={B}, H={H}, L={L}, Dh={Dh}")
    x = torch.randn(B, H, L, Dh)
    print(f"输入张量 x 的形状: {x.shape}")

    # 🔧 修正：正确创建位置编码（freqs）
    # freqs 应该是 [L, Dh/2] 的角度值
    position = torch.arange(L, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    dim_indices = torch.arange(0, Dh, 2, dtype=torch.float32)  # [Dh/2]
    # 计算频率（标准 RoPE 公式）
    inv_freq = 1.0 / (10000 ** (dim_indices / Dh))
    freqs_half = position @ inv_freq.unsqueeze(0)  # [L, Dh/2]

    print(f"基础频率张量 freqs_half 的形状: {freqs_half.shape}")

    # 为 _apply_rotary_pos_emb_bshd 准备 freqs (interleaved模式)
    # _rope_apply_reference 使用的是 interleaved 模式，所以需要用 rotary_interleaved=True
    # interleaved模式下，每个角度重复两次：[θ0, θ0, θ1, θ1, ...]
    freqs_interleaved = torch.repeat_interleave(freqs_half, 2, dim=-1)  # [L, Dh]
    freqs_expanded = freqs_interleaved.unsqueeze(0).unsqueeze(0).expand(B, H, L, Dh)

    print("\n" + "=" * 60)
    print("方法1: 使用 _apply_rotary_pos_emb_bshd (interleaved=True)")
    print("=" * 60)
    result1 = _apply_rotary_pos_emb_bshd(x.clone(), freqs_expanded, rotary_interleaved=True)
    print(f"输出形状: {result1.shape}")
    print(f"输出样例 [0, 0, 0, :8]: {result1[0, 0, 0, :8]}")

    print("\n" + "=" * 60)
    print("方法2: 使用 _rope_apply_reference（你的参考实现）")
    print("=" * 60)
    # 从 freqs_half 计算 cos 和 sin
    cos_half = torch.cos(freqs_half)  # [L, Dh/2]
    sin_half = torch.sin(freqs_half)  # [L, Dh/2]

    result2 = _rope_apply_reference(x.clone(), cos_half, sin_half)
    print(f"输出形状: {result2.shape}")
    print(f"输出样例 [0, 0, 0, :8]: {result2[0, 0, 0, :8]}")


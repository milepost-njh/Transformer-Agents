#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 19:08
@Author  : nijiahui
@FileName: mask_utils.py
@Software: PyCharm
 
'''

import torch

def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    """
    输入:
        batch_data: [batch_size, seq_len]，填充位置用 pad_token_id 表示
        pad_token_id: 默认是 0
    输出:
        padding_mask: [batch_size, 1, 1, seq_len]
    """
    # 等价于 tf.math.equal(batch_data, 0)
    mask = (batch_data == pad_token_id).float()
    # 插入维度
    return mask[:, None, None, :]  # [B, 1, 1, L]


def create_look_ahead_mask(size: int):
    """
    生成 Look-ahead mask (上三角矩阵)
    参数:
        size: 序列长度 (seq_len)
    返回:
        mask: [seq_len, seq_len]，上三角为 1，其他为 0
    """
    # ones: [size, size]
    ones = torch.ones((size, size))
    # 取上三角（不含对角线）=1，下三角和对角线=0
    mask = torch.triu(ones, diagonal=1)
    return mask



def create_masks(
        inp_ids: torch.Tensor,  # [B, L_src]
        tar_ids: torch.Tensor,  # [B, L_tgt] —— 通常是 decoder 输入（已左移）
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
):
    """
    返回:
      encoder_padding_mask         : [B, 1, 1, L_src]  (给 EncoderLayer self-attn)
      decoder_mask (LA + padding)  : [B, 1, L_tgt, L_tgt]  (给 DecoderLayer 自注意力)
      encoder_decoder_padding_mask : [B, 1, 1, L_src]  (给 DecoderLayer cross-attn)
    语义:
      1 = 屏蔽（masked），0 = 保留
    """
    # 1) Encoder 端 padding mask
    encoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]
    encoder_decoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]

    # 2) Decoder 端 look-ahead + padding 合并
    B, L_tgt = tar_ids.size(0), tar_ids.size(1)

    # [L_tgt, L_tgt] → [1,1,L_tgt,L_tgt]，放到与输入相同 device/dtype
    look_ahead = create_look_ahead_mask(L_tgt).to(
        device=tar_ids.device, dtype=encoder_padding_mask.dtype
    ).unsqueeze(0).unsqueeze(1)  # [1,1,L_tgt,L_tgt]

    # 目标端 padding： [B,1,1,L_tgt] → 扩到 [B,1,L_tgt,L_tgt]
    decoder_padding_mask = create_padding_mask(tar_ids, pad_token_id=tgt_pad_id)  # [B,1,1,L_tgt]
    decoder_padding_mask = decoder_padding_mask.expand(-1, -1, L_tgt, -1)  # [B,1,L_tgt,L_tgt]

    # 合并（任一为 1 即屏蔽）
    decoder_mask = torch.maximum(decoder_padding_mask, look_ahead)  # [B,1,L_tgt,L_tgt]

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

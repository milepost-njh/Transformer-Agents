#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 16:33
@Author  : nijiahui
@FileName: transformer_model.py
@Software: PyCharm
 
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_position_embedding(sentence_length: int, d_model: int, device="cuda", dtype=torch.float32):
    """
    返回 position 对应的 embedding 矩阵
    形状: [1, sentence_length, d_model]
    """

    def get_angles(pos: torch.Tensor, i: torch.Tensor, d_model: int):
        """
        获取单词 pos 对应 embedding 的角度
        pos: [sentence_length, 1]
        i  : [1, d_model]
        return: [sentence_length, d_model]
        """
        angle_rates = 1.0 / torch.pow(
            10000,
            (2 * torch.div(i, 2, rounding_mode='floor')).float() / d_model
        )
        return pos.float() * angle_rates

    if device is None:
        device = torch.device("cpu")

    pos = torch.arange(sentence_length, device=device).unsqueeze(1)  # [L, 1]
    i = torch.arange(d_model, device=device).unsqueeze(0)  # [1, D]

    angle_rads = get_angles(pos, i, d_model)  # [L, D]

    # 偶数下标：sin
    sines = torch.sin(angle_rads[:, 0::2])
    # 奇数下标：cos
    cosines = torch.cos(angle_rads[:, 1::2])

    # 拼接还原成 [L, D]
    position_embedding = torch.zeros((sentence_length, d_model), device=device, dtype=dtype)
    position_embedding[:, 0::2] = sines
    position_embedding[:, 1::2] = cosines

    # 增加 batch 维度 [1, L, D]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding


def plot_position_embedding(position_embedding: torch.Tensor):
    """
    可视化位置编码矩阵
    参数:
        position_embedding: [1, L, D] 的张量
    """
    # 转到 CPU，并转成 numpy
    pe = position_embedding.detach().cpu().numpy()[0]  # [L, D]

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe, cmap='RdBu')  # L × D 矩阵
    plt.xlabel("Depth (d_model)")
    plt.xlim((0, pe.shape[1]))
    plt.ylabel("Position (pos)")
    plt.colorbar()
    plt.title("Positional Encoding Visualization")
    plt.show()





def scaled_dot_product_attention(q, k, v, mask=None):
    """
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


class MultiHeadAttention(nn.Module):
    """
    PyTorch 版 MHA，与 Keras 版本对应：
      q -> WQ -> 分头
      k -> WK -> 分头
      v -> WV -> 分头
      计算 scaled dot-product attention
      合并 -> 线性层
    期望输入形状：
      q, k, v: [B, L, d_model]
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度 Dh

        # 对应 Keras 的 Dense(d_model)
        self.WQ = nn.Linear(d_model, d_model, bias=True)
        self.WK = nn.Linear(d_model, d_model, bias=True)
        self.WV = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

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

    def forward(self, q, k, v, mask=None, return_attn: bool = True):
        """
        q, k, v: [B, Lq/Lk/Lv, d_model]
        mask: 期望形状为 [B, 1, Lq, Lk] 或 [B, Lq, Lk]；值为1表示屏蔽，0表示保留
        return:
          output: [B, Lq, d_model]
          attention_weights (可选): [B, num_heads, Lq, Lk]
        """
        B = q.size(0)

        # 线性映射
        q = self.WQ(q)  # [B, Lq, d_model]
        k = self.WK(k)  # [B, Lk, d_model]
        v = self.WV(v)  # [B, Lv, d_model]

        # 分头
        q = self._split_heads(q)  # [B, H, Lq, Dh]
        k = self._split_heads(k)  # [B, H, Lk, Dh]
        v = self._split_heads(v)  # [B, H, Lv, Dh]

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
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)  # [B,H,Lq,Dh], [B,H,Lq,Lk]

        # 合并头
        attn_out = self._combine_heads(attn_out)  # [B, Lq, d_model]

        # 输出线性层
        output = self.out_proj(attn_out)  # [B, Lq, d_model]

        if return_attn:
            return output, attn_weights
        return output


def feed_forward_network(d_model, dff):
    """
    前馈网络 FFN
    Args:
        d_model: 输出维度 (embedding 维度)
        dff: 内部隐层维度 (feed-forward 网络的中间层大小)
    Returns:
        nn.Sequential 模型
    """
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )


class EncoderLayer(nn.Module):
    """
    x -> self-attention -> add & norm & dropout
      -> feed-forward   -> add & norm & dropout
    期望输入:
      x: [B, L, d_model]
      src_mask: [B, 1, L, L] 或 [B, L, L]，其中 1 表示屏蔽，0 表示保留
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)  # 前面已实现
        self.ffn = feed_forward_network(d_model, dff)  # 前面已实现

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        返回:
          out: [B, L, d_model]
        """
        # Self-Attention
        attn_out, _ = self.mha(x, x, x, mask=src_mask)  # [B, L, d_model], [B, H, L, L]
        attn_out = self.dropout1(attn_out)  # 训练模式下生效
        out1 = self.norm1(x + attn_out)  # 残差 + LayerNorm

        # Feed Forward
        ffn_out = self.ffn(out1)  # [B, L, d_model]
        ffn_out = self.dropout2(ffn_out)
        out2 = self.norm2(out1 + ffn_out)

        return out2


class DecoderLayer(nn.Module):
    """
    x -> masked self-attention -> add & norm & dropout -> out1
    out1, enc_out -> cross-attention -> add & norm & dropout -> out2
    out2 -> FFN -> add & norm & dropout -> out3
    期望输入:
      x: [B, L_tgt, d_model]
      enc_out: [B, L_src, d_model]
      tgt_mask: [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]  (look-ahead + padding 的合并掩码，1=屏蔽)
      enc_dec_mask: [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]  (decoder 对 encoder 的 padding 掩码，1=屏蔽)
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # masked self-attn
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # cross-attn

        self.ffn = feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            enc_dec_mask: torch.Tensor = None,
    ):
        # 1) Masked Self-Attention (decoder 自注意力，使用 look-ahead+padding 的合并 mask)
        attn1_out, attn_weights1 = self.mha1(x, x, x, mask=tgt_mask)  # [B,Lt,D], [B,H,Lt,Lt]
        attn1_out = self.dropout1(attn1_out)
        out1 = self.norm1(x + attn1_out)

        # 2) Cross-Attention (query=out1, key/value=enc_out)，使用 encoder padding 掩码
        attn2_out, attn_weights2 = self.mha2(out1, enc_out, enc_out, mask=enc_dec_mask)  # [B,Lt,D], [B,H,Lt,Ls]
        attn2_out = self.dropout2(attn2_out)
        out2 = self.norm2(out1 + attn2_out)

        # 3) FFN
        ffn_out = self.ffn(out2)  # [B,Lt,D]
        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)  # [B,Lt,D]

        return out3, attn_weights1, attn_weights2


class EncoderModel(nn.Module):
    def __init__(self, num_layers: int, input_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        """
        参数与 Keras 版本对齐；额外提供 padding_idx 以便 Embedding 忽略 pad 的梯度。
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        # Embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码：注册为 buffer（不参与训练/优化器）
        pe = get_position_embedding(max_length, d_model)  # [1, max_len, d_model]
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠 EncoderLayer（前面我们已实现过）
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        # 预存缩放因子
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        x: [B, L]  （token ids）
        src_mask: [B, 1, L, L] 或 [B, L, L]，1=屏蔽，0=保留（与前文一致）
        return: 编码结果 [B, L, d_model]
        """
        B, L = x.shape
        # 等价于 tf.debugging.assert_less_equal
        if L > self.max_length:
            raise ValueError(f"input_seq_len ({L}) should be ≤ max_length ({self.max_length})")

        # [B, L, D]
        x = self.embedding(x)
        # 缩放：使 embedding 的尺度与位置编码相近（论文做法）
        x = x * self.scale
        # 加位置编码（按实际序列长度切片）
        x = x + self.position_embedding[:, :L, :]

        x = self.dropout(x)

        # 逐层 Encoder
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

class DecoderModel(nn.Module):
    """
    x -> masked self-attn -> add & norm & dropout
      -> cross-attn(enc_out) -> add & norm & dropout
      -> FFN -> add & norm & dropout
    """

    def __init__(self, num_layers: int, target_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码（注册为 buffer，不参与训练）
        pe = get_position_embedding(max_length, d_model)
        self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠解码层
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )

        self.scale = math.sqrt(d_model)

    def forward(
            self,
            x: torch.Tensor,  # [B, L_tgt] 目标端 token ids
            enc_out: torch.Tensor,  # [B, L_src, D] 编码器输出
            tgt_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
            enc_dec_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]（对 encoder 的 padding）
    ):
        B, Lt = x.shape
        if Lt > self.max_length:
            raise ValueError(f"output_seq_len ({Lt}) should be ≤ max_length ({self.max_length})")

        # (B, Lt, D)
        x = self.embedding(x) * self.scale
        x = x + self.position_embedding[:, :Lt, :]
        x = self.dropout(x)

        attention_weights = {}

        for i, layer in enumerate(self.decoder_layers, start=1):
            x, attn1, attn2 = layer(x, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask)
            attention_weights[f"decoder_layer{i}_att1"] = attn1  # [B, H, Lt, Lt]
            attention_weights[f"decoder_layer{i}_att2"] = attn2  # [B, H, Lt, Ls]

        # x: (B, Lt, D)
        return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None):
        super().__init__()
        self.encoder_model = EncoderModel(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=src_padding_idx,
        )
        self.decoder_model = DecoderModel(
            num_layers=num_layers,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=rate,
            padding_idx=tgt_padding_idx,
        )
        # 等价于 Keras 的 Dense(target_vocab_size)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        """
        inp_ids: [B, L_src]  源端 token ids
        tgt_ids: [B, L_tgt]  目标端 token ids（训练时通常是 shift 后的 decoder 输入）
        src_mask:    [B, 1, L_src, L_src] 或 [B, L_src, L_src]（1=屏蔽）
        tgt_mask:    [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
        enc_dec_mask:[B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]
        返回:
          logits: [B, L_tgt, target_vocab_size]
          attention_weights: dict，包含每层的 attn
        """
        enc_out = self.encoder_model(inp_ids, src_mask=src_mask)  # [B, L_src, D]
        dec_out, attention_weights = self.decoder_model(
            tgt_ids, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask
        )  # [B, L_tgt, D], dict
        logits = self.final_layer(dec_out)  # [B, L_tgt, V_tgt]
        return logits, attention_weights

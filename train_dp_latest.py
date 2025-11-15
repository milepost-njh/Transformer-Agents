# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from core.models.modeling_deepseek import DeepseekV3MoE
from collections import OrderedDict
from core.normalization import RMSNorm, LayerNorm
from training.parallel.config import ParallelConfig, ParallelMode
from training.parallel.factory import create_backend

# 多卡训练设置（DP DataParallel 单进程多GPU）
# 不在代码中强行设置 CUDA_VISIBLE_DEVICES，改由启动脚本/外部环境控制

# 修复警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizers并行以避免fork警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用TensorFlow oneDNN优化信息


# MoE 配置类
class MoEConfig:
    """MoE 配置类"""

    def __init__(
            self,
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=512,
            intermediate_size=2048,
            hidden_act="silu",
            router_aux_loss_coef=0.001,
            use_moe=True,
            n_routed_experts=8,
            routed_scaling_factor=1.0,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            n_shared_experts=None,
            moe_intermediate_size=2048,
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_moe = use_moe
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size


def setup_multi_gpu():
    """
    设置多卡训练环境
    返回: (device, use_multi_gpu, gpu_count)
    """
    if not torch.cuda.is_available():
        logger.info("CUDA不可用，使用CPU训练")
        return torch.device("cpu"), False, 0

    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 张GPU")

    if gpu_count > 1:
        logger.info(f"启用多卡训练，使用 {gpu_count} 张GPU")
        device = torch.device("cuda:0")  # 主GPU
        return device, True, gpu_count
    else:
        logger.info("只有1张GPU，使用单卡训练")
        device = torch.device("cuda:0")
        return device, False, 1


def wrap_model_for_multi_gpu(model, use_multi_gpu, gpu_count):
    """
    为多卡训练包装模型
    """
    if use_multi_gpu and gpu_count > 1:
        # 兼容旧接口：若需要DP，动态导入；当前方案用DP
        try:
            from torch.nn.parallel import DataParallel as _DP
            logger.info(f"使用DataParallel包装模型，GPU数量: {gpu_count}")
            device_ids = list(range(gpu_count))
            model = _DP(model, device_ids=device_ids)
            return model
        except Exception:
            logger.info("DataParallel 不可用，保持单卡/后端自处理")
            return model
    else:
        logger.info("使用单卡训练")
        return model


def log_gpu_memory_usage(step_name="", use_multi_gpu=False, gpu_count=1):
    """
    记录GPU内存使用情况
    """
    if not torch.cuda.is_available():
        return

    if use_multi_gpu and gpu_count > 1:
        for i in range(gpu_count):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            logger.info(f"GPU {i} {step_name} - 内存使用: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
    else:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        logger.info(f"GPU {step_name} - 内存使用: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")


def check_env():
    """
    检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本。
    返回推荐的 device ('cuda' 或 'cpu')。
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ 检测到 {gpu_count} 张GPU: {torch.cuda.get_device_name(0)}, bf16支持: {torch.cuda.is_bf16_supported()}")

        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        device = "cuda"
    else:
        logger.info("⚠️ 没检测到 CUDA，使用 CPU")
        device = "cpu"
    return device


def load_translation_dataset(train_path: str, val_path: str, delimiter: str = "\t"):
    """
    加载葡萄牙语-英语翻译数据集 (TED Talks)

    参数:
        train_path: 训练集 CSV 文件路径
        val_path: 验证集 CSV 文件路径
        delimiter: 分隔符，默认制表符 '\t'

    返回:
        train_dataset, val_dataset
    """
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path
        },
        column_names=["pt", "en"],
        delimiter=delimiter
    )

    logger.info(f"✅ 数据集加载完成: 训练集 {len(dataset['train'])} 条, 验证集 {len(dataset['validation'])} 条")

    return dataset["train"], dataset["validation"]


def train_and_load_tokenizers(
        train_dataset,
        pt_key="pt",
        en_key="en",
        vocab_size=2 ** 13,
        min_freq=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        save_dir_pt="tok_pt",
        save_dir_en="tok_en",
        max_length=1024
):
    """
    训练并加载葡萄牙语和英语的 ByteLevel BPE Tokenizer

    参数:
        train_dataset: 数据集 (需包含 pt_key 和 en_key 两列)
        pt_key: 源语言字段名 (默认 "pt")
        en_key: 目标语言字段名 (默认 "en")
        vocab_size: 词表大小
        min_freq: 最小词频
        special_tokens: 特殊符号
        save_dir_pt: 葡语 tokenizer 保存路径
        save_dir_en: 英语 tokenizer 保存路径
        max_length: 模型最大序列长度

    返回:
        pt_tokenizer, en_tokenizer
    """

    def iter_lang(ds, key):
        for ex in ds:
            txt = ex[key]
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8")
            yield txt

    # 初始化 tokenizer
    pt_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)
    en_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)

    # 训练 tokenizer
    pt_bbpe.train_from_iterator(
        iter_lang(train_dataset, pt_key),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )
    en_bbpe.train_from_iterator(
        iter_lang(train_dataset, en_key),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    # 保存 vocab/merges + tokenizer.json
    Path(save_dir_pt).mkdir(exist_ok=True)
    Path(save_dir_en).mkdir(exist_ok=True)
    pt_bbpe.save_model(save_dir_pt)
    en_bbpe.save_model(save_dir_en)
    pt_bbpe._tokenizer.save(f"{save_dir_pt}/tokenizer.json")
    en_bbpe._tokenizer.save(f"{save_dir_en}/tokenizer.json")

    # 用 PreTrainedTokenizerFast 加载
    pt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_dir_pt}/tokenizer.json")
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_dir_en}/tokenizer.json")

    # 设置特殊符号
    for tok in (pt_tokenizer, en_tokenizer):
        tok.pad_token = "<pad>"
        tok.unk_token = "<unk>"
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok.mask_token = "<mask>"
        tok.model_max_length = max_length
        tok.padding_side = "right"

    logger.info(f"✅ Tokenizer构建完成: pt词表 {len(pt_tokenizer)}, en词表 {len(en_tokenizer)}")

    return pt_tokenizer, en_tokenizer


def test_tokenizers(en_tokenizer, pt_tokenizer,
                    en_sample: str = "Transformer is awesome.",
                    pt_sample: str = "Transformers são incríveis."):
    """
    测试英文和葡萄牙语的 tokenizer 编码/解码是否正确，
    并打印 token IDs、单个 ID 对应的 token 结果。

    参数:
        en_tokenizer: 英语 tokenizer
        pt_tokenizer: 葡语 tokenizer
        en_sample: 英文测试句子
        pt_sample: 葡文测试句子
    """

    # --- English ---
    en_ids = en_tokenizer.encode(en_sample, add_special_tokens=False)
    en_decoded = en_tokenizer.decode(en_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert en_decoded == en_sample, "EN decode != original input!"

    # --- Portuguese ---
    pt_ids = pt_tokenizer.encode(pt_sample, add_special_tokens=False)
    pt_decoded = pt_tokenizer.decode(pt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert pt_decoded == pt_sample, "PT decode != original input!"
    
    logger.info(f"✅ Tokenizer测试通过: EN({len(en_ids)} tokens), PT({len(pt_ids)} tokens)")


def build_dataloaders(
        train_dataset,
        val_dataset,
        pt_tokenizer,
        en_tokenizer,
        batch_size: int = 64,
        max_length: int = 48,
        num_workers: int = 0,
        shuffle_train: bool = True,
        use_multi_gpu: bool = False,
        gpu_count: int = 1,
        train_sampler=None,
        val_sampler=None,
):
    """
    构建训练和验证 DataLoader（等价 TF 的 filter_by_max_length + padded_batch）

    参数:
        train_dataset: HuggingFace Dataset (训练集)
        val_dataset: HuggingFace Dataset (验证集)
        pt_tokenizer: 葡语 tokenizer (源语言)
        en_tokenizer: 英语 tokenizer (目标语言)
        batch_size: 批大小
        max_length: 样本最大长度（超过则过滤）
        num_workers: DataLoader worker 数量
        shuffle_train: 是否打乱训练集
        use_multi_gpu: 是否使用多卡训练
        gpu_count: GPU数量

    返回:
        train_loader, val_loader
    """

    # 多卡训练时调整batch size和num_workers
    if use_multi_gpu and gpu_count > 1:
        effective_batch_size = batch_size * gpu_count
        effective_num_workers = min(num_workers * 2, 8)
    else:
        effective_batch_size = batch_size
        effective_num_workers = num_workers

    # 1) 小工具：编码 + 添加 BOS/EOS
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]

    # 2) 构造已过滤的样本对
    def build_filtered_pairs(hf_split, pt_tok, en_tok, max_len: int):
        pairs = []
        for ex in hf_split:
            pt_ids = encode_with_bos_eos(pt_tok, ex["pt"])
            en_ids = encode_with_bos_eos(en_tok, ex["en"])
            if len(pt_ids) <= max_len and len(en_ids) <= max_len:
                pairs.append((pt_ids, en_ids))
        return pairs

    train_pairs = build_filtered_pairs(train_dataset, pt_tokenizer, en_tokenizer, max_length)
    val_pairs = build_filtered_pairs(val_dataset, pt_tokenizer, en_tokenizer, max_length)

    # 3) Dataset 类
    class PairsDataset(Dataset):
        def __init__(self, pairs): self.pairs = pairs

        def __len__(self): return len(self.pairs)

        def __getitem__(self, idx):
            pt_ids, en_ids = self.pairs[idx]
            return {"pt_input_ids": pt_ids, "en_input_ids": en_ids}

    # 4) Collate 函数（动态 padding）
    def collate_padded(batch, pad_id_pt: int, pad_id_en: int):
        def pad_block(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                L = len(s)
                out[i, :L] = torch.tensor(s, dtype=torch.long)
                attn[i, :L] = 1
            return out, attn

        pt_ids_list = [ex["pt_input_ids"] for ex in batch]
        en_ids_list = [ex["en_input_ids"] for ex in batch]
        pt_input_ids, pt_attention_mask = pad_block(pt_ids_list, pt_tokenizer.pad_token_id)
        en_input_ids, en_attention_mask = pad_block(en_ids_list, en_tokenizer.pad_token_id)

        return {
            "pt_input_ids": pt_input_ids,
            "pt_attention_mask": pt_attention_mask,
            "en_input_ids": en_input_ids,
            "en_attention_mask": en_attention_mask,
        }

    # 5) DataLoader
    train_loader = DataLoader(
        PairsDataset(train_pairs),
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None and shuffle_train),
        sampler=train_sampler,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        PairsDataset(val_pairs),
        batch_size=effective_batch_size,
        shuffle=False if val_sampler is not None else False,
        sampler=val_sampler,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def test_dataloaders(train_loader, val_loader, show_val: bool = True):
    """
    检查 DataLoader 的 batch 输出，打印张量形状和一个示例

    参数:
        train_loader: 训练 DataLoader
        val_loader: 验证 DataLoader
        show_val: 是否展示验证集的一个样本（默认 True）
    """
    batch = next(iter(train_loader))
    logger.info(f"✅ DataLoader测试通过: batch_size={batch['pt_input_ids'].shape[0]}, seq_len={batch['pt_input_ids'].shape[1]}")


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
        # 在 mask==1 的位置加上极小值，使 softmax 后趋近于0
        # 使用 -1e4 而不是 -1e9，避免 bfloat16 溢出
        mask_value = -1e4 if q.dtype in [torch.float16, torch.bfloat16] else -1e9
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, mask_value)

    # softmax 得到注意力权重
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    PyTorch 版 MHA，支持标准注意力和 MLA（Multi-head Latent Attention）：
      标准模式：
        q -> WQ -> 分头
        k -> WK -> 分头
        v -> WV -> 分头
      MLA 模式（借鉴 DeepseekV3）：
        q -> q_a_proj -> layernorm -> q_b_proj -> 分头（分离 nope 和 pe）
        kv -> kv_a_proj -> layernorm -> kv_b_proj -> 分头（分离 nope 和 pe）
      期望输入形状：
        q, k, v: [B, L, d_model]
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
            self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else d_model // 4
            self.qk_rope_head_dim = qk_rope_head_dim if qk_rope_head_dim is not None else self.depth // 2
            self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else self.depth // 2
            self.v_head_dim = v_head_dim if v_head_dim is not None else self.depth
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

            # Q 投影：低秩分解
            self.q_a_proj = nn.Linear(d_model, self.q_lora_rank, bias=True)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(self.q_lora_rank, num_heads * self.q_head_dim, bias=False)

            # KV 投影：压缩的 KV 投影
            self.kv_a_proj_with_mqa = nn.Linear(d_model, self.kv_lora_rank + self.qk_rope_head_dim, bias=True)
            self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=1e-6)
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

    def forward(self, q, k, v, mask=None, return_attn: bool = True, past_key_value=None, use_cache: bool = False):
        """
        q, k, v: [B, Lq/Lk/Lv, d_model]
        mask: 期望形状为 [B, 1, Lq, Lk] 或 [B, Lq, Lk]；值为1表示屏蔽，0表示保留
        past_key_value: 可选的KV-cache，格式为(past_k, past_v)
        use_cache: 是否返回KV-cache
        return:
          output: [B, Lq, d_model]
          attention_weights (可选): [B, num_heads, Lq, Lk]
          present_key_value (可选): 当前的KV-cache
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
            
            # KV-cache 处理
            if past_key_value is not None:
                # 使用缓存的K和V
                past_k, past_v = past_key_value
                key_states = torch.cat([past_k, key_states], dim=2)  # [B, H, Lk_past+Lk, head_dim]
                v_states = torch.cat([past_v, v_states], dim=2)  # [B, H, Lk_past+Lk, v_head_dim]
            
            # 保存当前的KV用于下一步
            present_key_value = (key_states, v_states) if use_cache else None

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
            attn_out, attn_weights = scaled_dot_product_attention(
                query_states, key_states, v_states, mask
            )  # [B, H, Lq, v_head_dim]

            # 合并头
            attn_out = attn_out.transpose(1, 2).contiguous()  # [B, Lq, H, v_head_dim]
            attn_out = attn_out.reshape(bsz, q_len, self.num_heads * self.v_head_dim)  # [B, Lq, H*v_head_dim]

            # 输出投影
            output = self.out_proj(attn_out)  # [B, Lq, d_model]
            
            # 返回值处理
            if use_cache:
                if return_attn:
                    return output, attn_weights, present_key_value
                return output, present_key_value
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
            
            # KV-cache 处理
            if past_key_value is not None:
                # 使用缓存的K和V
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)  # [B, H, Lk_past+Lk, Dh]
                v = torch.cat([past_v, v], dim=2)  # [B, H, Lk_past+Lk, Dh]
            
            # 保存当前的KV用于下一步
            present_key_value = (k, v) if use_cache else None

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

        # 统一返回值处理
        if use_cache:
            if return_attn:
                return output, attn_weights, present_key_value
            return output, present_key_value
        else:
            if return_attn:
                return output, attn_weights
            return output


def feed_forward_network(d_model, dff, use_moe=False, moe_config=None):
    """
    前馈网络 FFN
    Args:
        d_model: 输出维度 (embedding 维度)
        dff: 内部隐层维度 (feed-forward 网络的中间层大小)
        use_moe: 是否使用 MoE
        moe_config: MoE 配置
    Returns:
        nn.Module 模型
    """
    if use_moe and moe_config is not None:
        return DeepseekV3MoE(moe_config)
    else:
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

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, use_rope: bool = True,
                 use_moe: bool = False, moe_config=None, use_mla: bool = False,
                 q_lora_rank: int = None, kv_lora_rank: int = None):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, use_rope=use_rope, use_mla=use_mla,
                                      q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)  # 支持 MLA
        self.ffn = feed_forward_network(d_model, dff, use_moe=use_moe, moe_config=moe_config)  # 支持 MoE

        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None, 
                past_key_value: Tuple[torch.Tensor, torch.Tensor] = None, use_cache: bool = False):
        """
        返回:
          out: [B, L, d_model] 或 (out, router_logits) 如果使用 MoE
          present_key_value: 当前的KV-cache（如果use_cache=True）
        """
        # TODO: 全局自注意力 - 编码器中的自注意力，可以关注序列中的所有位置
        mha_output = self.mha(x, x, x, mask=src_mask, past_key_value=past_key_value, use_cache=use_cache)
        
        # 处理MHA的返回值
        if use_cache:
            attn_out, _, present_key_value = mha_output
        else:
            attn_out, _ = mha_output
            present_key_value = None
        
        attn_out = self.dropout1(attn_out)  # 训练模式下生效
        out1 = self.norm1(x + attn_out)  # 残差 + RMSNorm

        # Feed Forward
        ffn_out = self.ffn(out1)  # [B, L, d_model] 或 (ffn_out, router_logits) 如果使用 MoE
        if isinstance(ffn_out, tuple):
            ffn_out, router_logits = ffn_out
        else:
            router_logits = None

        ffn_out = self.dropout2(ffn_out)
        out2 = self.norm2(out1 + ffn_out)

        # 返回值处理
        if use_cache:
            if router_logits is not None:
                return out2, router_logits, present_key_value
            return out2, present_key_value
        else:
            if router_logits is not None:
                return out2, router_logits
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

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1, use_rope: bool = True,
                 use_moe: bool = False, moe_config=None, use_mla: bool = False,
                 q_lora_rank: int = None, kv_lora_rank: int = None):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, use_rope=use_rope, use_mla=use_mla,
                                       q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)  # masked self-attn
        self.mha2 = MultiHeadAttention(d_model, num_heads, use_rope=use_rope, use_mla=use_mla,
                                       q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)  # cross-attn

        self.ffn = feed_forward_network(d_model, dff, use_moe=use_moe, moe_config=moe_config)

        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)
        self.norm3 = RMSNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            enc_dec_mask: torch.Tensor = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
    ):
        """
        Args:
            x: decoder输入 [B, Lt, D]
            enc_out: encoder输出 [B, Ls, D]
            tgt_mask: decoder自注意力mask
            enc_dec_mask: encoder-decoder交叉注意力mask
            past_key_values: (self_attn_cache, cross_attn_cache) 元组
            use_cache: 是否返回KV-cache
        
        Returns:
            out3: decoder输出
            attn_weights1: self-attention权重
            attn_weights2: cross-attention权重
            present_key_values: 当前的KV-cache (如果use_cache=True)
            router_logits: MoE路由logits (如果使用MoE)
        """
        # 解析past_key_values
        if past_key_values is not None:
            self_attn_past_kv, cross_attn_past_kv = past_key_values
        else:
            self_attn_past_kv, cross_attn_past_kv = None, None
        
        # TODO: 掩码自注意力 - 解码器自注意力，使用look-ahead+padding掩码防止信息泄露
        mha1_output = self.mha1(x, x, x, mask=tgt_mask, past_key_value=self_attn_past_kv, use_cache=use_cache)
        
        if use_cache:
            attn1_out, attn_weights1, self_attn_present_kv = mha1_output
        else:
            attn1_out, attn_weights1 = mha1_output
            self_attn_present_kv = None
        
        attn1_out = self.dropout1(attn1_out)
        out1 = self.norm1(x + attn1_out)

        # TODO: 交叉注意力 - 解码器对编码器输出的注意力，query来自decoder，key/value来自encoder
        # 注意：cross-attention不使用cache，因为encoder输出是固定的
        mha2_output = self.mha2(out1, enc_out, enc_out, mask=enc_dec_mask, past_key_value=None, use_cache=False)
        attn2_out, attn_weights2 = mha2_output
        cross_attn_present_kv = None  # cross-attention不需要cache
        
        attn2_out = self.dropout2(attn2_out)
        out2 = self.norm2(out1 + attn2_out)

        # 3) FFN
        ffn_out = self.ffn(out2)  # [B,Lt,D] 或 (ffn_out, router_logits) 如果使用 MoE
        if isinstance(ffn_out, tuple):
            ffn_out, router_logits = ffn_out
        else:
            router_logits = None

        ffn_out = self.dropout3(ffn_out)
        out3 = self.norm3(out2 + ffn_out)  # [B,Lt,D]

        # 组合present_key_values
        present_key_values = None
        if use_cache:
            present_key_values = (self_attn_present_kv, cross_attn_present_kv)

        # 返回值处理
        if use_cache:
            if router_logits is not None:
                return out3, attn_weights1, attn_weights2, present_key_values, router_logits
            return out3, attn_weights1, attn_weights2, present_key_values
        else:
            if router_logits is not None:
                return out3, attn_weights1, attn_weights2, router_logits
            return out3, attn_weights1, attn_weights2


class EncoderModel(nn.Module):
    def __init__(self, num_layers: int, input_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None, use_rope: bool = True, use_moe: bool = False, moe_config=None,
                 use_mla: bool = False, q_lora_rank: int = None, kv_lora_rank: int = None):
        """
        参数与 Keras 版本对齐；额外提供 padding_idx 以便 Embedding 忽略 pad 的梯度。
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        # Embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码（默认使用 RoPE，不注册绝对位置编码）
        self.use_rope = use_rope
        if not self.use_rope:
            pe = get_position_embedding(max_length, d_model)  # [1, max_len, d_model]
            self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠 EncoderLayer（前面我们已实现过）
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate, use_rope=self.use_rope, use_moe=use_moe,
                          moe_config=moe_config, use_mla=use_mla, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)
             for _ in range(num_layers)]
        )

        # 预存缩放因子
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None, 
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False):
        """
        x: [B, L]  （token ids）
        src_mask: [B, 1, L, L] 或 [B, L, L]，1=屏蔽，0=保留（与前文一致）
        past_key_values: 每层的KV-cache列表
        use_cache: 是否返回KV-cache
        return: 编码结果 [B, L, d_model] 和可选的cache/router_logits
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
        if not self.use_rope:
            x = x + self.position_embedding[:, :L, :]

        x = self.dropout(x)

        # 逐层 Encoder
        router_logits_list = []
        present_key_values = [] if use_cache else None
        
        for layer_idx, layer in enumerate(self.encoder_layers):
            # 获取该层的past_key_value
            layer_past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            
            layer_output = layer(x, src_mask, past_key_value=layer_past_kv, use_cache=use_cache)
            
            # 处理返回值（可能包含router_logits和present_key_value）
            if use_cache:
                if len(layer_output) == 3:
                    # (out, router_logits, present_kv)
                    x, router_logits, layer_present_kv = layer_output
                    router_logits_list.append(router_logits)
                    present_key_values.append(layer_present_kv)
                else:
                    # (out, present_kv)
                    x, layer_present_kv = layer_output
                    present_key_values.append(layer_present_kv)
            else:
                if isinstance(layer_output, tuple):
                    x, router_logits = layer_output
                    router_logits_list.append(router_logits)
                else:
                    x = layer_output

        # 返回值处理
        if use_cache:
            if router_logits_list:
                return x, router_logits_list, present_key_values
            return x, present_key_values
        else:
            if router_logits_list:
                return x, router_logits_list
            return x


class DecoderModel(nn.Module):
    """
    x -> masked self-attn -> add & norm & dropout
      -> cross-attn(enc_out) -> add & norm & dropout
      -> FFN -> add & norm & dropout
    """

    def __init__(self, num_layers: int, target_vocab_size: int, max_length: int,
                 d_model: int, num_heads: int, dff: int, rate: float = 0.1,
                 padding_idx: int = None, use_rope: bool = True, use_moe: bool = False, moe_config=None,
                 use_mla: bool = False, q_lora_rank: int = None, kv_lora_rank: int = None):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        # 词嵌入
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=padding_idx)

        # 位置编码（默认使用 RoPE，不注册绝对位置编码）
        self.use_rope = use_rope
        if not self.use_rope:
            pe = get_position_embedding(max_length, d_model)
            self.register_buffer("position_embedding", pe, persistent=False)

        self.dropout = nn.Dropout(rate)

        # 堆叠解码层
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, rate, use_rope=self.use_rope, use_moe=use_moe,
                          moe_config=moe_config, use_mla=use_mla, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank)
             for _ in range(num_layers)]
        )

        self.scale = math.sqrt(d_model)

    def forward(
            self,
            x: torch.Tensor,  # [B, L_tgt] 目标端 token ids
            enc_out: torch.Tensor,  # [B, L_src, D] 编码器输出
            tgt_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
            enc_dec_mask: torch.Tensor = None,  # [B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]（对 encoder 的 padding）
            past_key_values: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]] = None,
            use_cache: bool = False,
    ):
        """
        Args:
            x: decoder输入token ids [B, Lt]
            enc_out: encoder输出 [B, Ls, D]
            tgt_mask: decoder自注意力mask
            enc_dec_mask: encoder-decoder交叉注意力mask
            past_key_values: 每层的(self_attn_kv, cross_attn_kv)元组列表
            use_cache: 是否返回KV-cache
        
        Returns:
            x: decoder输出 [B, Lt, D]
            attention_weights: 注意力权重字典
            present_key_values: 当前的KV-cache列表 (如果use_cache=True)
            router_logits_list: MoE路由logits (如果使用MoE)
        """
        B, Lt = x.shape
        if Lt > self.max_length:
            raise ValueError(f"output_seq_len ({Lt}) should be ≤ max_length ({self.max_length})")

        # (B, Lt, D)
        x = self.embedding(x) * self.scale
        if not self.use_rope:
            x = x + self.position_embedding[:, :Lt, :]
        x = self.dropout(x)

        attention_weights = {}
        router_logits_list = []
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.decoder_layers, start=1):
            layer_idx = i - 1
            # 获取该层的past_key_values
            layer_past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            
            layer_output = layer(
                x, enc_out, 
                tgt_mask=tgt_mask, 
                enc_dec_mask=enc_dec_mask,
                past_key_values=layer_past_kv,
                use_cache=use_cache
            )
            
            # 处理返回值
            if use_cache:
                if len(layer_output) == 5:  # 包含 router_logits
                    x, attn1, attn2, layer_present_kv, router_logits = layer_output
                    router_logits_list.append(router_logits)
                    present_key_values.append(layer_present_kv)
                else:  # 不包含 router_logits
                    x, attn1, attn2, layer_present_kv = layer_output
                    present_key_values.append(layer_present_kv)
            else:
                if len(layer_output) == 4:  # 包含 router_logits
                    x, attn1, attn2, router_logits = layer_output
                    router_logits_list.append(router_logits)
                else:  # 不包含 router_logits
                    x, attn1, attn2 = layer_output

            attention_weights[f"decoder_layer{i}_att1"] = attn1  # [B, H, Lt, Lt]
            attention_weights[f"decoder_layer{i}_att2"] = attn2  # [B, H, Lt, Ls]

        # x: (B, Lt, D)
        # 返回值处理
        if use_cache:
            if router_logits_list:
                return x, attention_weights, present_key_values, router_logits_list
            return x, attention_weights, present_key_values
        else:
            if router_logits_list:
                return x, attention_weights, router_logits_list
            return x, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None,
                 use_rope: bool = True, use_moe: bool = False, moe_config=None,
                 use_mla: bool = False, q_lora_rank: int = None, kv_lora_rank: int = None):
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
            use_rope=use_rope,
            use_moe=use_moe,
            moe_config=moe_config,
            use_mla=use_mla,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
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
            use_rope=use_rope,
            use_moe=use_moe,
            moe_config=moe_config,
            use_mla=use_mla,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
        )
        # 等价于 Keras 的 Dense(target_vocab_size)
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None,
                past_key_values: Optional[Dict[str, List]] = None, use_cache: bool = False,
                encoder_outputs: Optional[torch.Tensor] = None):
        """
        Transformer前向传播
        
        Args:
            inp_ids: [B, L_src]  源端 token ids
            tgt_ids: [B, L_tgt]  目标端 token ids（训练时通常是 shift 后的 decoder 输入）
            src_mask:    [B, 1, L_src, L_src] 或 [B, L_src, L_src]（1=屏蔽）
            tgt_mask:    [B, 1, L_tgt, L_tgt] 或 [B, L_tgt, L_tgt]（look-ahead+padding）
            enc_dec_mask:[B, 1, L_tgt, L_src] 或 [B, L_tgt, L_src]
            past_key_values: {"encoder": encoder_cache_list, "decoder": decoder_cache_list}
                           仅在推理时使用，训练时为None
            use_cache: 是否返回KV-cache（训练时=False，推理时=True）
            encoder_outputs: 预计算的encoder输出（仅推理decode阶段使用）
        
        Returns:
            logits: [B, L_tgt, target_vocab_size]
            attention_weights: dict，包含每层的 attn
            present_key_values: 当前的KV-cache (如果use_cache=True)
            router_logits: list，包含每层的 router_logits（如果使用 MoE）
        
        使用说明：
            训练模式：use_cache=False（默认），正常并行计算所有位置
            推理模式：use_cache=True，启用KV-cache加速自回归生成
                    - Prefill阶段：计算encoder一次，保存cache
                    - Decode阶段：逐token生成，复用历史KV
        """
        # 解析past_key_values
        encoder_past_kv = None
        decoder_past_kv = None
        if past_key_values is not None:
            encoder_past_kv = past_key_values.get("encoder", None)
            decoder_past_kv = past_key_values.get("decoder", None)
        
        # Encoder阶段
        if encoder_outputs is None:
            # 需要运行encoder（prefill阶段或训练阶段）
            enc_output = self.encoder_model(inp_ids, src_mask=src_mask, 
                                           past_key_values=encoder_past_kv, use_cache=use_cache)
            
            if use_cache:
                if len(enc_output) == 3:
                    # (enc_out, enc_router_logits, encoder_present_kv)
                    enc_out, enc_router_logits, encoder_present_kv = enc_output
                else:
                    # (enc_out, encoder_present_kv)
                    enc_out, encoder_present_kv = enc_output
                    enc_router_logits = None
            else:
                if isinstance(enc_output, tuple):
                    enc_out, enc_router_logits = enc_output
                else:
                    enc_out = enc_output
                    enc_router_logits = None
                encoder_present_kv = None
        else:
            # 复用已有的encoder输出（decode阶段）
            enc_out = encoder_outputs
            enc_router_logits = None
            encoder_present_kv = encoder_past_kv  # 保持encoder cache不变

        # Decoder阶段
        dec_output = self.decoder_model(
            tgt_ids, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask,
            past_key_values=decoder_past_kv, use_cache=use_cache
        )

        # 处理decoder输出
        if use_cache:
            if len(dec_output) == 4:
                # (dec_out, attention_weights, decoder_present_kv, dec_router_logits)
                dec_out, attention_weights, decoder_present_kv, dec_router_logits = dec_output
            else:
                # (dec_out, attention_weights, decoder_present_kv)
                dec_out, attention_weights, decoder_present_kv = dec_output
                dec_router_logits = None
        else:
            if isinstance(dec_output, tuple) and len(dec_output) == 3:
                dec_out, attention_weights, dec_router_logits = dec_output
            else:
                dec_out, attention_weights = dec_output
                dec_router_logits = None
            decoder_present_kv = None

        logits = self.final_layer(dec_out)  # [B, L_tgt, V_tgt]

        # 合并所有 router_logits
        router_logits = []
        if enc_router_logits:
            router_logits.extend(enc_router_logits)
        if dec_router_logits:
            router_logits.extend(dec_router_logits)

        # 组合present_key_values
        present_key_values = None
        if use_cache:
            present_key_values = {
                "encoder": encoder_present_kv,
                "decoder": decoder_present_kv
            }

        # 返回值处理
        if use_cache:
            if router_logits:
                return logits, attention_weights, present_key_values, router_logits
            return logits, attention_weights, present_key_values
        else:
            if router_logits:
                return logits, attention_weights, router_logits
            return logits, attention_weights


class CustomizedSchedule(_LRScheduler):
    """
    Noam / Transformer LR:
      lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # 确保从 1 开始
        scale = self.d_model ** -0.5
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]


def loss_function(real, pred, router_logits=None, moe_config=None, mtp_logits=None, mtp_config=None):
    """
    Args:
        real: (B, L) target ids (shift 后)
        pred: (B, L, V) logits
        router_logits: list of router logits (如果使用 MoE)
        moe_config: MoE 配置
        mtp_logits: list of MTP logits (如果使用 MTP)
    Returns:
        loss (float): 平均有效 token 的交叉熵损失 + MoE 辅助损失 + MTP 损失
    """
    B, L, V = pred.shape

    # 展平
    pred = pred.reshape(-1, V)  # (B*L, V)
    real = real.reshape(-1)  # (B*L,)

    # token 级别交叉熵 (padding 已被 ignore_index 屏蔽)
    loss_ = loss_object(pred, real)  # (B*L,)
    main_loss = loss_.mean()

    total_loss = main_loss

    # 添加 MoE 辅助损失
    if router_logits is not None and moe_config is not None:
        aux_loss = load_balancing_loss_func(
            router_logits,
            num_experts=moe_config.n_routed_experts,
            top_k=moe_config.num_experts_per_tok
        )
        total_loss = total_loss + moe_config.router_aux_loss_coef * aux_loss

    # 添加 MTP 损失
    if mtp_logits is not None:
        from core.models.deepseek_mtp import compute_mtp_loss
        # 获取MTP配置中的损失权重
        mtp_loss_weight = mtp_config.mtp_loss_weight if mtp_config else 0.1
        mtp_loss = compute_mtp_loss(mtp_logits, real.reshape(B, L), mtp_loss_weight)
        total_loss = total_loss + mtp_loss
        
        # 只在第一次计算MTP损失时打印日志（避免DP模式重复打印）
        if not hasattr(loss_function, '_mtp_loss_logged'):
            logger.info(f"✅ MTP损失已启用: 权重={mtp_loss_weight}, 首次损失={mtp_loss:.4f}, 总损失={total_loss:.4f}")
            loss_function._mtp_loss_logged = True

    return total_loss


def load_balancing_loss_func(gate_logits, num_experts: int = None, top_k=2) -> float:
    """计算负载均衡损失"""
    if gate_logits is None or len(gate_logits) == 0:
        return 0

    # 如果 gate_logits 是列表，需要合并所有层的 router_logits
    if isinstance(gate_logits, list):
        if len(gate_logits) == 0:
            return 0
        # 合并所有层的 router_logits
        compute_device = gate_logits[0].device
        gate_logits = torch.cat([gate.to(compute_device) for gate in gate_logits], dim=0)
    elif isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        gate_logits = torch.cat([gate.to(compute_device) for gate in gate_logits], dim=0)

    # 确保 gate_logits 是 2D 张量 [batch_size * seq_len, num_experts]
    if gate_logits.dim() == 3:
        batch_size, seq_len, num_experts_dim = gate_logits.shape
        gate_logits = gate_logits.view(-1, num_experts_dim)

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    expert_mask = torch.max(expert_mask, axis=-2).values
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)) * (num_experts ** 2)


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


@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    mask = (real != pad_id)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)


class AverageMeter:
    def __init__(self, name="meter"): self.name = name; self.reset()

    def reset(self): self.sum = 0.0; self.n = 0

    def update(self, val, count=1): self.sum += float(val) * count; self.n += count

    @property
    def avg(self): return self.sum / max(1, self.n)


def train_step(batch, transformer, optimizer, scheduler=None, device=None, moe_config=None, use_multi_gpu=False, mtp_config=None):
    """
    训练单步
    
    注意：训练时不使用KV-cache（use_cache默认为False）
    原因：
    - 训练使用teacher forcing，有完整的目标序列
    - 可以并行计算所有位置的attention，无需逐token生成
    - KV-cache主要用于推理阶段的自回归生成
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.train()

    # 只在必要时清理GPU缓存（减少频率以提高性能）
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    inp = batch["pt_input_ids"].to(device)
    tar = batch["en_input_ids"].to(device)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    SRC_PAD_ID = pt_tokenizer.pad_token_id
    TGT_PAD_ID = en_tokenizer.pad_token_id

    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
        inp, tar_inp, src_pad_id=SRC_PAD_ID, tgt_pad_id=TGT_PAD_ID
    )
    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

    # 使用 PyTorch SDPA + bf16 autocast（L20 支持bf16）
    use_autocast = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_autocast:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    with autocast_ctx:
        # 训练时不使用KV-cache（use_cache=False是默认值，这里不显式传递）
        # 原因：训练有完整序列，可以并行计算，无需缓存历史K/V
        # KV-cache只在推理的自回归生成时才有用
        transformer_output = transformer(
            inp, tar_inp,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask
            # use_cache=False  # 默认值，不需要显式写
        )

    # 处理 MoE 和 MTP 输出
    if isinstance(transformer_output, tuple) and len(transformer_output) == 4:
        # MTP 模式：logits, attention_weights, router_logits, mtp_logits
        logits, _, router_logits, mtp_logits = transformer_output
    elif isinstance(transformer_output, tuple) and len(transformer_output) == 3:
        # MoE 模式：logits, attention_weights, router_logits
        logits, _, router_logits = transformer_output
        mtp_logits = None
    else:
        # 标准模式：logits, attention_weights
        logits, _ = transformer_output
        router_logits = None
        mtp_logits = None

    loss = loss_function(tar_real, logits, router_logits=router_logits, moe_config=moe_config, mtp_logits=mtp_logits, mtp_config=mtp_config)

    # 检测NaN或Inf损失
    if not torch.isfinite(loss):
        logger.error(f"Loss is {loss.item()}, skipping this batch")
        return 0.0, 0.0

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 改进的梯度裁剪策略
    # 1. 先计算梯度范数
    # 对于DataParallel包装的模型，需要访问module属性
    model_for_grad_clip = transformer.module if use_multi_gpu else transformer
    grad_norm = torch.nn.utils.clip_grad_norm_(model_for_grad_clip.parameters(), max_norm=0.5)

    # 2. 更严格的梯度监控
    if grad_norm > 5.0:
        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
        # 如果梯度范数过大，进一步裁剪
        torch.nn.utils.clip_grad_norm_(model_for_grad_clip.parameters(), max_norm=0.1)
        logger.warning(f"Applied additional gradient clipping to 0.1")

    # 3. 检查是否有NaN梯度
    has_nan_grad = False
    for name, param in model_for_grad_clip.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.error(f"NaN gradient detected in {name}")
            has_nan_grad = True
            break

    if has_nan_grad:
        logger.error("Skipping this batch due to NaN gradients")
        return 0.0, 0.0

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    acc = token_accuracy(tar_real, logits, pad_id=TGT_PAD_ID)
    return loss.item(), acc


def train_model(
        epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        scheduler=None,
        device: str = None,
        log_every: int = 100,
        ckpt_dir: str = "checkpoints",
        ckpt_prefix: str = "ckpt",
        tensorboard_dir: str = "runs",
        moe_config=None,
        use_multi_gpu: bool = False,
        mtp_config=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 初始化 TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(tensorboard_dir, f"transformer_rope_{timestamp}"))
    logger.info(f"TensorBoard logs will be saved to: {os.path.join(tensorboard_dir, f'transformer_rope_{timestamp}')}")

    train_loss_meter = AverageMeter("train_loss")
    train_acc_meter = AverageMeter("train_accuracy")
    global_step = 0

    for epoch in range(epochs):
        try:
            start = time.time()
            train_loss_meter.reset()
            train_acc_meter.reset()
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                loss_val, acc_val = train_step(
                    batch=batch, transformer=model, optimizer=optimizer, scheduler=scheduler, device=device,
                    moe_config=moe_config, use_multi_gpu=use_multi_gpu, mtp_config=mtp_config
                )
                train_loss_meter.update(loss_val, 1)
                train_acc_meter.update(acc_val, 1)

                global_step += 1

                # 记录到 TensorBoard
                writer.add_scalar('Train/Loss', loss_val, global_step)
                writer.add_scalar('Train/Accuracy', acc_val, global_step)

                # 记录学习率
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('Train/Learning_Rate', current_lr, global_step)
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Train/Learning_Rate', current_lr, global_step)

                # 记录梯度统计信息
                if batch_idx % log_every == 0:
                    grad_norms = []
                    param_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            param_norm = param.data.norm(2).item()
                            grad_norms.append(grad_norm)
                            param_norms.append(param_norm)

                    if grad_norms:
                        avg_grad_norm = sum(grad_norms) / len(grad_norms)
                        max_grad_norm = max(grad_norms)
                        avg_param_norm = sum(param_norms) / len(param_norms)

                        writer.add_scalar('Train/Avg_Gradient_Norm', avg_grad_norm, global_step)
                        writer.add_scalar('Train/Max_Gradient_Norm', max_grad_norm, global_step)
                        writer.add_scalar('Train/Avg_Parameter_Norm', avg_param_norm, global_step)

                if batch_idx % log_every == 0:
                    # 添加内存监控
                    memory_info = ""
                    if torch.cuda.is_available():
                        if use_multi_gpu and gpu_count > 1:
                            # 多卡训练时显示主GPU内存使用
                            memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                            memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                            memory_info = f" GPU0内存: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB"
                        else:
                            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
                            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
                            memory_info = f" GPU内存: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB"

                    
                    logger.info(
                        f"Epoch {epoch + 1} Batch {batch_idx} global_step {global_step}"
                        f"Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}{memory_info}"
                    )

            # 记录每个 epoch 的平均指标到 TensorBoard
            writer.add_scalar('Epoch/Train_Loss', train_loss_meter.avg, epoch + 1)
            writer.add_scalar('Epoch/Train_Accuracy', train_acc_meter.avg, epoch + 1)
            writer.add_scalar('Epoch/Time', time.time() - start, epoch + 1)

            logger.info(f"Epoch {epoch + 1} Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}")
            logger.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

            # 每个epoch结束后进行验证集评测
            validate_loss, validate_acc = evaluate_on_val(model, val_loader, device, moe_config=moe_config, mtp_config=mtp_config)

            # 记录验证指标到 TensorBoard
            writer.add_scalar('Epoch/Validation_Loss', validate_loss, epoch + 1)
            writer.add_scalar('Epoch/Validation_Accuracy', validate_acc, epoch + 1)

            logger.info(f"Validation - Epoch {epoch + 1} Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}\n")

            # 每个epoch结束后保存checkpoint
            save_ckpt(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=global_step,
                ckpt_dir=ckpt_dir,
                tag="latest",
                use_multi_gpu=use_multi_gpu
            )

        except Exception as e:
            import traceback
            error_msg = f"训练出错: {e}"
            logger.error(error_msg)
            logger.error(f"详细错误信息: {traceback.format_exc()}")

            # 检查CUDA内存
            if torch.cuda.is_available():
                logger.error(f"CUDA内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
                logger.error(f"CUDA内存缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

            # 只保存一次错误检查点，避免产生太多文件
            if not hasattr(train_model, '_error_saved'):
                save_ckpt(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    tag="error",
                    use_multi_gpu=use_multi_gpu
                )
                train_model._error_saved = True

            # 继续训练而不是中断
            logger.info("跳过当前batch，继续训练...")
            continue

    # 训练结束后关闭 TensorBoard writer
    writer.close()
    logger.info("TensorBoard logging completed. Use 'tensorboard --logdir=runs' to view the logs.")


@torch.no_grad()
def evaluate_on_val(model, val_loader, device, moe_config=None, mtp_config=None):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0

    for batch in val_loader:
        inp = batch["pt_input_ids"].to(device)
        tar = batch["en_input_ids"].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            inp, tar_inp, src_pad_id=pt_tokenizer.pad_token_id, tgt_pad_id=en_tokenizer.pad_token_id
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

        model_output = model(
            inp, tar_inp,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask
        )

        # 处理不同模式的输出
        if isinstance(model_output, tuple) and len(model_output) == 4:
            # MTP 模式：logits, attention_weights, router_logits, mtp_logits
            logits, _, router_logits, mtp_logits = model_output
        elif isinstance(model_output, tuple) and len(model_output) == 3:
            # MoE 模式：logits, attention_weights, router_logits
            logits, _, router_logits = model_output
            mtp_logits = None
        else:
            # 标准模式：logits, attention_weights
            logits, _ = model_output
            router_logits = None
            mtp_logits = None

        loss = loss_function(tar_real, logits, router_logits=router_logits, moe_config=moe_config, mtp_logits=mtp_logits, mtp_config=mtp_config)
        acc = token_accuracy(tar_real, logits, pad_id=en_tokenizer.pad_token_id)

        total_loss += loss.item() * inp.size(0)
        total_acc += acc * inp.size(0)
        total_count += inp.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


def save_ckpt(model, optimizer, scheduler, epoch, step, ckpt_dir="checkpoints", tag="latest", use_multi_gpu=False):
    """
    保存 checkpoint
    Args:
        model: nn.Module
        optimizer: torch.optim
        scheduler: torch.optim.lr_scheduler (可选)
        epoch: 当前 epoch
        step: 全局 step
        ckpt_dir: 保存目录
        tag: 保存标识 ("latest", "error", "custom" 等)
        use_multi_gpu: 是否使用多卡训练
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # 对于DataParallel包装的模型，保存原始模型的状态
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model_state,
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
        "use_multi_gpu": use_multi_gpu,
    }

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(ckpt, latest_path)
    # logger.info(f"✅ checkpoint updated: {latest_path}")

    # 1. 默认保存 latest
    if tag == "latest":
        path = os.path.join(ckpt_dir, f"mid_e{epoch}_s{step}.pt")

    elif tag == "error":
        # 避免覆盖，用时间戳
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ckpt_dir, f"error_e{epoch}_s{step}_{ts}.pt")
    else:
        path = os.path.join(ckpt_dir, f"{tag}_e{epoch}_s{step}.pt")

    torch.save(ckpt, path)
    # logger.info(f"✅ checkpoint saved: {path}")
    return path


def load_ckpt(model, optimizer=None, scheduler=None, ckpt_dir="checkpoints", device="cpu", use_multi_gpu=False):
    """
    加载最新 checkpoint
    """
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        logger.info("⚠️ No checkpoint found, training from scratch.")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)

    # 对于DataParallel包装的模型，加载到原始模型
    target_model = model.module if use_multi_gpu else model
    target_model.load_state_dict(ckpt["model"])

    if optimizer: optimizer.load_state_dict(ckpt["optim"])
    if scheduler and ckpt["sched"]: scheduler.load_state_dict(ckpt["sched"])

    saved_multi_gpu = ckpt.get("use_multi_gpu", False)
    logger.info(f"✅ checkpoint loaded (epoch={ckpt['epoch']}, step={ckpt['step']}, multi_gpu={saved_multi_gpu})")
    return ckpt["epoch"], ckpt["step"]


if __name__ == "__main__":
    # 训练配置（直接在代码中配置）
    use_mla = True   # Multi-head Latent Attention
    use_mtp = True   # Multi-Token Prediction
    
    logger.info(f"🚀 Training Configuration:")
    logger.info(f"   - MLA (Multi-head Latent Attention): {use_mla}")
    logger.info(f"   - MTP (Multi-Token Prediction): {use_mtp}")
    logger.info(f"   - MoE: True (fixed)")

    # 0. 常量定义

    # 数据文件地址（相对路径）
    train_path = "tensorflow_datasets/por_en_train.csv"
    val_path = "tensorflow_datasets/por_en_test.csv"
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # 根据是否使用MLA设置不同的checkpoint目录
    if use_mla:
        checkpoint_dir = 'checkpoints'
    else:
        checkpoint_dir = 'checkpoints_no_mla'
    logger.info(f"   - Checkpoint目录: {checkpoint_dir}")

    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小
    min_freq = 2  # 最小词频
    special_tokens = special_tokens  # 特殊符号
    max_length = 64  # 最大序列长度

    # 模型训练超参数
    batch_size = 64  # 批处理数 (增大以充分利用46GB显存)
    warmup_steps = 4000  # warmup steps数
    epochs = 30  # 训练轮数
    # learning_rate = 1.0           # 学习率
    # betas = (0.9, 0.98)           # Adam 的一阶矩（梯度均值）；二阶矩（梯度平方的均值）
    # eps = 1e-9                    # 防止除零错误的小常数
    learning_rate = 1e-4  # 进一步降低学习率以稳定MoE训练，防止梯度爆炸
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01

    # 模型结构
    num_layers = 8  # 模型层数 (对应Qwen-72B的80层)
    d_model = 512  # hidden-size (对应Qwen-72B的4096)
    dff = 2048
    num_heads = 8  # 注意力头数 (对应Qwen-72B的64个Head)
    dropout_rate = 0.1
    
    # # KV Cache 计算相关参数
    # head_dim = d_model // num_heads  # 每个Head的向量维度 = 512/8 = 64 (对应Qwen-72B的128)
    

    # MoE 配置 - 优化以减少梯度不稳定
    use_moe = True  # 是否使用 MoE
    moe_config = MoEConfig(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=d_model,
        intermediate_size=dff,
        hidden_act="silu",
        router_aux_loss_coef=0.005,  # 降低辅助损失权重以减少梯度波动
        use_moe=use_moe,
        n_routed_experts=8,
        routed_scaling_factor=0.8,  # 降低缩放因子以稳定训练
        scoring_func="sigmoid",
        topk_method="noaux_tc",  # 现在支持训练模式
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        n_shared_experts=None,
        moe_intermediate_size=dff,
    )

    # 1. 环境初始化
    device = check_env()
    device, use_multi_gpu, gpu_count = setup_multi_gpu()

    # 2. 加载数据集
    train_dataset, val_dataset = load_translation_dataset(train_path=train_path, val_path=val_path)

    # 3. 构建 Tokenizer
    pt_tokenizer, en_tokenizer = train_and_load_tokenizers(
        train_dataset=train_dataset,
        pt_key="pt",
        en_key="en",
        vocab_size=vocab_size,
        min_freq=min_freq,
        special_tokens=special_tokens,
        save_dir_pt="tok_pt",
        save_dir_en="tok_en",
        max_length=max_length
    )
    test_tokenizers(en_tokenizer=en_tokenizer, pt_tokenizer=pt_tokenizer)

    # MLA 配置
    # use_mla 已在命令行参数中定义
    q_lora_rank = d_model // 2  # Q 的低秩维度，默认为 d_model 的一半
    kv_lora_rank = 4 * (d_model // num_heads)  # KV 的低秩维度，遵循DeepSeek标准：4 × head_dim

    # 4. 构建模型
    input_vocab_size = pt_tokenizer.vocab_size
    target_vocab_size = en_tokenizer.vocab_size

    # 4. 构建模型
    model = Transformer(
        num_layers=num_layers,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate,
        src_padding_idx=pt_tokenizer.pad_token_id if hasattr(pt_tokenizer, "pad_token_id") else None,
        tgt_padding_idx=en_tokenizer.pad_token_id if hasattr(en_tokenizer, "pad_token_id") else None,
        use_rope=True,
        use_moe=use_moe,
        moe_config=moe_config,
        use_mla=use_mla,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
    )

    # 权重初始化
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.8)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    model.apply(init_weights)
    logger.info("✅ 标准 Transformer 模型初始化完成")
    
    # MTP 集成（可选）
    # use_mtp 已在命令行参数中定义
    
    if use_mtp:
        from core.models.deepseek_mtp import DeepSeekMTPConfig, add_mtp_to_transformer
        
        # 创建 MTP 配置
        mtp_config = DeepSeekMTPConfig(
            hidden_size=d_model,
            num_nextn_predict_layers=2,  # MTP 预测层数
            vocab_size=target_vocab_size,
            max_position_embeddings=max_length,
            use_moe=use_moe,
            moe_config=moe_config,
            mtp_loss_weight=0.1,  # MTP 损失权重
        )
        
        # 为现有 Transformer 添加 MTP 功能
        model = add_mtp_to_transformer(model, mtp_config)

    # 5. DP 后端：初始化并包装模型（DataParallel 单进程多GPU）
    p_cfg = ParallelConfig(mode=ParallelMode.dp)
    backend = create_backend(p_cfg)
    backend.init_dist()  # DP 模式下此方法为空操作
    
    # 构建过滤后的训练数据对
    logger.info("开始构建过滤后的训练数据对...")
    
    # 构建过滤后的样本对（这部分逻辑从 build_dataloaders 中提取）
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]
    
    def build_filtered_pairs(hf_split, pt_tok, en_tok, max_len: int):
        pairs = []
        for ex in hf_split:
            pt_ids = encode_with_bos_eos(pt_tok, ex["pt"])
            en_ids = encode_with_bos_eos(en_tok, ex["en"])
            if len(pt_ids) <= max_len and len(en_ids) <= max_len:
                pairs.append((pt_ids, en_ids))
        return pairs
    
    train_pairs = build_filtered_pairs(train_dataset, pt_tokenizer, en_tokenizer, max_length)
    val_pairs = build_filtered_pairs(val_dataset, pt_tokenizer, en_tokenizer, max_length)
    
    logger.info(f"✅ 过滤后数据集: 训练集 {len(train_pairs)} 条, 验证集 {len(val_pairs)} 条")
    
    # 创建 PairsDataset
    class PairsDataset(Dataset):
        def __init__(self, pairs): 
            self.pairs = pairs
        
        def __len__(self): 
            return len(self.pairs)
        
        def __getitem__(self, idx):
            pt_ids, en_ids = self.pairs[idx]
            return {"pt_input_ids": pt_ids, "en_input_ids": en_ids}
    
    filtered_train_dataset = PairsDataset(train_pairs)
    filtered_val_dataset = PairsDataset(val_pairs)
    
    # DP 模式不需要分布式采样器（返回 None）
    train_sampler, val_sampler = backend.get_samplers(filtered_train_dataset, filtered_val_dataset)
    
    # 直接创建 DataLoader（使用过滤后的 dataset 和 sampler）
    def collate_padded(batch):
        def pad_block(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                L = len(s)
                out[i, :L] = torch.tensor(s, dtype=torch.long)
                attn[i, :L] = 1
            return out, attn

        pt_ids_list = [ex["pt_input_ids"] for ex in batch]
        en_ids_list = [ex["en_input_ids"] for ex in batch]
        pt_input_ids, pt_attention_mask = pad_block(pt_ids_list, pt_tokenizer.pad_token_id)
        en_input_ids, en_attention_mask = pad_block(en_ids_list, en_tokenizer.pad_token_id)

        return {
            "pt_input_ids": pt_input_ids,
            "pt_attention_mask": pt_attention_mask,
            "en_input_ids": en_input_ids,
            "en_attention_mask": en_attention_mask,
        }
    
    # DP 模式：单进程多GPU，batch_size 保持不变（DataParallel 会自动分配）
    train_loader2 = DataLoader(
        filtered_train_dataset,
        batch_size=batch_size,  # DP 自动分配到多个 GPU
        shuffle=True,  # DP 模式使用 shuffle
        collate_fn=collate_padded,
        num_workers=4,  # 并行数据加载，避免GPU等待
        prefetch_factor=2,  # 预取2个batch
        persistent_workers=True,  # 保持worker进程，避免重复创建
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader2 = DataLoader(
        filtered_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded,
        num_workers=4,  # 并行数据加载
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    # 测试 DataLoader
    test_dataloaders(train_loader2, val_loader2)
    
    # 用 DataParallel 包装模型
    model, _device = backend.wrap_model(model)
    
    # torch.compile 优化（PyTorch 2.0+）- 可提升10-40%速度
    if hasattr(torch, 'compile'):
        try:
            logger.info("🚀 启用 torch.compile 优化...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("✅ torch.compile 优化已启用")
        except Exception as e:
            logger.warning(f"⚠️ torch.compile 失败，继续使用普通模式: {e}")
    num_training_steps = len(train_loader2) * epochs

    # Fused AdamW（在 torch>=2.0 + CUDA 可用时）
    fused_available = hasattr(optim, "AdamW") and "fused" in optim.AdamW.__init__.__code__.co_varnames
    adamw_kwargs = dict(lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    if fused_available and torch.cuda.is_available():
        adamw_kwargs["fused"] = True
    optimizer = optim.AdamW(
        model.parameters(),
        **adamw_kwargs
    )

    warmup_steps = int(0.15 * num_training_steps)  # 15% 步数用作 warmup（MoE需要更长warmup）
    # 获取学习率调度器 - 使用更激进的衰减
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,  # 保持0.5个周期，让学习率充分衰减
    )

    # 7. 自定义损失函数
    # PyTorch 的 CrossEntropyLoss 默认就支持 from_logits=True
    PAD_ID_TGT = en_tokenizer.pad_token_id
    global loss_object
    loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_ID_TGT)

    # 8. 开始训练
    logger.info(f"✅ 开始训练: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")
    
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_model(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader2,
        val_loader=val_loader2,
        scheduler=scheduler,  # Noam 调度
        device=_device if torch.cuda.is_available() else device,
        log_every=100,
        ckpt_dir=checkpoint_dir,  # 使用动态的checkpoint目录
        ckpt_prefix="transformer",
        tensorboard_dir="runs",  # TensorBoard 日志目录
        moe_config=moe_config,  # MoE 配置
        use_multi_gpu=True,  # DP 多卡训练
        mtp_config=mtp_config if use_mtp else None,  # MTP 配置
    )

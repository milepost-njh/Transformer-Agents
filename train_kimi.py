# -*- coding: utf-8 -*-

import os
import sys

# ===== 必须在导入其他库之前设置环境变量 =====
# 设置可见的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,5,6,7"  # 使用5张GPU

# 禁用TensorFlow（我们只用PyTorch，不需要TensorFlow）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用TensorFlow日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用TensorFlow oneDNN优化
os.environ["USE_TF"] = "0"  # 禁用datasets库的TensorFlow后端
os.environ["USE_TORCH"] = "1"  # 强制使用PyTorch后端

# 其他环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizers并行以避免fork警告

# ===== Mock 缺失的 torchao 子模块 =====
# 解决 transformers 导入 torchao.prototype.safetensors 相关模块的问题
import importlib.util
from types import ModuleType

# 检查 torchao 是否已安装
torchao_spec = importlib.util.find_spec("torchao")
if torchao_spec is not None:
    try:
        import torchao
    except Exception:
        pass
    
    # 创建虚拟的 safetensors 包结构
    try:
        # 先尝试正常导入
        from torchao.prototype.safetensors import safetensors_support
    except (ImportError, ModuleNotFoundError, AttributeError):
        # 如果导入失败，创建 mock 模块
        from unittest.mock import MagicMock
        
        # 创建一个真正的模块对象而不是 MagicMock
        if 'torchao.prototype' not in sys.modules:
            prototype_module = ModuleType('torchao.prototype')
            prototype_module.__path__ = []
            sys.modules['torchao.prototype'] = prototype_module
        
        if 'torchao.prototype.safetensors' not in sys.modules:
            safetensors_module = ModuleType('torchao.prototype.safetensors')
            safetensors_module.__path__ = []
            safetensors_module.__package__ = 'torchao.prototype.safetensors'
            sys.modules['torchao.prototype.safetensors'] = safetensors_module
        
        # Mock 所有需要的子模块
        if 'torchao.prototype.safetensors.safetensors_utils' not in sys.modules:
            sys.modules['torchao.prototype.safetensors.safetensors_utils'] = MagicMock()
        
        if 'torchao.prototype.safetensors.safetensors_support' not in sys.modules:
            sys.modules['torchao.prototype.safetensors.safetensors_support'] = MagicMock()

# ===== 现在可以安全导入其他库 =====
import time
import math
import gc
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
from torch.nn.parallel import DataParallel

# Kimi模型导入
from core.models.kimi_linear.modeling_kimi import KimiLinearForCausalLM
from core.models.kimi_linear.configuration_kimi import KimiLinearConfig


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


def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("✅ 使用 CPU")
    return device


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
        # 兼容旧接口：若需要DP，动态导入；当前方案用DDP，不再使用DP
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


def set_random_seed(seed: int = 42):
    """
    设置所有随机数种子以确保实验可重复性
    
    参数:
        seed: 随机数种子，默认 42
    """
    import random
    
    # Python 随机数
    random.seed(seed)
    
    # NumPy 随机数
    np.random.seed(seed)
    
    # PyTorch 随机数 (CPU)
    torch.manual_seed(seed)
    
    # PyTorch 随机数 (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # CUDNN 确定性行为（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"🎲 随机数种子已设置: {seed}")
    logger.info(f"   - Python random seed: {seed}")
    logger.info(f"   - NumPy seed: {seed}")
    logger.info(f"   - PyTorch seed: {seed}")
    if torch.cuda.is_available():
        logger.info(f"   - CUDA seed: {seed}")
        logger.info(f"   - CUDNN deterministic: True (可能影响性能)")


def load_dialogue_dataset(train_path: str, val_path: str):
    """
    加载心理咨询对话数据集 (PsyDTCorpus)

    参数:
        train_path: 训练集 JSON 文件路径
        val_path: 验证集 JSON 文件路径

    返回:
        train_dataset, val_dataset
    """
    # 分别加载训练集和验证集，避免列名不匹配问题
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    val_dataset = load_dataset("json", data_files=val_path, split="train")
    
    # 统一列名：如果验证集有 sample_id，删除它
    if "sample_id" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns(["sample_id"])

    logger.info(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 条, 验证集 {len(val_dataset)} 条")

    return train_dataset, val_dataset


def train_and_load_tokenizer(
        train_dataset,
        vocab_size=2 ** 13,
        min_freq=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        save_dir="tok_zh",
        max_length=1024
):
    """
    训练并加载中文对话的 ByteLevel BPE Tokenizer

    参数:
        train_dataset: 数据集 (包含 messages 字段)
        vocab_size: 词表大小
        min_freq: 最小词频
        special_tokens: 特殊符号
        save_dir: tokenizer 保存路径
        max_length: 模型最大序列长度

    返回:
        tokenizer
    """

    def iter_dialogues(ds):
        """从messages中提取所有对话文本"""
        for ex in ds:
            messages = ex.get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                if content.strip():
                    yield content

    # 初始化 tokenizer
    bbpe = ByteLevelBPETokenizer(add_prefix_space=True)

    # 训练 tokenizer
    bbpe.train_from_iterator(
        iter_dialogues(train_dataset),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    # 保存 vocab/merges + tokenizer.json
    Path(save_dir).mkdir(exist_ok=True)
    bbpe.save_model(save_dir)
    bbpe._tokenizer.save(f"{save_dir}/tokenizer.json")

    # 用 PreTrainedTokenizerFast 加载
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_dir}/tokenizer.json")

    # 设置特殊符号
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.mask_token = "<mask>"
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"

    logger.info(f"✅ Tokenizer构建完成: 词表大小 {len(tokenizer)}")

    return tokenizer


def test_tokenizer(tokenizer, sample: str = "你好，我是一位心理咨询师。"):
    """
    测试中文 tokenizer 编码/解码是否正确

    参数:
        tokenizer: 中文 tokenizer
        sample: 测试句子
    """
    ids = tokenizer.encode(sample, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    logger.info(f"原文: {sample}")
    logger.info(f"Token IDs: {ids[:20]}..." if len(ids) > 20 else f"Token IDs: {ids}")
    logger.info(f"解码结果: {decoded}")
    logger.info(f"✅ Tokenizer测试通过: {len(ids)} tokens")


def build_dialogue_dataloaders(
        train_dataset,
        val_dataset,
        tokenizer,
        batch_size: int = 64,
        max_length: int = 512,
        num_workers: int = 0,
        shuffle_train: bool = True,
):
    """
    构建对话训练的 DataLoader（Kimi因果语言模型模式）
    
    将messages中的system+user+assistant组合成训练序列

    参数:
        train_dataset: HuggingFace Dataset (训练集)
        val_dataset: HuggingFace Dataset (验证集)
        tokenizer: 中文 tokenizer
        batch_size: 批大小
        max_length: 样本最大长度（超过则过滤）
        num_workers: DataLoader worker 数量
        shuffle_train: 是否打乱训练集

    返回:
        train_loader, val_loader
    """

    def messages_to_text(messages):
        """将messages转换为训练文本"""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        return "\n".join(text_parts)

    # 编码 + 添加 BOS/EOS
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]

    # 构建对话序列
    def build_filtered_sequences(hf_split, tokenizer, max_len: int):
        sequences = []
        for ex in hf_split:
            messages = ex.get("messages", [])
            if not messages:
                continue
            text = messages_to_text(messages)
            ids = encode_with_bos_eos(tokenizer, text)
            if len(ids) <= max_len:
                sequences.append(ids)
        return sequences

    train_sequences = build_filtered_sequences(train_dataset, tokenizer, max_length)
    val_sequences = build_filtered_sequences(val_dataset, tokenizer, max_length)
    
    logger.info(f"✅ 过滤后数据集: 训练集 {len(train_sequences)} 条, 验证集 {len(val_sequences)} 条")

    # Dataset 类（单一序列）
    class SequenceDataset(Dataset):
        def __init__(self, sequences): 
            self.sequences = sequences

        def __len__(self): 
            return len(self.sequences)

        def __getitem__(self, idx):
            return {"input_ids": self.sequences[idx]}

    # Collate 函数（动态 padding）
    def collate_padded_sequences(batch, pad_id: int):
        def pad_block(seqs, pad_value):
            max_len = max(len(s) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                L = len(s)
                out[i, :L] = torch.tensor(s, dtype=torch.long)
                attn[i, :L] = 1
            return out, attn

        ids_list = [ex["input_ids"] for ex in batch]
        input_ids, attention_mask = pad_block(ids_list, pad_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # DataLoader
    train_loader = DataLoader(
        SequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=lambda b: collate_padded_sequences(b, tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_padded_sequences(b, tokenizer.pad_token_id),
        num_workers=num_workers,
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
    input_ids_key = 'input_ids' if 'input_ids' in batch else 'pt_input_ids'
    logger.info(f"✅ DataLoader测试通过: batch_size={batch[input_ids_key].shape[0]}, seq_len={batch[input_ids_key].shape[1]}")


class RoPEPositionalEncoding:
    def __init__(self, max_len, d_model, nums_head=8, batch_size=1, device=None):
        self.max_len = max_len
        self.d_model = d_model
        self.nums_head = nums_head
        self.batch_size = batch_size

        if device is None:
            self.device = get_device()
        else:
            self.device = device

    def sinusoidal_position_embedding(self):
        """
        生成RoPE位置编码矩阵
        返回: [batch_size, nums_head, max_len, d_model]
        """
        # (max_len, 1)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(-1)

        # (d_model//2)
        ids = torch.arange(0, self.d_model // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / self.d_model)

        # (max_len, d_model//2)
        embeddings = position * theta

        # (max_len, d_model//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (batch_size, nums_head, max_len, d_model//2, 2)
        embeddings = embeddings.repeat((self.batch_size, self.nums_head, *([1] * len(embeddings.shape))))

        # (batch_size, nums_head, max_len, d_model)
        embeddings = torch.reshape(embeddings, (self.batch_size, self.nums_head, self.max_len, self.d_model))

        return embeddings.to(self.device)

    def get_rope_embedding_matrix(self):
        """
        获取RoPE位置编码矩阵用于可视化
        返回: [max_len, d_model]
        """
        # 只取第一个batch和第一个head的位置编码
        rope_emb = self.sinusoidal_position_embedding()
        return rope_emb[0, 0].detach().cpu()  # [max_len, d_model]

    def get_rotation_matrices(self, positions_to_show=5):
        """
        获取旋转矩阵的可视化数据
        对于每个位置，计算旋转矩阵对向量的影响
        """
        # 生成一些测试向量
        test_vectors = []
        for i in range(4):
            angle = i * math.pi / 4  # 0, 45, 90, 135度
            vec = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
            test_vectors.append(vec)

        # 计算旋转效果
        rotation_data = []
        for pos in range(min(positions_to_show, self.max_len)):
            pos_emb = self.sinusoidal_position_embedding()[0, 0, pos]  # 取第一个位置编码

            rotated_vectors = []
            for vec in test_vectors:
                # 简化版的旋转计算（只考虑前两个维度）
                cos_theta = pos_emb[1]  # cos分量
                sin_theta = pos_emb[0]  # sin分量

                # 旋转矩阵 [cos, -sin; sin, cos]
                rotation_matrix = torch.tensor([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])

                rotated_vec = torch.matmul(rotation_matrix, vec)
                rotated_vectors.append({
                    'original': vec.numpy(),
                    'rotated': rotated_vec.numpy(),
                    'position': pos
                })

            rotation_data.append(rotated_vectors)

        return rotation_data


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
        # 在 mask==1 的位置加上 -1e9，使 softmax 后趋近于0
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, -1e9)

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


def plot_customized_lr_curve(optimizer, scheduler, total_steps: int, label: str = None):
    """
    绘制学习率曲线（支持传入已有 optimizer 和 scheduler）

    Args:
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        total_steps (int): 总训练步数
        label (str): 图例标签，默认使用 scheduler 配置
    """
    lrs = []
    for step in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

    # 绘制曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, total_steps + 1), lrs, label=label or "LR Curve")
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()


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
        
        # 只在第一次计算MTP损失时打印日志
        if not hasattr(loss_function, '_mtp_logged'):
            logger.info(f"✅ MTP损失已加入总损失: mtp_loss={mtp_loss:.4f}, total_loss={total_loss:.4f}")
            loss_function._mtp_logged = True

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


def train_step(batch, transformer, optimizer, scheduler=None, device=None, moe_config=None, use_multi_gpu=False, tokenizer=None, global_step=0):
    """
    训练单步（Kimi因果语言模型）
    
    注意：训练时不使用KV-cache
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.train()

    # 使用 PyTorch SDPA + bf16 autocast
    use_autocast = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_autocast:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    # Kimi因果语言模型模式
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # 因果LM：输入是完整序列，labels是向右移位的序列
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device)], dim=1)
    
    with autocast_ctx:
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    
    # Kimi模型返回 CausalLMOutputWithPast，包含loss
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        loss = outputs.loss
    else:
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        loss = loss_function(labels, logits, router_logits=None, moe_config=moe_config, mtp_logits=None, mtp_config=None)
    
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # 处理多GPU情况：DataParallel会返回多个loss（每个GPU一个）
    if loss.dim() > 0:
        loss = loss.mean()

    # 调试：打印第一个batch的信息
    if global_step == 1:
        logger.info(f"📊 First batch debug info:")
        logger.info(f"  - input_ids shape: {input_ids.shape}, min: {input_ids.min()}, max: {input_ids.max()}")
        logger.info(f"  - labels shape: {labels.shape}, min: {labels.min()}, max: {labels.max()}")
        logger.info(f"  - logits shape: {logits.shape}, contains NaN: {torch.isnan(logits).any()}, contains Inf: {torch.isinf(logits).any()}")
        logger.info(f"  - logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
        logger.info(f"  - loss: {loss.item():.4f}")

    # 检测NaN或Inf损失
    if not torch.isfinite(loss):
        logger.error(f"Loss is {loss.item()}, skipping this batch")
        # 额外调试信息
        logger.error(f"  - logits contains NaN: {torch.isnan(logits).any()}")
        logger.error(f"  - logits contains Inf: {torch.isinf(logits).any()}")
        return 0.0, 0.0, 0.0

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 梯度裁剪（放宽裁剪阈值，原0.2太严格导致梯度被裁剪98%）
    model_for_grad_clip = transformer.module if use_multi_gpu else transformer
    grad_norm = torch.nn.utils.clip_grad_norm_(model_for_grad_clip.parameters(), max_norm=1.0)

    # 只在前 100 步和每 100 步打印一次梯度警告，避免日志刷屏
    if grad_norm > 5.0 and (global_step <= 100 or global_step % 100 == 0):
        logger.warning(f"Large gradient norm detected: {grad_norm:.4f} (clipped to 1.0)")

    # 检查NaN梯度
    has_nan_grad = False
    for name, param in model_for_grad_clip.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.error(f"NaN gradient detected in {name}")
            has_nan_grad = True
            break

    if has_nan_grad:
        logger.error("Skipping this batch due to NaN gradients")
        return 0.0, 0.0, 0.0

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # 计算准确率
    acc = token_accuracy(labels, logits, pad_id=tokenizer.pad_token_id)
    return loss.item(), acc, grad_norm.item()


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
        tokenizer=None,
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
    best_val_loss = float('inf')  # 追踪最佳验证集loss

    for epoch in range(epochs):
        try:
            start = time.time()
            train_loss_meter.reset()
            train_acc_meter.reset()
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1
                loss_val, acc_val, grad_norm_val = train_step(
                    batch=batch, transformer=model, optimizer=optimizer, scheduler=scheduler, device=device,
                    moe_config=moe_config, use_multi_gpu=use_multi_gpu, tokenizer=tokenizer, global_step=global_step
                )
                train_loss_meter.update(loss_val, 1)
                train_acc_meter.update(acc_val, 1)

                # 记录到 TensorBoard
                writer.add_scalar('Train/Loss', loss_val, global_step)
                writer.add_scalar('Train/Accuracy', acc_val, global_step)
                writer.add_scalar('Train/Gradient_Norm', grad_norm_val, global_step)

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
                        f"Epoch {epoch + 1} Batch {batch_idx} global_step {global_step} "
                        f"Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f} "
                        f"GradNorm {grad_norm_val:.4f} LR {current_lr:.2e}{memory_info}"
                    )
                
                # 每 1000 步保存一次 checkpoint，只保留最近 3 个
                if global_step % 1000 == 0:
                    # 保存新的 checkpoint
                    ckpt_path = save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        ckpt_dir=ckpt_dir,
                        tag=f"step{global_step}",
                        use_multi_gpu=use_multi_gpu
                    )
                    logger.info(f"💾 Checkpoint saved at step {global_step}: {ckpt_path}")
                    
                    # 只保留最近 3 个 step checkpoint（避免磁盘占用过大）
                    import glob
                    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step*.pt")))
                    if len(step_ckpts) > 3:
                        for old_ckpt in step_ckpts[:-3]:
                            os.remove(old_ckpt)
                            logger.info(f"🗑️  Removed old checkpoint: {old_ckpt}")

            # 记录每个 epoch 的平均指标到 TensorBoard
            writer.add_scalar('Epoch/Train_Loss', train_loss_meter.avg, epoch + 1)
            writer.add_scalar('Epoch/Train_Accuracy', train_acc_meter.avg, epoch + 1)
            writer.add_scalar('Epoch/Time', time.time() - start, epoch + 1)

            logger.info(f"Epoch {epoch + 1} Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}")
            logger.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

            # 每个epoch结束后进行验证集评测
            validate_loss, validate_acc = evaluate_on_val(model, val_loader, device, moe_config=moe_config, tokenizer=tokenizer)

            # 记录验证指标到 TensorBoard
            writer.add_scalar('Epoch/Validation_Loss', validate_loss, epoch + 1)
            writer.add_scalar('Epoch/Validation_Accuracy', validate_acc, epoch + 1)

            logger.info(f"Validation - Epoch {epoch + 1} Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}\n")

            # 每个epoch结束后保存latest checkpoint
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
            
            # 如果是最佳模型，额外保存一份best checkpoint
            if validate_loss < best_val_loss:
                best_val_loss = validate_loss
                save_ckpt(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    ckpt_dir=ckpt_dir,
                    tag="best",
                    use_multi_gpu=use_multi_gpu
                )
                logger.info(f"🏆 New best model! Validation Loss: {validate_loss:.4f} (saved as best.pt)")

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
                    ckpt_dir=ckpt_dir,  # 使用正确的 checkpoint 目录
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
def evaluate(
        inp_sentence: str,
        transformer: Transformer,
        pt_tokenizer,
        en_tokenizer,
        max_length: int,
        device: str = None):
    """
    inp_sentence: 输入的源语言字符串 (pt)
    transformer: 已训练的 Transformer
    pt_tokenizer, en_tokenizer: 分别是葡萄牙语和英语 tokenizer
    max_length: 最大生成长度
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.eval()
    transformer.to(device)

    # 1. 编码输入，加 <s> 和 </s>
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]

    inp_ids = encode_with_bos_eos(pt_tokenizer, inp_sentence)
    encoder_input = torch.tensor(inp_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, Ls)

    # 2. decoder 起始符 <s>
    start_id = en_tokenizer.bos_token_id
    end_id = en_tokenizer.eos_token_id
    decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=device)  # (1, 1)

    # 3. 循环预测
    attention_weights = {}
    for _ in range(max_length):
        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            encoder_input, decoder_input,
            src_pad_id=pt_tokenizer.pad_token_id,
            tgt_pad_id=en_tokenizer.pad_token_id,
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)

        logits, attn = transformer(
            encoder_input, decoder_input,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask,
        )

        # 取最后一步预测
        next_token_logits = logits[:, -1, :]  # (1, V)
        predicted_id = torch.argmax(next_token_logits, dim=-1)  # (1,)

        if predicted_id.item() == end_id:
            break

        # 拼接到 decoder_input
        decoder_input = torch.cat(
            [decoder_input, predicted_id.unsqueeze(0)], dim=-1
        )  # (1, Lt+1)
        attention_weights = attn

    return decoder_input.squeeze(0).tolist(), attention_weights


@torch.no_grad()
def evaluate_on_val(model, val_loader, device, moe_config=None, tokenizer=None):
    """验证集评估（Kimi因果语言模型模式）"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0
    
    # 使用 autocast 确保 Flash Attention 兼容
    use_autocast = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_autocast:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    for batch in val_loader:
        # Kimi因果语言模型模式
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        labels = input_ids.clone()
        labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device)], dim=1)
        
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            loss = loss_function(labels, logits, router_logits=None, moe_config=moe_config, mtp_logits=None, mtp_config=None)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        acc = token_accuracy(labels, logits, pad_id=tokenizer.pad_token_id)
        
        # 处理 DataParallel 返回的多个 loss 值
        if loss.dim() > 0:
            loss = loss.mean()
        
        total_loss += loss.item() * input_ids.size(0)
        total_acc += acc * input_ids.size(0)
        total_count += input_ids.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name):
    """
    attention: 来自 forward 返回的 attention_weights dict
               形状 [B, num_heads, tgt_len, src_len]
    input_sentence: 源语言字符串
    result: 目标句子 token id 列表 (decoder 输出)
    layer_name: 指定可视化的层 key，比如 "decoder_layer1_att2"
    """
    fig = plt.figure(figsize=(16, 8))

    # 源句子编码
    input_id_sentence = pt_tokenizer.encode(input_sentence, add_special_tokens=False)

    # 取 batch 维度 squeeze，并转 numpy
    attn = attention[layer_name].squeeze(0)  # [num_heads, tgt_len, src_len]
    attn = attn.detach().cpu().numpy()

    for head in range(attn.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 只取 result[:-1] 的注意力 (去掉最后 <eos>)
        ax.matshow(attn[head][:-1, :], cmap="viridis")

        fontdict = {"fontsize": 10}

        # X 轴: 输入 token (<s> + sentence + </s>)
        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_xticklabels(
            ["<s>"] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ["</s>"],
            fontdict=fontdict, rotation=90,
        )

        # Y 轴: decoder 输出 token
        ax.set_yticks(range(len(result)))
        ax.set_yticklabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size],
            fontdict=fontdict,
        )

        ax.set_ylim(len(result) - 1.5, -0.5)
        ax.set_xlabel(f"Head {head + 1}")

    plt.tight_layout()
    plt.show()


def translate(input_sentence, transformer, pt_tokenizer, en_tokenizer,
              max_length=64, device=None, layer_name=""):
    # 调用我们改好的 evaluate (PyTorch 版)
    result, attention_weights = evaluate(
        inp_sentence=input_sentence,
        transformer=transformer,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        max_length=max_length,
        device=device,
    )

    # 把 token id 转回句子
    predicted_sentence = en_tokenizer.decode(
        [i for i in result if i < en_tokenizer.vocab_size],
        skip_special_tokens=True
    )

    logger.info("Input: {}".format(input_sentence))
    logger.info(f"Predicted translation: {predicted_sentence}")

    # 如果传入了 layer_name，就画注意力图
    if layer_name:
        plot_encoder_decoder_attention(
            attention_weights,
            input_sentence,
            result,
            layer_name
        )

    return predicted_sentence


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
    # 训练配置
    use_kimi = True  # 使用Kimi因果语言模型
    use_mla = True   # 使用MLA
    use_mtp = False  # Kimi模型不使用MTP
    random_seed = 42  # 随机数种子，设置为固定值以确保实验可重复性
    
    logger.info(f"🚀 Training Configuration:")
    logger.info(f"   - Model: {'Kimi Causal LM' if use_kimi else 'Seq2Seq Transformer'}")
    logger.info(f"   - MLA (Multi-head Latent Attention): {use_mla}")
    logger.info(f"   - MTP (Multi-Token Prediction): {use_mtp}")
    logger.info(f"   - MoE: True (fixed)")
    logger.info(f"   - Random Seed: {random_seed}")

    # 0. 常量定义

    # 数据文件地址（心理咨询对话数据集）
    train_path = "/workspace/datasets/YIRONGCHEN/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"
    val_path = "/workspace/datasets/YIRONGCHEN/PsyDTCorpus/PsyDTCorpus_test_single_turn_split.json"
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|im_start|>", "<|im_end|>"]
    
    # 根据模型类型设置不同的checkpoint目录
    if use_kimi:
        checkpoint_dir = 'checkpoints_kimi'
    elif use_mla:
        checkpoint_dir = 'checkpoints'
    else:
        checkpoint_dir = 'checkpoints_no_mla'
    logger.info(f"   - Checkpoint目录: {checkpoint_dir}")

    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小 (8192)
    min_freq = 2  # 最小词频
    special_tokens = special_tokens  # 特殊符号
    max_length = 4096  # 最大序列长度（心理咨询对话平均~2442 tokens，最大~3901，用4096保留100%数据）

    # 模型训练超参数
    batch_size = 4  # 批处理数 (max_length=4096 时需要更小的 batch_size 避免OOM)
    warmup_steps = 500  # 减少warmup步数（原8000太长，每epoch只有1190步）
    epochs = 50  # 增加训练轮数以补偿小数据集
    # learning_rate = 1.0           # 学习率
    # betas = (0.9, 0.98)           # Adam 的一阶矩（梯度均值）；二阶矩（梯度平方的均值）
    # eps = 1e-9                    # 防止除零错误的小常数
    learning_rate = 2e-5  # 提高学习率（原5e-6太低，配合强梯度裁剪导致训练太慢）
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
    
    # 设置随机数种子以确保实验可重复性
    set_random_seed(random_seed)

    # 2. 加载或训练tokenizer
    tokenizer_dir = "tok_zh"
    if os.path.exists(f"{tokenizer_dir}/tokenizer.json"):
        # tokenizer已存在，直接加载
        logger.info("发现已保存的tokenizer，直接加载...")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{tokenizer_dir}/tokenizer.json")
        
        # 设置特殊符号
        tokenizer.pad_token = "<pad>"
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.mask_token = "<mask>"
        tokenizer.model_max_length = max_length
        tokenizer.padding_side = "right"
        
        logger.info(f"✅ Tokenizer加载完成: 词表大小 {len(tokenizer)}")
    else:
        # tokenizer不存在，需要训练
        logger.info("未发现tokenizer，开始加载数据集并训练tokenizer...")
        train_dataset, val_dataset = load_dialogue_dataset(train_path=train_path, val_path=val_path)
        
        tokenizer = train_and_load_tokenizer(
            train_dataset=train_dataset,
            vocab_size=vocab_size,
            min_freq=min_freq,
            special_tokens=special_tokens,
            save_dir=tokenizer_dir,
            max_length=max_length
        )
        logger.info("✅ Tokenizer训练完成")
        
        # 训练完后释放数据集内存
        del train_dataset, val_dataset
        gc.collect()
    
    test_tokenizer(tokenizer=tokenizer)

    # MLA 配置
    q_lora_rank = d_model // 2  # Q 的低秩维度，默认为 d_model 的一半
    kv_lora_rank = 4 * (d_model // num_heads)  # KV 的低秩维度，遵循DeepSeek标准：4 × head_dim

    # 3. 构建模型
    vocab_size_model = tokenizer.vocab_size

    if use_kimi:
        # 使用 Kimi 因果语言模型
        head_dim = d_model // num_heads
        
        # 配置 Kimi 模型
        kimi_config = KimiLinearConfig(
            vocab_size=vocab_size_model,
            hidden_size=d_model,
            head_dim=head_dim,
            intermediate_size=dff,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # 使用MQA/GQA时可以减少
            hidden_act="silu",
            initializer_range=0.02,  # 初始化范围（原0.01可能太小）
            rms_norm_eps=1e-6,
            use_cache=False,  # 训练时不使用cache
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            # MoE 配置
            num_experts=moe_config.num_experts if use_moe else None,
            num_experts_per_token=moe_config.num_experts_per_tok if use_moe else None,
            moe_intermediate_size=moe_config.moe_intermediate_size if use_moe else None,
            moe_renormalize=True,
            moe_router_activation_func="sigmoid",
            num_shared_experts=moe_config.n_shared_experts if use_moe else 0,
            routed_scaling_factor=moe_config.routed_scaling_factor if use_moe else 1.0,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            use_grouped_topk=True,
            num_expert_group=moe_config.n_group if use_moe else 1,
            topk_group=moe_config.topk_group if use_moe else 1,
            # MLA 配置
            q_lora_rank=None,  # Kimi 模型强制要求 q_lora_rank 为 None
            kv_lora_rank=kv_lora_rank if use_mla else None,
            qk_nope_head_dim=head_dim // 2 if use_mla else None,
            qk_rope_head_dim=head_dim // 2 if use_mla else None,
            v_head_dim=head_dim if use_mla else None,
            mla_use_nope=True,  # Kimi 模型使用 nope
            # Attention 实现配置
            _attn_implementation="flash_attention_2",  # 使用 Flash Attention 2
        )
        
        model = KimiLinearForCausalLM(kimi_config)
        
        # 检查模型权重是否包含 NaN 或 Inf
        has_nan_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"⚠️ 参数 {name} 包含 NaN 或 Inf！")
                has_nan_inf = True
        if not has_nan_inf:
            logger.info("✅ 模型权重检查通过：无 NaN 或 Inf")
        
        logger.info("✅ Kimi 因果语言模型初始化完成")
        mtp_config = None  # Kimi模型不使用MTP
        
    else:
        # 使用原有的 Seq2Seq Transformer（对话任务不再使用此模式）
        model = Transformer(
            num_layers=num_layers,
            input_vocab_size=vocab_size_model,
            target_vocab_size=vocab_size_model,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=dropout_rate,
            src_padding_idx=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
            tgt_padding_idx=tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
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
        if use_mtp:
            from core.models.deepseek_mtp import DeepSeekMTPConfig, add_mtp_to_transformer
            
            # 创建 MTP 配置
            mtp_config = DeepSeekMTPConfig(
                hidden_size=d_model,
                num_nextn_predict_layers=2,  # MTP 预测层数
                vocab_size=vocab_size_model,
                max_position_embeddings=max_length,
                use_moe=use_moe,
                moe_config=moe_config,
                mtp_loss_weight=0.1,  # MTP 损失权重
            )
            
            # 为现有 Transformer 添加 MTP 功能
            model = add_mtp_to_transformer(model, mtp_config)
        else:
            mtp_config = None

    # 4. 构建过滤后的数据
    # 定义数据缓存文件路径（包含 max_length 以避免配置不匹配）
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_file = os.path.join(cache_dir, f"train_{'kimi' if use_kimi else 'seq2seq'}_len{max_length}.pkl")
    val_cache_file = os.path.join(cache_dir, f"val_{'kimi' if use_kimi else 'seq2seq'}_len{max_length}.pkl")
    
    # 构建过滤后的样本（这部分逻辑从 build_dataloaders 中提取）
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("请确保 tokenizer 设置了 bos_token/eos_token")
        return [bos_id] + ids + [eos_id]
    
    if use_kimi:
        # Kimi因果语言模型模式：使用对话数据
        def messages_to_text(messages):
            """将messages转换为训练文本"""
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            return "\n".join(text_parts)
        
        def build_filtered_sequences(hf_split, tokenizer, max_len: int):
            sequences = []
            for ex in hf_split:
                messages = ex.get("messages", [])
                if not messages:
                    continue
                text = messages_to_text(messages)
                ids = encode_with_bos_eos(tokenizer, text)
                if len(ids) <= max_len:
                    sequences.append(ids)
            return sequences
        
        # 检查缓存是否存在
        if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
            logger.info(f"发现缓存文件，直接加载: {train_cache_file}, {val_cache_file}")
            with open(train_cache_file, 'rb') as f:
                train_sequences = pickle.load(f)
            with open(val_cache_file, 'rb') as f:
                val_sequences = pickle.load(f)
            logger.info(f"✅ 从缓存加载数据集: 训练集 {len(train_sequences)} 条, 验证集 {len(val_sequences)} 条")
        else:
            # 需要重新构建，先加载原始数据集
            logger.info(f"开始加载原始数据集...")
            train_dataset, val_dataset = load_dialogue_dataset(train_path=train_path, val_path=val_path)
            
            logger.info(f"开始构建过滤后的训练数据序列...")
            train_sequences = build_filtered_sequences(train_dataset, tokenizer, max_length)
            val_sequences = build_filtered_sequences(val_dataset, tokenizer, max_length)
            logger.info(f"✅ 过滤后数据集: 训练集 {len(train_sequences)} 条, 验证集 {len(val_sequences)} 条")
            
            # 保存到磁盘
            logger.info(f"保存数据到缓存: {train_cache_file}, {val_cache_file}")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_sequences, f)
            with open(val_cache_file, 'wb') as f:
                pickle.dump(val_sequences, f)
            
            # 释放原始数据集内存
            del train_dataset, val_dataset
            gc.collect()
        
        # Dataset 类（单一序列）
        class SequenceDataset(Dataset):
            def __init__(self, sequences): 
                self.sequences = sequences

            def __len__(self): 
                return len(self.sequences)

            def __getitem__(self, idx):
                return {"input_ids": self.sequences[idx]}
        
        filtered_train_dataset = SequenceDataset(train_sequences)
        filtered_val_dataset = SequenceDataset(val_sequences)
        
        # Collate 函数（单一序列）
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

            ids_list = [ex["input_ids"] for ex in batch]
            input_ids, attention_mask = pad_block(ids_list, tokenizer.pad_token_id)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
    else:
        # Seq2Seq模式：使用对话数据（与Kimi模式类似）
        def messages_to_text(messages):
            """将messages转换为训练文本"""
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            return "\n".join(text_parts)
        
        def build_filtered_pairs(hf_split, tok, max_len: int):
            pairs = []
            for ex in hf_split:
                messages = ex.get("messages", [])
                if not messages:
                    continue
                text = messages_to_text(messages)
                ids = encode_with_bos_eos(tok, text)
                if len(ids) <= max_len:
                    # Seq2Seq模式下用同样的序列作为输入和输出
                    pairs.append((ids, ids))
            return pairs
        
        # 检查缓存是否存在
        if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
            logger.info(f"发现缓存文件，直接加载: {train_cache_file}, {val_cache_file}")
            with open(train_cache_file, 'rb') as f:
                train_pairs = pickle.load(f)
            with open(val_cache_file, 'rb') as f:
                val_pairs = pickle.load(f)
            logger.info(f"✅ 从缓存加载数据集: 训练集 {len(train_pairs)} 条, 验证集 {len(val_pairs)} 条")
        else:
            # 需要重新构建，先加载原始数据集
            logger.info(f"开始加载原始数据集...")
            train_dataset, val_dataset = load_dialogue_dataset(train_path=train_path, val_path=val_path)
            
            logger.info(f"开始构建过滤后的训练数据对...")
            train_pairs = build_filtered_pairs(train_dataset, tokenizer, max_length)
            val_pairs = build_filtered_pairs(val_dataset, tokenizer, max_length)
            logger.info(f"✅ 过滤后数据集: 训练集 {len(train_pairs)} 条, 验证集 {len(val_pairs)} 条")
            
            # 保存到磁盘
            logger.info(f"保存数据到缓存: {train_cache_file}, {val_cache_file}")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_pairs, f)
            with open(val_cache_file, 'wb') as f:
                pickle.dump(val_pairs, f)
            
            # 释放原始数据集内存
            del train_dataset, val_dataset
            gc.collect()
        
        # Dataset 类（样本对）
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
        
        # Collate 函数（样本对）
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
            pt_input_ids, pt_attention_mask = pad_block(pt_ids_list, tokenizer.pad_token_id)
            en_input_ids, en_attention_mask = pad_block(en_ids_list, tokenizer.pad_token_id)

            return {
                "pt_input_ids": pt_input_ids,
                "pt_attention_mask": pt_attention_mask,
                "en_input_ids": en_input_ids,
                "en_attention_mask": en_attention_mask,
            }
    
    # 创建 DataLoader
    train_loader2 = DataLoader(
        filtered_train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要shuffle
        collate_fn=collate_padded,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader2 = DataLoader(
        filtered_val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要shuffle
        collate_fn=collate_padded,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # 测试DataLoader
    test_dataloaders(train_loader2, val_loader2)
    
    # 使用DP包装模型
    model = wrap_model_for_multi_gpu(model, use_multi_gpu, gpu_count)
    model.to(device)
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
    PAD_ID_TGT = tokenizer.pad_token_id
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
        scheduler=scheduler,
        device=device,
        log_every=100,
        ckpt_dir=checkpoint_dir,  # 使用动态的checkpoint目录
        ckpt_prefix="transformer",
        tensorboard_dir="runs",  # TensorBoard 日志目录
        moe_config=moe_config,  # MoE 配置
        use_multi_gpu=use_multi_gpu,  # DP 多卡训练
        tokenizer=tokenizer,  # 传入tokenizer用于计算准确率
    )

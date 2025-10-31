#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本：训练tokenizer并构建数据缓存
分步执行，避免内存溢出
"""

import os
import sys
import gc
import pickle

# 禁用TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from loguru import logger

# 配置参数
train_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/por_en_train.csv"
val_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/por_en_test.csv"
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
vocab_size = 2 ** 13
min_freq = 2
max_length = 64

def load_translation_dataset(train_path: str, val_path: str, delimiter: str = "\t"):
    """加载翻译数据集"""
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

def train_tokenizers():
    """步骤1: 训练tokenizer"""
    if os.path.exists("tok_pt/tokenizer.json") and os.path.exists("tok_en/tokenizer.json"):
        logger.info("✅ Tokenizer已存在，跳过训练")
        return
    
    logger.info("📚 开始训练tokenizer...")
    
    # 加载数据集
    train_dataset, val_dataset = load_translation_dataset(train_path, val_path)
    
    def iter_lang(ds, key):
        for ex in ds:
            txt = ex[key]
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8")
            yield txt
    
    # 训练葡语tokenizer
    logger.info("🔤 训练葡语tokenizer...")
    pt_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)
    pt_bbpe.train_from_iterator(
        iter_lang(train_dataset, "pt"),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )
    Path("tok_pt").mkdir(exist_ok=True)
    pt_bbpe.save_model("tok_pt")
    pt_bbpe._tokenizer.save("tok_pt/tokenizer.json")
    logger.info("✅ 葡语tokenizer训练完成")
    
    # 释放内存
    del pt_bbpe
    gc.collect()
    
    # 训练英语tokenizer
    logger.info("🔤 训练英语tokenizer...")
    en_bbpe = ByteLevelBPETokenizer(add_prefix_space=True)
    en_bbpe.train_from_iterator(
        iter_lang(train_dataset, "en"),
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )
    Path("tok_en").mkdir(exist_ok=True)
    en_bbpe.save_model("tok_en")
    en_bbpe._tokenizer.save("tok_en/tokenizer.json")
    logger.info("✅ 英语tokenizer训练完成")
    
    # 释放内存
    del en_bbpe, train_dataset, val_dataset
    gc.collect()

def build_data_cache():
    """步骤2: 构建数据缓存"""
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_file = os.path.join(cache_dir, "train_kimi.pkl")
    val_cache_file = os.path.join(cache_dir, "val_kimi.pkl")
    
    if os.path.exists(train_cache_file) and os.path.exists(val_cache_file):
        logger.info("✅ 数据缓存已存在，跳过构建")
        return
    
    logger.info("💾 开始构建数据缓存...")
    
    # 加载tokenizer
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
    en_tokenizer.pad_token = "<pad>"
    en_tokenizer.unk_token = "<unk>"
    en_tokenizer.bos_token = "<s>"
    en_tokenizer.eos_token = "</s>"
    en_tokenizer.mask_token = "<mask>"
    en_tokenizer.model_max_length = max_length
    en_tokenizer.padding_side = "right"
    
    # 加载数据集
    train_dataset, val_dataset = load_translation_dataset(train_path, val_path)
    
    def encode_with_bos_eos(tokenizer, text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        return [bos_id] + ids + [eos_id]
    
    def build_filtered_sequences(hf_split, tokenizer, max_len: int):
        sequences = []
        for ex in hf_split:
            ids = encode_with_bos_eos(tokenizer, ex["en"])
            if len(ids) <= max_len:
                sequences.append(ids)
        return sequences
    
    # 构建训练集
    logger.info("📝 处理训练集...")
    train_sequences = build_filtered_sequences(train_dataset, en_tokenizer, max_length)
    logger.info(f"✅ 训练集过滤完成: {len(train_sequences)} 条")
    
    with open(train_cache_file, 'wb') as f:
        pickle.dump(train_sequences, f)
    logger.info(f"💾 训练集缓存已保存: {train_cache_file}")
    
    # 释放训练集内存
    del train_sequences, train_dataset
    gc.collect()
    
    # 构建验证集
    logger.info("📝 处理验证集...")
    val_sequences = build_filtered_sequences(val_dataset, en_tokenizer, max_length)
    logger.info(f"✅ 验证集过滤完成: {len(val_sequences)} 条")
    
    with open(val_cache_file, 'wb') as f:
        pickle.dump(val_sequences, f)
    logger.info(f"💾 验证集缓存已保存: {val_cache_file}")
    
    # 释放内存
    del val_sequences, val_dataset, en_tokenizer
    gc.collect()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 开始数据预处理")
    logger.info("=" * 60)
    
    # 步骤1: 训练tokenizer
    logger.info("\n📌 步骤1: 训练tokenizer")
    train_tokenizers()
    
    # 步骤2: 构建数据缓存
    logger.info("\n📌 步骤2: 构建数据缓存")
    build_data_cache()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 数据预处理完成！现在可以运行 python train_tmp.py")
    logger.info("=" * 60)


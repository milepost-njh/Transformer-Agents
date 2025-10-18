# -*- coding: utf-8 -*-
"""
Step 4: 构建和测试DataLoader
构建训练和验证 DataLoader（等价 TF 的 filter_by_max_length + padded_batch）
"""

import os
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger

def load_translation_dataset(train_path: str, val_path: str, delimiter: str = "\t"):
    """加载数据集"""
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path
        },
        column_names=["pt", "en"],
        delimiter=delimiter
    )
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
    """训练并加载tokenizer"""
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

    # 保存
    Path(save_dir_pt).mkdir(exist_ok=True)
    Path(save_dir_en).mkdir(exist_ok=True)
    pt_bbpe.save_model(save_dir_pt)
    en_bbpe.save_model(save_dir_en)
    pt_bbpe._tokenizer.save(f"{save_dir_pt}/tokenizer.json")
    en_bbpe._tokenizer.save(f"{save_dir_en}/tokenizer.json")

    # 加载
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

    return pt_tokenizer, en_tokenizer

def build_dataloaders(
    train_dataset,
    val_dataset,
    pt_tokenizer,
    en_tokenizer,
    batch_size: int = 64,
    max_length: int = 48,
    num_workers: int = 0,
    shuffle_train: bool = True,
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

    返回:
        train_loader, val_loader
    """

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
        pairs, kept, skipped = [], 0, 0
        for ex in hf_split:
            pt_ids = encode_with_bos_eos(pt_tok, ex["pt"])
            en_ids = encode_with_bos_eos(en_tok, ex["en"])
            if len(pt_ids) <= max_len and len(en_ids) <= max_len:
                pairs.append((pt_ids, en_ids));
                kept += 1
            else:
                skipped += 1
        logger.info(f"[filter] kept={kept}, skipped={skipped}, max_length={max_len}")
        logger.info(f"{ex = }")
        logger.info(f"{pairs[1] = }")
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
        logger.info(f"{en_ids_list[1] = }")
        pt_input_ids, pt_attention_mask = pad_block(pt_ids_list, pt_tokenizer.pad_token_id)
        en_input_ids, en_attention_mask = pad_block(en_ids_list, en_tokenizer.pad_token_id)
        logger.info(f"{en_input_ids[1] = }")
        logger.info(f"{en_attention_mask[1] = }")
        return {
            "pt_input_ids": pt_input_ids,
            "pt_attention_mask": pt_attention_mask,
            "en_input_ids": en_input_ids,
            "en_attention_mask": en_attention_mask,
        }

    # 5) DataLoader
    train_loader = DataLoader(
        PairsDataset(train_pairs),
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        PairsDataset(val_pairs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=num_workers,
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
    # 1. 拿一个训练 batch 看 shape
    batch = next(iter(train_loader))
    logger.info("=== Train Loader Batch Shapes ===")
    for k, v in batch.items():
        logger.info(f"{k:20s} {tuple(v.shape)}")

    # 2. 验证集样本
    if show_val:
        logger.info("\n=== Validation Loader Example ===")
        for i in val_loader:
            logger.info("pt_input_ids:     ", i["pt_input_ids"][0])
            logger.info("pt_attention_mask:", i["pt_attention_mask"][0])
            break

if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: 构建和测试DataLoader")
    print("=" * 60)
    
    # 数据文件地址
    train_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv"
    val_path = "/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv"
    
    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小
    min_freq = 2  # 最小词频
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]  # 特殊符号
    max_length = 30  # 最大序列长度
    
    # DataLoader参数
    batch_size = 32  # 批处理数
    
    print(f"🔧 DataLoader参数:")
    print(f"   批大小: {batch_size}")
    print(f"   最大序列长度: {max_length}")
    
    try:
        # 1. 加载数据集
        print(f"\n📁 加载数据集...")
        train_dataset, val_dataset = load_translation_dataset(
            train_path=train_path,
            val_path=val_path
        )
        
        # 2. 构建 Tokenizer
        print(f"\n🔨 构建 Tokenizer...")
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
        
        # 3. 构建 DataLoader
        print(f"\n🔨 构建 DataLoader...")
        train_loader, val_loader = build_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            pt_tokenizer=pt_tokenizer,
            en_tokenizer=en_tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=0,
            shuffle_train=True
        )
        
        print(f"\n✅ DataLoader构建完成！")
        print(f"📊 训练集批次数: {len(train_loader)}")
        print(f"📊 验证集批次数: {len(val_loader)}")
        
        # 4. 测试 DataLoader
        print(f"\n🧪 测试 DataLoader...")
        test_dataloaders(train_loader, val_loader)
        
        print(f"\n✅ DataLoader测试完成！")
        
    except Exception as e:
        print(f"❌ DataLoader构建失败: {e}")
        print("💡 请检查数据文件路径是否正确")

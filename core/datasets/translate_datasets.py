#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 17:19
@Author  : nijiahui
@FileName: translate_datasets.py
@Software: PyCharm
 
'''

import torch
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader


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
    print("开始加载数据...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": val_path
        },
        column_names=["pt", "en"],
        delimiter=delimiter
    )

    print("数据集类型:", type(dataset))
    print(dataset)

    # 打印一个样本
    sample = dataset["train"][0]
    print(f"示例数据 -> pt: {sample['pt']} | en: {sample['en']}")

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
        max_length=1024):
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

    print("pt vocab size:", len(pt_tokenizer))
    print("en vocab size:", len(en_tokenizer))

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
    print("=== English Tokenizer Test ===")
    en_ids = en_tokenizer.encode(en_sample, add_special_tokens=False)
    print(f"[EN] Tokenized IDs: {en_ids}")

    en_decoded = en_tokenizer.decode(
        en_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(f"[EN] Decoded string: {en_decoded}")
    assert en_decoded == en_sample, "EN decode != original input!"

    print("[EN] id --> decoded([id])  |  id --> token(str)")
    for tid in en_ids:
        single_decoded = en_tokenizer.decode(
            [tid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        token_str = en_tokenizer.convert_ids_to_tokens(tid)
        print(f"{tid:>6} --> {single_decoded!r}  |  {tid:>6} --> {token_str!r}")

    print("\n" + "-" * 60 + "\n")

    # --- Portuguese ---
    print("=== Portuguese Tokenizer Test ===")
    pt_ids = pt_tokenizer.encode(pt_sample, add_special_tokens=False)
    print(f"[PT] Tokenized IDs: {pt_ids}")

    pt_decoded = pt_tokenizer.decode(
        pt_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(f"[PT] Decoded string: {pt_decoded}")
    assert pt_decoded == pt_sample, "PT decode != original input!"

    print("[PT] id --> decoded([id])  |  id --> token(str)")
    for tid in pt_ids:
        single_decoded = pt_tokenizer.decode(
            [tid],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        token_str = pt_tokenizer.convert_ids_to_tokens(tid)
        print(f"{tid:>6} --> {single_decoded!r}  |  {tid:>6} --> {token_str!r}")


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
        print(f"[filter] kept={kept}, skipped={skipped}, max_length={max_len}")
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
    print("=== Train Loader Batch Shapes ===")
    for k, v in batch.items():
        print(f"{k:20s} {tuple(v.shape)}")

    # 2. 验证集样本
    if show_val:
        print("\n=== Validation Loader Example ===")
        for i in val_loader:
            print("pt_input_ids:     ", i["pt_input_ids"][0])
            print("pt_attention_mask:", i["pt_attention_mask"][0])
            break

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练数据是否正确
"""

import os
import sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 加载tokenizer
en_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
en_tokenizer.pad_token = "<pad>"
en_tokenizer.unk_token = "<unk>"
en_tokenizer.bos_token = "<s>"
en_tokenizer.eos_token = "</s>"
en_tokenizer.mask_token = "<mask>"

# 加载数据集
train_path = "/workspace/tensorflow_datasets/por_en_train.csv"
dataset = load_dataset(
    "csv",
    data_files={"train": train_path},
    column_names=["pt", "en"],
    delimiter="\t"
)

print("=" * 80)
print("检查训练数据样本")
print("=" * 80)

# 打印前5个样本
for i in range(5):
    sample = dataset["train"][i]
    pt_text = sample["pt"]
    en_text = sample["en"]
    
    print(f"\n样本 {i+1}:")
    print(f"  葡语: {pt_text}")
    print(f"  英语: {en_text}")
    
    # 构建prompt（和训练时一样）
    prompt = f"Translate Portuguese to English:\n{pt_text}\nEnglish: "
    
    # 编码
    prompt_ids = en_tokenizer.encode(prompt, add_special_tokens=False)
    en_ids = en_tokenizer.encode(en_text, add_special_tokens=False)
    
    bos_id = en_tokenizer.bos_token_id
    eos_id = en_tokenizer.eos_token_id
    
    input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]
    prompt_len = len([bos_id] + prompt_ids)
    labels = [-100] * prompt_len + en_ids + [eos_id]
    
    print(f"  Prompt长度: {prompt_len}")
    print(f"  英文长度: {len(en_ids) + 1}")  # +1 for EOS
    print(f"  总长度: {len(input_ids)}")
    print(f"  Input IDs (前10个): {input_ids[:10]}")
    print(f"  Labels (前10个): {labels[:10]}")
    print(f"  Labels (中间部分，英文开始): {labels[prompt_len:prompt_len+10]}")
    
    # 解码验证
    decoded = en_tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"  解码结果: {decoded[:100]}...")

print("\n" + "=" * 80)
print("✅ 检查完成")
print("=" * 80)


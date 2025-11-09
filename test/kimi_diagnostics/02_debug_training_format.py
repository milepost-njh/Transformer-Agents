#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试训练数据格式 - 看看训练时模型到底学到了什么
"""

import os
import sys
import torch
from transformers import PreTrainedTokenizerFast
from loguru import logger

# 加载tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.mask_token = "<mask>"

logger.info("=" * 80)
logger.info("调试训练数据格式")
logger.info("=" * 80)

# 测试样本
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

# 方法1: 模拟 train_kimi.py 的编码方式
logger.info("\n【方法1】模拟 train_kimi.py 的编码")
logger.info("-" * 80)

prompt = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
en_ids = tokenizer.encode(test_sample['en'], add_special_tokens=False)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]

prompt_len = len([bos_id] + prompt_ids)
labels = [-100] * prompt_len + en_ids + [eos_id]

logger.info(f"Prompt: {prompt}")
logger.info(f"English: {test_sample['en']}")
logger.info(f"\nPrompt长度: {prompt_len}")
logger.info(f"English长度: {len(en_ids)}")
logger.info(f"总长度: {len(input_ids)}")

logger.info(f"\ninput_ids[:10]: {input_ids[:10]}")
logger.info(f"input_ids[{prompt_len-2}:{prompt_len+5}]: {input_ids[prompt_len-2:prompt_len+5]}")
logger.info(f"input_ids[-5:]: {input_ids[-5:]}")

logger.info(f"\nlabels[:10]: {labels[:10]}")
logger.info(f"labels[{prompt_len-2}:{prompt_len+5}]: {labels[prompt_len-2:prompt_len+5]}")
logger.info(f"labels[-5:]: {labels[-5:]}")

# 解码看看
logger.info("\n解码结果:")
logger.info(f"完整输入: {tokenizer.decode(input_ids, skip_special_tokens=False)}")
logger.info(f"只看prompt部分: {tokenizer.decode(input_ids[:prompt_len], skip_special_tokens=False)}")
logger.info(f"只看英文部分: {tokenizer.decode(input_ids[prompt_len:], skip_special_tokens=False)}")

# 方法2: 测试 tokenizer 的 encode 行为
logger.info("\n" + "=" * 80)
logger.info("【方法2】测试 tokenizer 编码细节")
logger.info("-" * 80)

# 测试不同的编码方式
text1 = test_sample['en']
text2 = " " + test_sample['en']  # 前面加空格
text3 = "\n" + test_sample['en']  # 前面加换行

ids1 = tokenizer.encode(text1, add_special_tokens=False)
ids2 = tokenizer.encode(text2, add_special_tokens=False)
ids3 = tokenizer.encode(text3, add_special_tokens=False)

logger.info(f"'{text1}' -> {ids1}")
logger.info(f"'{text2}' -> {ids2}")
logger.info(f"'{text3}' -> {ids3}")

logger.info(f"\n解码:")
logger.info(f"ids1: {tokenizer.decode(ids1)}")
logger.info(f"ids2: {tokenizer.decode(ids2)}")
logger.info(f"ids3: {tokenizer.decode(ids3)}")

# 方法3: 检查 "English: " 后面应该接什么
logger.info("\n" + "=" * 80)
logger.info("【方法3】检查 'English: ' 后面的token")
logger.info("-" * 80)

# 看看 "English: I" 是怎么编码的
test_texts = [
    "English: ",
    "English: I",
    "English:I",
    "English:\nI",
    "I",
    " I",
]

for text in test_texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    logger.info(f"'{text}' -> {ids} -> {tokenizer.decode(ids)}")

# 方法4: 检查训练时的 loss 计算范围
logger.info("\n" + "=" * 80)
logger.info("【方法4】检查 loss 计算范围")
logger.info("-" * 80)

logger.info(f"\n训练时的 labels:")
logger.info(f"  - 前 {prompt_len} 个位置被mask为 -100（不计算loss）")
logger.info(f"  - 从第 {prompt_len} 个位置开始计算loss")
logger.info(f"  - labels[{prompt_len}] = {labels[prompt_len]} -> token: {tokenizer.decode([input_ids[prompt_len]])}")
logger.info(f"  - labels[{prompt_len+1}] = {labels[prompt_len+1]} -> token: {tokenizer.decode([input_ids[prompt_len+1]])}")
logger.info(f"  - labels[-1] = {labels[-1]} -> token: {tokenizer.decode([input_ids[-1]])}")

logger.info("\n训练时模型要学习的映射:")
logger.info(f"  - 看到 'English: ' 后，要预测: {tokenizer.decode([input_ids[prompt_len]])}")
logger.info(f"  - 看到 'English: I' 后，要预测: {tokenizer.decode([input_ids[prompt_len+1]])}")

logger.info("\n" + "=" * 80)
logger.info("✅ 调试完成")
logger.info("=" * 80)


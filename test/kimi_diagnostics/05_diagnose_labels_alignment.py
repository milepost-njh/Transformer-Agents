#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极诊断：搞清楚labels和logits的对齐关系
"""

import torch
from transformers import PreTrainedTokenizerFast
from loguru import logger

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"

logger.info("=" * 80)
logger.info("终极诊断：Labels 和 Logits 的对齐关系")
logger.info("=" * 80)

# 构建测试样本
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

prompt = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
en_ids = tokenizer.encode(test_sample['en'], add_special_tokens=False)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

# 构建input_ids和labels（和train_kimi.py完全一样）
input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]
prompt_len = len([bos_id] + prompt_ids)
labels = [-100] * prompt_len + en_ids + [eos_id]

logger.info(f"\n测试样本:")
logger.info(f"  葡语: {test_sample['pt']}")
logger.info(f"  英语: {test_sample['en']}")
logger.info(f"  Prompt长度: {prompt_len}")
logger.info(f"  总长度: {len(input_ids)}")

logger.info("\n" + "=" * 80)
logger.info("关键问题：在Causal LM中，logits[i] 应该预测什么？")
logger.info("=" * 80)

logger.info("\n选项A：logits[i] 预测 input_ids[i+1]")
logger.info("  - 这是标准的Causal LM定义")
logger.info("  - 即：看到前i个token，预测第i+1个token")
logger.info("  - 那么 labels[i] 应该 = input_ids[i+1]")
logger.info("  - 需要提前shift labels")

logger.info("\n选项B：logits[i] 预测 input_ids[i]")
logger.info("  - 这不是标准做法，但有些实现可能这样做")
logger.info("  - 那么 labels[i] 应该 = input_ids[i]")
logger.info("  - 不需要shift")

logger.info("\n" + "=" * 80)
logger.info("检查当前的labels构建方式")
logger.info("=" * 80)

logger.info(f"\n当前labels的构建:")
logger.info(f"  labels = [-100] * {prompt_len} + en_ids + [eos]")
logger.info(f"  labels[{prompt_len}] = {labels[prompt_len]} -> '{tokenizer.decode([labels[prompt_len]])}'")
logger.info(f"  input_ids[{prompt_len}] = {input_ids[prompt_len]} -> '{tokenizer.decode([input_ids[prompt_len]])}'")

if labels[prompt_len] == input_ids[prompt_len]:
    logger.warning(f"  ⚠️  labels[i] == input_ids[i]，说明采用的是【选项B】")
    logger.warning(f"  但这不符合标准的Causal LM定义！")
else:
    logger.info(f"  ✅ labels[i] != input_ids[i]，说明labels已经shift了")

logger.info("\n" + "=" * 80)
logger.info("检查期望的对齐关系")
logger.info("=" * 80)

logger.info(f"\n标准Causal LM的期望:")
logger.info(f"  位置 {prompt_len-1}（prompt最后一个）:")
logger.info(f"    input_ids[{prompt_len-1}] = {input_ids[prompt_len-1]} -> '{tokenizer.decode([input_ids[prompt_len-1]])}'")
logger.info(f"    logits[{prompt_len-1}] 应该预测 input_ids[{prompt_len}] = {input_ids[prompt_len]} -> '{tokenizer.decode([input_ids[prompt_len]])}'")
logger.info(f"    所以 labels[{prompt_len-1}] 应该 = {input_ids[prompt_len]} -> '{tokenizer.decode([input_ids[prompt_len]])}'")
logger.info(f"    实际 labels[{prompt_len-1}] = {labels[prompt_len-1]}")

if labels[prompt_len-1] == input_ids[prompt_len]:
    logger.success(f"    ✅ labels已经正确shift！")
elif labels[prompt_len-1] == -100:
    logger.error(f"    ❌ labels[{prompt_len-1}] = -100，这个位置的loss不计算")
    logger.error(f"    💥 模型无法学习如何从prompt预测第一个英文词！")

logger.info("\n" + "=" * 80)
logger.info("彻底分析：如果labels没有提前shift会怎样？")
logger.info("=" * 80)

logger.info(f"\n如果labels没有shift（labels[i] = input_ids[i]）:")
for i in range(prompt_len-1, min(prompt_len+3, len(input_ids))):
    if i < 0 or i >= len(labels):
        continue
    
    input_token = tokenizer.decode([input_ids[i]])
    label = labels[i]
    label_token = tokenizer.decode([label]) if label != -100 else "-100"
    
    # 模型会预测什么
    expected_predict = input_ids[i+1] if i+1 < len(input_ids) else "EOS"
    expected_token = tokenizer.decode([expected_predict]) if isinstance(expected_predict, int) else expected_predict
    
    match = "✅" if label == expected_predict else "❌"
    
    logger.info(f"  位置{i}: input='{input_token}', logits预测'{expected_token}', labels={label_token} {match}")

logger.info("\n" + "=" * 80)
logger.info("结论")
logger.info("=" * 80)

# 检查是否需要shift
needs_shift = labels[prompt_len] == input_ids[prompt_len]

if needs_shift:
    logger.error("\n❌ 发现问题：labels没有提前shift！")
    logger.error("\n当前:")
    logger.error(f"  labels[{prompt_len}] = {labels[prompt_len]} ('{tokenizer.decode([labels[prompt_len]])}')")
    logger.error(f"  input_ids[{prompt_len}] = {input_ids[prompt_len]} ('{tokenizer.decode([input_ids[prompt_len]])}')")
    logger.error(f"  两者相等！")
    
    logger.error("\n应该:")
    logger.error(f"  labels[{prompt_len-1}] 应该 = input_ids[{prompt_len}]")
    logger.error(f"  即：labels 应该向左shift一位")
    
    logger.error("\n修复方案:")
    logger.error("  在构建labels时，应该这样做:")
    logger.error(f"  labels = [-100] * (prompt_len - 1) + [en_ids[0]] + en_ids[1:] + [eos] + [-100]")
    logger.error("  或者等价地:")
    logger.error(f"  先构建 labels = [-100] * prompt_len + en_ids + [eos]")
    logger.error(f"  然后 labels = labels[1:] + [-100]")
else:
    logger.success("\n✅ labels的对齐是正确的（或者模型使用了非标准方式）")
    logger.info("\n可能的原因:")
    logger.info("  1. 模型内部已经自动处理了shift")
    logger.info("  2. 或者这个模型使用了不同的对齐方式")
    logger.info("\n需要检查Kimi模型的loss计算代码")

logger.info("\n" + "=" * 80)
logger.info("✅ 诊断完成")
logger.info("=" * 80)


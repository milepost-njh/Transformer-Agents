#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 labels 的 shift 问题
检查训练时 labels 和 input_ids 的对齐关系
"""

from transformers import PreTrainedTokenizerFast
from loguru import logger

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.mask_token = "<mask>"

logger.info("=" * 80)
logger.info("调试 Labels Shift 问题")
logger.info("=" * 80)

# 测试样本
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

# 构建训练数据
prompt = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
en_ids = tokenizer.encode(test_sample['en'], add_special_tokens=False)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]

prompt_len = len([bos_id] + prompt_ids)
labels = [-100] * prompt_len + en_ids + [eos_id]

logger.info(f"\n葡语: {test_sample['pt']}")
logger.info(f"英语: {test_sample['en']}")
logger.info(f"\nPrompt长度: {prompt_len}")
logger.info(f"英文IDs: {en_ids}")
logger.info(f"总长度: {len(input_ids)}")

logger.info("\n" + "=" * 80)
logger.info("在 Causal LM 中，模型的训练目标是：")
logger.info("  给定 input_ids[:i+1]，预测 input_ids[i+1]")
logger.info("  即：logits[i] 用来预测 input_ids[i+1]")
logger.info("=" * 80)

logger.info("\n" + "-" * 80)
logger.info("HuggingFace 的实现中，labels 会被自动shift")
logger.info("即：loss 计算时会将 logits[:, :-1, :] 与 labels[:, 1:] 对齐")
logger.info("或者等价地：shift_logits = logits[:, :-1, :], shift_labels = labels[:, 1:]")
logger.info("-" * 80)

logger.info("\n" + "=" * 80)
logger.info("检查当前labels的对齐情况")
logger.info("=" * 80)

# 关键位置分析
key_positions = [
    (prompt_len - 2, "prompt倒数第二个token"),
    (prompt_len - 1, "prompt最后一个token"),
    (prompt_len, "第一个英文token"),
    (prompt_len + 1, "第二个英文token"),
]

for pos, desc in key_positions:
    if pos >= len(input_ids):
        continue
    
    input_token = input_ids[pos]
    input_token_str = tokenizer.decode([input_token])
    label = labels[pos] if pos < len(labels) else "N/A"
    label_str = tokenizer.decode([label]) if label != -100 and label != "N/A" else str(label)
    
    # HuggingFace shift 后，位置pos的logits会与labels[pos+1]计算loss
    label_shifted = labels[pos + 1] if pos + 1 < len(labels) else "N/A"
    label_shifted_str = tokenizer.decode([label_shifted]) if label_shifted != -100 and label_shifted != "N/A" else str(label_shifted)
    
    logger.info(f"\n位置 {pos} ({desc}):")
    logger.info(f"  input_ids[{pos}] = {input_token} -> '{input_token_str}'")
    logger.info(f"  labels[{pos}] = {label} -> '{label_str}'")
    logger.info(f"  ⚠️  HF shift后，logits[{pos}]会与labels[{pos+1}]={label_shifted}计算loss")
    logger.info(f"     即：看到'{input_token_str}'后，模型被训练预测 '{label_shifted_str}'")

logger.info("\n" + "=" * 80)
logger.info("🔍 关键问题分析")
logger.info("=" * 80)

logger.info(f"\n当前设置下：")
logger.info(f"  prompt_len = {prompt_len}")
logger.info(f"  labels[:prompt_len] = [-100] * {prompt_len}")
logger.info(f"  labels[prompt_len:] = en_ids + [eos]")

logger.info(f"\nHF shift后的实际训练目标：")
logger.info(f"  位置{prompt_len-1}（prompt最后一个token）:")
logger.info(f"    看到: input_ids[:{prompt_len}] = prompt完整内容")
logger.info(f"    预测: labels[{prompt_len}] = {labels[prompt_len]} -> '{tokenizer.decode([labels[prompt_len]])}'")
logger.info(f"    ✅ 这个位置**参与训练**，模型学习预测第一个英文词")

logger.info(f"\n  位置{prompt_len}（第一个英文token 'I'）:")
logger.info(f"    看到: input_ids[:{prompt_len+1}] = prompt + 'I'")
logger.info(f"    预测: labels[{prompt_len+1}] = {labels[prompt_len+1]} -> '{tokenizer.decode([labels[prompt_len+1]])}'")
logger.info(f"    ✅ 这个位置**参与训练**，模型学习预测' gave'")

logger.info("\n" + "=" * 80)
logger.info("✅ Labels对齐是正确的！")
logger.info("=" * 80)

logger.info("\n那为什么Teacher Forcing测试失败了？")
logger.info("让我检查测试代码的位置索引...")

logger.info(f"\nTeacher Forcing测试中：")
logger.info(f"  检查位置33（prompt_len-1 = {prompt_len-1}）:")
logger.info(f"    input_ids[33] = {input_ids[33]} -> '{tokenizer.decode([input_ids[33]])}'")
logger.info(f"    logits[33]预测的是input_ids[34] = {input_ids[34]} -> '{tokenizer.decode([input_ids[34]])}'")
logger.info(f"    实际应该预测: {input_ids[34]} -> '{tokenizer.decode([input_ids[34]])}'")
logger.info(f"    模型预测: 225 -> '{tokenizer.decode([225])}'")
logger.info(f"    ❌ 预测错误！")

logger.info(f"\n但根据labels，位置33的训练目标是什么？")
logger.info(f"  labels[33] = {labels[33]} (被mask)")
logger.info(f"  HF shift后，logits[33]与labels[34]对齐")
logger.info(f"  labels[34] = {labels[34]} -> '{tokenizer.decode([labels[34]])}'")
logger.info(f"  ✅ 训练目标是预测'{tokenizer.decode([labels[34]])}'")

logger.info("\n" + "=" * 80)
logger.info("问题诊断完成")
logger.info("=" * 80)


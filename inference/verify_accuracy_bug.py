#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 token_accuracy 函数的bug
"""

import torch
from transformers import PreTrainedTokenizerFast
from loguru import logger

# 模拟 train_kimi.py 中的函数
def token_accuracy_original(real, pred, pad_id):
    """原始的（有bug的）版本"""
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    mask = (real != pad_id) & (real != -100)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)


def token_accuracy_fixed(real, pred, pad_id):
    """修复后的版本 - 考虑shift"""
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    
    # 在Causal LM中，logits[i]预测的是token[i+1]
    # 所以应该比较 pred_ids[:, :-1] 和 real[:, 1:]
    pred_ids_shifted = pred_ids[:, :-1]  # 去掉最后一个预测
    real_shifted = real[:, 1:]  # 去掉第一个标签
    
    mask = (real_shifted != pad_id) & (real_shifted != -100)
    correct = ((pred_ids_shifted == real_shifted) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)


logger.info("=" * 80)
logger.info("验证 token_accuracy 的 Bug")
logger.info("=" * 80)

# 加载tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"

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
input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]

prompt_len = len([bos_id] + prompt_ids)
labels = [-100] * prompt_len + en_ids + [eos_id]

logger.info(f"\n测试样本:")
logger.info(f"  Prompt长度: {prompt_len}")
logger.info(f"  英文长度: {len(en_ids)}")
logger.info(f"  总长度: {len(input_ids)}")
logger.info(f"  有效labels数量: {sum(1 for x in labels if x != -100)}")

# 转换为tensor
input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
labels_tensor = torch.tensor([labels], dtype=torch.long)

# 创建假的logits：
# 场景1: 完美预测（每个位置都预测对）
logger.info("\n" + "=" * 80)
logger.info("场景1：模型完美预测（每个位置都对）")
logger.info("=" * 80)

vocab_size = len(tokenizer)
batch_size = 1
seq_len = len(input_ids)

# 创建logits，让每个位置的最大概率对应下一个token
perfect_logits = torch.zeros(batch_size, seq_len, vocab_size)
for i in range(seq_len - 1):
    next_token = input_ids[i + 1]
    perfect_logits[0, i, next_token] = 10.0  # 给正确答案很高的分数

# 最后一个位置随便给个值
perfect_logits[0, -1, eos_id] = 10.0

logger.info("\n使用原始函数（有bug）:")
acc_original = token_accuracy_original(labels_tensor, perfect_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_original*100:.2f}%")

logger.info("\n使用修复后的函数:")
acc_fixed = token_accuracy_fixed(labels_tensor, perfect_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_fixed*100:.2f}%")

if acc_original == 1.0 and acc_fixed == 1.0:
    logger.success("✅ 场景1通过：两个函数都识别出完美预测")
else:
    logger.error(f"❌ 场景1失败：原始={acc_original:.4f}, 修复={acc_fixed:.4f}")

# 场景2: 错误预测（每个位置都错）
logger.info("\n" + "=" * 80)
logger.info("场景2：模型完全预测错（每个位置都错）")
logger.info("=" * 80)

wrong_logits = torch.zeros(batch_size, seq_len, vocab_size)
for i in range(seq_len):
    # 故意给错误的token高分（选一个不是正确答案的token）
    wrong_token = (input_ids[i] + 100) % vocab_size
    wrong_logits[0, i, wrong_token] = 10.0

logger.info("\n使用原始函数（有bug）:")
acc_original_wrong = token_accuracy_original(labels_tensor, wrong_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_original_wrong*100:.2f}%")

logger.info("\n使用修复后的函数:")
acc_fixed_wrong = token_accuracy_fixed(labels_tensor, wrong_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_fixed_wrong*100:.2f}%")

if acc_original_wrong == 0.0 and acc_fixed_wrong == 0.0:
    logger.success("✅ 场景2通过：两个函数都识别出完全错误")
else:
    logger.error(f"❌ 场景2失败：原始={acc_original_wrong:.4f}, 修复={acc_fixed_wrong:.4f}")

# 场景3: 关键测试 - 对齐问题
logger.info("\n" + "=" * 80)
logger.info("场景3：测试对齐问题（Bug的关键）")
logger.info("=" * 80)
logger.info("构造一个特殊的logits：")
logger.info("  每个位置预测的都是**当前位置**的token（而不是下一个）")

misaligned_logits = torch.zeros(batch_size, seq_len, vocab_size)
for i in range(seq_len):
    current_token = input_ids[i]
    misaligned_logits[0, i, current_token] = 10.0  # 预测当前token

logger.info("\n使用原始函数（有bug）:")
acc_original_misaligned = token_accuracy_original(labels_tensor, misaligned_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_original_misaligned*100:.2f}%")
logger.info(f"  📊 分析: 如果准确率很高，说明函数有bug（没有考虑shift）")

logger.info("\n使用修复后的函数:")
acc_fixed_misaligned = token_accuracy_fixed(labels_tensor, misaligned_logits, pad_id=tokenizer.pad_token_id)
logger.info(f"  准确率: {acc_fixed_misaligned*100:.2f}%")
logger.info(f"  📊 分析: 应该是0%（因为位置完全错位了）")

# 场景4: 检查 train_kimi.py 中的手动shift
logger.info("\n" + "=" * 80)
logger.info("场景4：检查 train_kimi.py 的手动shift")
logger.info("=" * 80)

logger.info(f"\n原始labels:")
logger.info(f"  长度: {len(labels)}")
logger.info(f"  有效token数: {sum(1 for x in labels if x != -100)}")
logger.info(f"  最后5个: {labels[-5:]}")

# 模拟 train_kimi.py 第2390行的操作
labels_shifted_manual = labels[1:] + [-100]
logger.info(f"\n手动shift后的labels:")
logger.info(f"  长度: {len(labels_shifted_manual)}")
logger.info(f"  有效token数: {sum(1 for x in labels_shifted_manual if x != -100)}")
logger.info(f"  最后5个: {labels_shifted_manual[-5:]}")

logger.info(f"\n🔍 关键发现:")
logger.info(f"  原始labels最后一个token: {labels[-1]} (EOS={eos_id})")
logger.info(f"  shift后最后一个: {labels_shifted_manual[-1]} (应该是-100)")
logger.info(f"  ⚠️  EOS token被shift到了倒数第二个位置")
logger.info(f"  ❌ 最后一个位置变成-100，loss不计算")
logger.info(f"  💥 模型永远学不会在句子结束时生成EOS！")

logger.info("\n" + "=" * 80)
logger.info("总结")
logger.info("=" * 80)

logger.info("\n发现的Bug:")
logger.info("  1. token_accuracy 函数没有考虑 shift")
logger.info("  2. train_kimi.py 中手动shift labels会导致EOS token的训练目标丢失")
logger.info("  3. 这两个bug共同导致：")
logger.info("     - 训练日志显示虚高的准确率（因为对齐错误）")
logger.info("     - 模型学不会正确生成EOS（因为EOS位置被mask了）")
logger.info("     - 推理时模型无法正确结束生成")

logger.info("\n建议的修复:")
logger.info("  1. 移除 train_kimi.py 第2390行的手动shift")
logger.info("  2. 让HuggingFace的模型自动处理shift（它们已经内置了）")
logger.info("  3. 修复 token_accuracy 函数，正确处理shift")
logger.info("     或者在调用前手动shift: acc = token_accuracy(labels[:, 1:], logits[:, :-1, :], ...)")


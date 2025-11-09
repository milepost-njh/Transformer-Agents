#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证labels修复是否正确
"""

from transformers import PreTrainedTokenizerFast
from loguru import logger

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"

logger.info("=" * 80)
logger.info("验证Labels修复")
logger.info("=" * 80)

# 测试样本
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

# 构建数据（模拟修复后的代码）
prompt = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
en_ids = tokenizer.encode(test_sample['en'], add_special_tokens=False)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
input_ids = [bos_id] + prompt_ids + en_ids + [eos_id]

prompt_len = len([bos_id] + prompt_ids)

# 🔧 修复后的labels构建
labels_unshifted = [-100] * prompt_len + en_ids + [eos_id]
labels = labels_unshifted[1:] + [-100]

logger.info(f"\n测试样本:")
logger.info(f"  葡语: {test_sample['pt']}")
logger.info(f"  英语: {test_sample['en']}")
logger.info(f"  Prompt长度: {prompt_len}")
logger.info(f"  总长度: {len(input_ids)}")

logger.info("\n" + "=" * 80)
logger.info("验证修复后的对齐关系")
logger.info("=" * 80)

# 检查关键位置
checks = [
    (prompt_len - 2, "prompt倒数第二个"),
    (prompt_len - 1, "prompt最后一个"),
    (prompt_len, "第一个英文词"),
    (prompt_len + 1, "第二个英文词"),
    (len(input_ids) - 2, "倒数第二个（最后一个词）"),
    (len(input_ids) - 1, "最后一个（应该预测EOS但没有label）"),
]

all_correct = True

for pos, desc in checks:
    if pos < 0 or pos >= len(input_ids):
        continue
    
    input_token_id = input_ids[pos]
    input_token = tokenizer.decode([input_token_id])
    
    label = labels[pos] if pos < len(labels) else "N/A"
    label_token = tokenizer.decode([label]) if label != -100 and label != "N/A" else str(label)
    
    # logits[pos]应该预测input_ids[pos+1]
    expected_predict = input_ids[pos + 1] if pos + 1 < len(input_ids) else "N/A"
    expected_token = tokenizer.decode([expected_predict]) if isinstance(expected_predict, int) else expected_predict
    
    # 检查labels[pos]是否等于input_ids[pos+1]
    is_correct = label == expected_predict
    status = "✅" if is_correct else ("⚠️" if label == -100 else "❌")
    
    logger.info(f"\n{status} 位置 {pos} ({desc}):")
    logger.info(f"  input_ids[{pos}] = {input_token_id} -> '{input_token}'")
    logger.info(f"  logits[{pos}] 应该预测 input_ids[{pos+1}] = {expected_predict} -> '{expected_token}'")
    logger.info(f"  labels[{pos}] = {label} -> '{label_token}'")
    
    if label == -100:
        logger.info(f"  📝 这个位置不计算loss（mask掉了）")
    elif is_correct:
        logger.info(f"  ✅ labels[{pos}] == input_ids[{pos+1}]，对齐正确！")
    else:
        logger.error(f"  ❌ labels[{pos}] != input_ids[{pos+1}]，对齐错误！")
        all_correct = False

logger.info("\n" + "=" * 80)
logger.info("关键位置验证")
logger.info("=" * 80)

# 验证prompt最后一个位置
logger.info(f"\n位置 {prompt_len-1} (prompt最后一个，空格):")
logger.info(f"  input_ids[{prompt_len-1}] = {input_ids[prompt_len-1]} -> '{tokenizer.decode([input_ids[prompt_len-1]])}'")
logger.info(f"  labels[{prompt_len-1}] = {labels[prompt_len-1]}")

if labels[prompt_len-1] != -100:
    logger.success(f"  ✅ labels[{prompt_len-1}] != -100，这个位置参与训练！")
    logger.info(f"  labels[{prompt_len-1}] = {labels[prompt_len-1]} -> '{tokenizer.decode([labels[prompt_len-1]])}'")
    logger.info(f"  input_ids[{prompt_len}] = {input_ids[prompt_len]} -> '{tokenizer.decode([input_ids[prompt_len]])}'")
    
    if labels[prompt_len-1] == input_ids[prompt_len]:
        logger.success(f"  ✅✅ 完美对齐！模型可以学习从prompt预测第一个英文词！")
    else:
        logger.error(f"  ❌ 对齐错误")
        all_correct = False
else:
    logger.error(f"  ❌ labels[{prompt_len-1}] = -100，这个位置不参与训练！")
    logger.error(f"  💥 模型无法学习如何从prompt预测第一个英文词！")
    all_correct = False

logger.info("\n" + "=" * 80)
if all_correct:
    logger.success("✅✅✅ 修复验证通过！所有关键位置对齐正确！")
    logger.info("\n现在可以重新训练了：")
    logger.info("  1. 停止当前训练（Ctrl+C）")
    logger.info("  2. 清理旧checkpoints: rm -rf checkpoints_kimi_translation/*")
    logger.info("  3. git pull  # 获取最新修复")
    logger.info("  4. bash run_kimi.sh  # 重新训练")
    logger.info("\n期望结果：")
    logger.info("  - 训练初期准确率应该在10-30%左右（真实的准确率）")
    logger.info("  - 逐步上升到80-95%")
    logger.info("  - 推理时能输出正确的翻译")
else:
    logger.error("❌ 验证失败！仍然存在对齐问题！")
    logger.error("需要进一步检查代码")

logger.info("\n" + "=" * 80)
logger.info("✅ 验证完成")
logger.info("=" * 80)


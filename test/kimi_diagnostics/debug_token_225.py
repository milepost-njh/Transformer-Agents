#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 token 225 到底是什么
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
logger.info("调查 token 225 的真实身份")
logger.info("=" * 80)

# 直接解码 token 225
token_225 = tokenizer.decode([225])
logger.info(f"\nToken 225 解码: '{token_225}'")
logger.info(f"Token 225 repr: {repr(token_225)}")
logger.info(f"Token 225 bytes: {token_225.encode('utf-8')}")
logger.info(f"Token 225 长度: {len(token_225)}")

# 对比其他token
logger.info("\n" + "-" * 80)
logger.info("对比其他常见token:")
test_tokens = {
    "空格": " ",
    "换行": "\n",
    "制表符": "\t",
    "回车": "\r",
}

for name, char in test_tokens.items():
    ids = tokenizer.encode(char, add_special_tokens=False)
    logger.info(f"{name} '{repr(char)}' -> {ids}")

# 重新测试 prompt 构建
logger.info("\n" + "=" * 80)
logger.info("测试 prompt 末尾的编码")
logger.info("-" * 80)

test_cases = [
    "English:",
    "English: ",
    "English:  ",
    "English:\n",
    "English: \n",
]

for text in test_cases:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    logger.info(f"'{repr(text)}' -> {ids}")
    logger.info(f"  解码: '{repr(decoded)}'")
    logger.info(f"  最后一个token: {ids[-1]}")
    logger.info("")

# 测试拼接
logger.info("=" * 80)
logger.info("测试分开编码 vs 一起编码")
logger.info("-" * 80)

prompt = "English: "
answer = "I gave the boy a book."

# 方法1：分开编码
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
answer_ids = tokenizer.encode(answer, add_special_tokens=False)
combined_ids_separate = prompt_ids + answer_ids

# 方法2：一起编码
full_text = prompt + answer
combined_ids_together = tokenizer.encode(full_text, add_special_tokens=False)

logger.info(f"Prompt: '{repr(prompt)}'")
logger.info(f"Answer: '{repr(answer)}'")
logger.info(f"\n分开编码:")
logger.info(f"  prompt_ids: {prompt_ids}")
logger.info(f"  answer_ids: {answer_ids}")
logger.info(f"  拼接结果: {combined_ids_separate}")
logger.info(f"\n一起编码:")
logger.info(f"  结果: {combined_ids_together}")
logger.info(f"\n是否一致: {combined_ids_separate == combined_ids_together}")

if combined_ids_separate != combined_ids_together:
    logger.error("❌ 警告：分开编码和一起编码的结果不同！")
    logger.error(f"   差异位置: {len(prompt_ids)}")
    logger.error(f"   分开: token {combined_ids_separate[len(prompt_ids)]}")
    logger.error(f"   一起: token {combined_ids_together[len(prompt_ids)]}")
    
    # 解码对比
    decoded_separate = tokenizer.decode(combined_ids_separate)
    decoded_together = tokenizer.decode(combined_ids_together)
    logger.error(f"\n解码对比:")
    logger.error(f"   分开: '{repr(decoded_separate)}'")
    logger.error(f"   一起: '{repr(decoded_together)}'")

logger.info("\n" + "=" * 80)
logger.info("✅ 调试完成")
logger.info("=" * 80)


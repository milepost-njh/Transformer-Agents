#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的推理测试 - 不使用 KV-cache，测试模型是否真的学会了翻译
"""

import os
import sys
import torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from loguru import logger

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.models.kimi_linear.modeling_kimi import KimiLinearForCausalLM
from core.models.kimi_linear.configuration_kimi import KimiLinearConfig

# 加载tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.mask_token = "<mask>"

# 创建模型配置
d_model = 512
num_heads = 8
dff = 2048
head_dim = d_model // num_heads

config = KimiLinearConfig(
    vocab_size=len(tokenizer),
    hidden_size=d_model,
    head_dim=head_dim,
    intermediate_size=dff,
    num_hidden_layers=8,
    num_attention_heads=num_heads,
    num_key_value_heads=num_heads,
    hidden_act="silu",
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=False,  # 不使用cache
    rope_theta=10000.0,
    tie_word_embeddings=False,
    num_experts=8,
    num_experts_per_token=2,
    moe_intermediate_size=dff,
    kv_lora_rank=4 * head_dim,
    qk_nope_head_dim=head_dim // 2,
    qk_rope_head_dim=head_dim // 2,
    v_head_dim=head_dim,
    mla_use_nope=True,
    _attn_implementation="eager",
)

# 加载模型
model = KimiLinearForCausalLM(config)
checkpoint = torch.load("/workspace/checkpoints_kimi_translation_bak/best_e1_s1383.pt", map_location="cpu")
model_state = checkpoint['model']
if any(key.startswith('module.') for key in model_state.keys()):
    model_state = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
model.load_state_dict(model_state, strict=False)
model.eval()

logger.info("=" * 80)
logger.info("测试：给定训练样本，看模型能否正确预测")
logger.info("=" * 80)

# 测试1：使用训练数据中的样本（应该能预测对）
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

# 构建完整输入（和训练时一样）
prompt = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
full_text = prompt + test_sample['en']

# 编码
input_ids = tokenizer.encode(full_text, add_special_tokens=True, return_tensors="pt")
logger.info(f"\n完整输入: {full_text}")
logger.info(f"输入长度: {input_ids.shape[1]} tokens")

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

# 取最后一个位置的预测
last_logits = logits[0, -1, :]
predicted_token_id = torch.argmax(last_logits).item()
predicted_token = tokenizer.decode([predicted_token_id])

logger.info(f"模型预测的下一个token: {predicted_token} (ID: {predicted_token_id})")
logger.info(f"期望的下一个token: </s> (ID: {tokenizer.eos_token_id})")

# 测试2：只给prompt，看模型预测什么
logger.info("\n" + "=" * 80)
logger.info("测试：只给 prompt，看模型预测第一个单词")
logger.info("=" * 80)

prompt_only = f"Translate Portuguese to English:\n{test_sample['pt']}\nEnglish: "
prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=True, return_tensors="pt")

logger.info(f"Prompt: {prompt_only}")
logger.info(f"Prompt长度: {prompt_ids.shape[1]} tokens")

with torch.no_grad():
    outputs = model(input_ids=prompt_ids)
    logits = outputs.logits

# 取最后一个位置的预测（这应该是翻译的第一个词）
last_logits = logits[0, -1, :]
top5_tokens = torch.topk(last_logits, 5)

logger.info(f"\nTop 5 预测:")
for i, (score, token_id) in enumerate(zip(top5_tokens.values, top5_tokens.indices)):
    token = tokenizer.decode([token_id.item()])
    logger.info(f"  {i+1}. {token} (ID: {token_id.item()}, score: {score.item():.4f})")

logger.info(f"\n期望的第一个词应该是: I")
first_word_id = tokenizer.encode(" I", add_special_tokens=False)[0]
logger.info(f"'I' 的 token ID: {first_word_id}")

logger.info("\n" + "=" * 80)
logger.info("✅ 测试完成")
logger.info("=" * 80)


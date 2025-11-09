#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher Forcing 测试 - 测试模型在"作弊"条件下的表现
如果模型在这个测试中都失败，说明根本没学到东西
"""

import os
import sys
import torch
from transformers import PreTrainedTokenizerFast
from loguru import logger

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

# 配置
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
    use_cache=False,
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KimiLinearForCausalLM(config)
checkpoint = torch.load("/workspace/checkpoints_kimi_translation_bak/best_e1_s1383.pt", map_location="cpu")
model_state = checkpoint['model']
if any(key.startswith('module.') for key in model_state.keys()):
    model_state = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
model.load_state_dict(model_state, strict=False)

model = model.to(device)
if torch.cuda.is_bf16_supported():
    model = model.to(torch.bfloat16)
else:
    model = model.to(torch.float16)
model.eval()

logger.info("=" * 80)
logger.info("Teacher Forcing 测试")
logger.info("给模型完整的输入序列，看它每一步的预测")
logger.info("=" * 80)

# 测试样本
test_sample = {
    "pt": "Eu dei um livro ao menino.",
    "en": "I gave the boy a book."
}

# 按照训练时的方式编码
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
logger.info(f"Prompt长度: {prompt_len}")
logger.info(f"总长度: {len(input_ids)}")

# 转换为tensor
input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids_tensor)
    logits = outputs.logits

logger.info("\n" + "=" * 80)
logger.info("检查关键位置的预测")
logger.info("=" * 80)

# 检查几个关键位置
check_positions = [
    (prompt_len - 1, "看到完整prompt后，预测第一个英文词"),
    (prompt_len, "看到'English: I'后，预测' gave'"),
    (prompt_len + 1, "看到'English: I gave'后，预测' the'"),
    (len(input_ids) - 2, "看到完整句子后，预测EOS"),
]

correct_count = 0
total_count = len(check_positions)

for pos, description in check_positions:
    if pos >= len(input_ids):
        continue
    
    # 模型在位置pos的预测（预测pos+1的token）
    predicted_logits = logits[0, pos, :]
    predicted_token_id = torch.argmax(predicted_logits).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    
    # 实际的下一个token
    actual_token_id = input_ids[pos + 1] if pos + 1 < len(input_ids) else eos_id
    actual_token = tokenizer.decode([actual_token_id])
    
    # 预测的概率分布（top 5）
    top5 = torch.topk(predicted_logits, 5)
    
    is_correct = predicted_token_id == actual_token_id
    if is_correct:
        correct_count += 1
    
    status = "✅" if is_correct else "❌"
    logger.info(f"\n{status} 位置 {pos}: {description}")
    logger.info(f"   预测: {predicted_token} (ID: {predicted_token_id})")
    logger.info(f"   实际: {actual_token} (ID: {actual_token_id})")
    logger.info(f"   Top 5 预测:")
    for i, (score, tid) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([tid.item()])
        marker = "👈" if tid.item() == actual_token_id else ""
        logger.info(f"      {i+1}. {token} (ID: {tid.item()}, score: {score.item():.4f}) {marker}")

logger.info("\n" + "=" * 80)
logger.info(f"准确率: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")
logger.info("=" * 80)

if correct_count == 0:
    logger.error("❌ 模型完全没有学到正确的预测！")
elif correct_count < total_count:
    logger.warning(f"⚠️ 模型只对了 {correct_count}/{total_count}，学习不够好")
else:
    logger.success("✅ 模型在 Teacher Forcing 条件下表现完美！")
    logger.info("   → 问题可能出在自回归生成时的累积误差")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi 翻译模型推理脚本
完全模拟训练时的编码方式，确保推理结果正确

使用方法:
    CUDA_VISIBLE_DEVICES=1 python inference/inference_kimi.py
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


def load_tokenizer(tokenizer_path: str):
    """加载 tokenizer"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.mask_token = "<mask>"
    tokenizer.model_max_length = 128
    tokenizer.padding_side = "right"
    
    logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
    logger.info(f"   - Vocabulary size: {len(tokenizer)}")
    logger.info(f"   - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"   - bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"   - eos_token_id: {tokenizer.eos_token_id}")
    
    return tokenizer


def create_model_config(vocab_size: int):
    """创建模型配置"""
    d_model = 512
    num_heads = 8
    dff = 2048
    head_dim = d_model // num_heads
    
    config = KimiLinearConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        head_dim=head_dim,
        intermediate_size=dff,
        num_hidden_layers=8,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
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
        _attn_implementation="flash_attention_2",
    )
    
    logger.info("✅ Model configuration created")
    logger.info(f"   - Hidden layers: {config.num_hidden_layers}")
    logger.info(f"   - Hidden size: {config.hidden_size}")
    logger.info(f"   - Attention heads: {config.num_attention_heads}")
    logger.info(f"   - FFN size: {config.intermediate_size}")
    logger.info(f"   - MoE experts: {config.num_experts}")
    logger.info(f"   - Use MLA: True")
    
    return config


def load_model(config, checkpoint_path: str, device: torch.device):
    """加载模型和检查点"""
    model = KimiLinearForCausalLM(config)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 处理 DDP 包装的模型
    model_state = checkpoint['model']
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
    
    model.load_state_dict(model_state, strict=False)
    
    logger.info("✅ Checkpoint loaded successfully")
    if 'epoch' in checkpoint:
        logger.info(f"   - Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        logger.info(f"   - Step: {checkpoint['global_step']}")
    
    # 移到设备并转换精度
    model = model.to(device)
    
    # 转换模型为 bfloat16（Flash Attention 要求）
    if torch.cuda.is_bf16_supported():
        logger.info("Converting model to bfloat16 for Flash Attention...")
        model = model.to(torch.bfloat16)
    else:
        logger.info("Converting model to float16 for Flash Attention...")
        model = model.to(torch.float16)
    
    model.eval()
    return model


@torch.no_grad()
def generate_training_style(
    model,
    tokenizer,
    input_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
):
    """
    使用训练时的编码方式进行推理
    关键：分开编码 prompt，然后逐token生成（模拟训练时的 token 序列）
    """
    
    # 构建 prompt（和训练时完全一样）
    prompt = f"Translate Portuguese to English:\n{input_text}\nEnglish: "
    
    # 编码 prompt（和训练时一样：add_special_tokens=False，然后手动添加bos）
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    # 构建初始 input_ids: [bos] + prompt_ids
    input_ids = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)
    
    logger.info(f"Input prompt: {prompt}")
    logger.info(f"Input length: {input_ids.shape[1]} tokens (including BOS)")
    logger.info(f"开始生成（训练风格）...")
    
    generated_ids = []
    past_key_values = None
    
    for step in range(max_new_tokens):
        # 第一步：输入完整 prompt；后续步骤：只输入最后一个 token
        if step == 0:
            current_input = input_ids
        else:
            current_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
        
        # 前向传播
        outputs = model(
            input_ids=current_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[0, -1, :]
        
        # 生成下一个 token
        if temperature == 0.0:
            # 贪婪解码
            token_id = torch.argmax(next_token_logits, dim=-1).item()
        else:
            # 温度采样
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
        
        # 停止条件
        if token_id == eos_id:
            logger.info(f"遇到 EOS token，停止生成（步骤 {step+1}）")
            break
        
        generated_ids.append(token_id)
        
        # 每隔10步打印一次当前结果
        if (step + 1) % 10 == 0:
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            logger.info(f"Step {step+1}: {partial_text}")
    
    logger.info(f"✅ 生成完成，共生成 {len(generated_ids)} tokens")
    
    # 解码生成的文本
    # 注意：这里我们只解码生成的部分（不包括prompt）
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    # 配置
    tokenizer_dir = "tok_en"
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    checkpoint_path = "/workspace/checkpoints_kimi_translation/best_e11_s15213.pt"
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("Kimi Translation Model Inference (Fixed)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Step 1: 加载 tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Tokenizer")
    logger.info("=" * 80)
    tokenizer = load_tokenizer(tokenizer_file)
    
    # Step 2: 创建模型配置
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating Model Configuration")
    logger.info("=" * 80)
    config = create_model_config(vocab_size=len(tokenizer))
    
    # Step 3: 加载模型
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Loading Model")
    logger.info("=" * 80)
    model = load_model(config, checkpoint_path, device)
    
    # Step 4: 运行推理
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Running Inference")
    logger.info("=" * 80)
    
    # 测试样本
    test_samples = [
        "Eu gosto de aprender idiomas.",
        "O Tom está procurando uma segunda opinião sobre o tratamento médico.",
        "Bom dia! Como você está?",
    ]
    
    for i, pt_text in enumerate(test_samples, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Example {i}")
        logger.info("=" * 80)
        logger.info(f"📥 Portuguese: {pt_text}")
        
        en_text = generate_training_style(
            model=model,
            tokenizer=tokenizer,
            input_text=pt_text,
            device=device,
            max_new_tokens=64,
            temperature=0.0,  # 贪婪解码
        )
        
        logger.success(f"📤 English: {en_text}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


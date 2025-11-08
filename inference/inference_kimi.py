#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi 模型简单推理脚本
用于加载训练好的 Kimi 翻译模型进行推理

使用方法:
    python inference/inference_kimi.py
"""

import os
import sys
import torch
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入必要的模块
from transformers import PreTrainedTokenizerFast
from core.models.kimi_linear.modeling_kimi import KimiLinearForCausalLM
from core.models.kimi_linear.configuration_kimi import KimiLinearConfig


def load_tokenizer(tokenizer_path: str = "tok_en/tokenizer.json"):
    """加载 tokenizer"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.mask_token = "<mask>"
    
    logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
    logger.info(f"   - Vocabulary size: {len(tokenizer)}")
    logger.info(f"   - pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"   - bos_token_id: {tokenizer.bos_token_id}")
    logger.info(f"   - eos_token_id: {tokenizer.eos_token_id}")
    
    return tokenizer


def create_model_config(vocab_size: int, use_mla: bool = True):
    """创建 Kimi 模型配置"""
    # 模型配置参数（与训练时保持一致）
    num_layers = 8
    d_model = 512
    num_heads = 8
    dff = 2048
    head_dim = d_model // num_heads
    kv_lora_rank = 4 * head_dim if use_mla else None
    
    # MoE 配置
    num_experts = 8
    num_experts_per_tok = 2
    
    config = KimiLinearConfig(
        vocab_size=vocab_size,
        hidden_size=d_model,
        head_dim=head_dim,
        intermediate_size=dff,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,  # 推理时使用 cache
        rope_theta=10000.0,
        tie_word_embeddings=False,
        # MoE 配置
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_tok,
        moe_intermediate_size=dff,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=0.8,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        # MLA 配置
        q_lora_rank=None,  # Kimi 模型强制要求 q_lora_rank 为 None
        kv_lora_rank=kv_lora_rank if use_mla else None,
        qk_nope_head_dim=head_dim // 2 if use_mla else None,
        qk_rope_head_dim=head_dim // 2 if use_mla else None,
        v_head_dim=head_dim if use_mla else None,
        mla_use_nope=True,
        # Attention 实现配置
        _attn_implementation="eager",  # 推理时使用 eager 模式
    )
    
    logger.info("✅ Model configuration created")
    logger.info(f"   - Hidden layers: {num_layers}")
    logger.info(f"   - Hidden size: {d_model}")
    logger.info(f"   - Attention heads: {num_heads}")
    logger.info(f"   - FFN size: {dff}")
    logger.info(f"   - MoE experts: {num_experts}")
    logger.info(f"   - Use MLA: {use_mla}")
    
    return config


def load_model(checkpoint_path: str, config: KimiLinearConfig, device: str = "cuda"):
    """加载模型和 checkpoint"""
    # 创建模型
    model = KimiLinearForCausalLM(config)
    
    # 加载 checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型权重
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        # 处理可能的 DataParallel 包装
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {k[7:] if k.startswith('module.') else k: v 
                          for k, v in model_state.items()}
        model.load_state_dict(model_state, strict=False)
        logger.info(f"✅ Checkpoint loaded successfully")
        if 'epoch' in checkpoint:
            logger.info(f"   - Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            logger.info(f"   - Step: {checkpoint['step']}")
    else:
        raise ValueError("Invalid checkpoint format: 'model' key not found")
    
    # 设置为评估模式
    model.eval()
    model.to(device)
    
    return model


@torch.no_grad()
def generate(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cuda"
):
    """生成翻译结果"""
    
    # 构建翻译 prompt
    prompt = f"Translate Portuguese to English:\n{input_text}\nEnglish: "
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    logger.info(f"Input prompt: {prompt}")
    logger.info(f"Input length: {input_ids.shape[1]} tokens")
    
    # 生成
    generated_tokens = []
    past_key_values = None
    
    for step in range(max_new_tokens):
        # 第一步：输入完整 prompt；后续步骤：只输入最后一个 token
        if step == 0:
            current_input = input_ids
        else:
            current_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
        
        # 前向传播
        outputs = model(
            input_ids=current_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        # 取最后一个 token 的 logits
        next_token_logits = logits[:, -1, :]
        
        # 应用温度
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过 top_p 的 tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        
        # 检查是否结束
        if token_id == tokenizer.eos_token_id:
            logger.info(f"Generation stopped at step {step+1} (EOS token)")
            break
        
        generated_tokens.append(token_id)
        
        # 每 20 步打印一次进度
        if (step + 1) % 20 == 0:
            partial_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info(f"Step {step+1}: {partial_text}")
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    # 配置
    checkpoint_path = "/workspace/checkpoints_kimi_translation_bak/best_e2_s2766.pt"
    tokenizer_path = "tok_en/tokenizer.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 80)
    logger.info("Kimi Translation Model Inference")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 1. 加载 tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Tokenizer")
    logger.info("=" * 80)
    tokenizer = load_tokenizer(tokenizer_path)
    
    # 2. 创建模型配置
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating Model Configuration")
    logger.info("=" * 80)
    config = create_model_config(vocab_size=len(tokenizer), use_mla=True)
    
    # 3. 加载模型
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Loading Model")
    logger.info("=" * 80)
    model = load_model(checkpoint_path, config, device)
    
    # 4. 测试推理
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Running Inference")
    logger.info("=" * 80)
    
    # 测试句子（葡萄牙语）
    test_sentences = [
        "Eu gosto de aprender idiomas.",
        "O Tom está procurando uma segunda opinião sobre o tratamento médico.",
        "Bom dia! Como você está?",
    ]
    
    for i, pt_text in enumerate(test_sentences, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Example {i}")
        logger.info(f"{'='*80}")
        logger.info(f"📥 Portuguese: {pt_text}")
        
        # 生成翻译
        en_text = generate(
            model=model,
            tokenizer=tokenizer,
            input_text=pt_text,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            device=device
        )
        
        logger.success(f"📤 English: {en_text}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP训练模型推理脚本
用于加载train_ddp_latest.py训练的checkpoint进行推理

使用方法:
    CUDA_VISIBLE_DEVICES=1 python inference/inference_ddp.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from loguru import logger
from typing import Optional, Dict, List

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入训练脚本中的模型定义和工具函数
from train_ddp_latest import (
    Transformer,
    MoEConfig,
    create_masks,
)


def load_tokenizer(tokenizer_path: str, max_length: int = 64, verbose: bool = False):
    """加载 tokenizer"""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.mask_token = "<mask>"
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = "right"
    
    if verbose:
        logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
        logger.info(f"   - Vocabulary size: {tokenizer.vocab_size}")
    
    return tokenizer


def create_model(
    input_vocab_size: int,
    target_vocab_size: int,
    num_layers: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    dff: int = 2048,
    max_length: int = 64,
    dropout_rate: float = 0.1,
    use_mla: bool = True,
    use_moe: bool = True,
    use_mtp: bool = True,
    pt_pad_token_id: int = None,
    en_pad_token_id: int = None,
    verbose: bool = False,
):
    """创建模型配置"""
    
    # MoE 配置
    moe_config = None
    if use_moe:
        moe_config = MoEConfig(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=d_model,
            intermediate_size=dff,
            hidden_act="silu",
            router_aux_loss_coef=0.005,
            use_moe=use_moe,
            n_routed_experts=8,
            routed_scaling_factor=0.8,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            n_shared_experts=None,
            moe_intermediate_size=dff,
        )
    
    # MLA 配置
    q_lora_rank = d_model // 2 if use_mla else None
    kv_lora_rank = 4 * (d_model // num_heads) if use_mla else None
    
    # 创建基础 Transformer 模型
    model = Transformer(
        num_layers=num_layers,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate,
        src_padding_idx=pt_pad_token_id,
        tgt_padding_idx=en_pad_token_id,
        use_rope=True,
        use_moe=use_moe,
        moe_config=moe_config,
        use_mla=use_mla,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
    )
    
    # 如果使用MTP，添加MTP功能
    if use_mtp:
        try:
            from core.models.deepseek_mtp import DeepSeekMTPConfig, add_mtp_to_transformer
            
            mtp_config = DeepSeekMTPConfig(
                hidden_size=d_model,
                num_nextn_predict_layers=2,
                vocab_size=target_vocab_size,
                max_position_embeddings=max_length,
                use_moe=use_moe,
                moe_config=moe_config,
                mtp_loss_weight=0.1,
            )
            
            model = add_mtp_to_transformer(model, mtp_config)
            if verbose:
                logger.info("✅ MTP功能已添加")
        except ImportError:
            logger.warning("⚠️ 无法导入MTP模块，跳过MTP功能")
    
    if verbose:
        logger.info(f"✅ Model: L={num_layers} H={d_model} Heads={num_heads} | MoE={use_moe} MLA={use_mla} MTP={use_mtp}")
    
    return model


def load_checkpoint(model, checkpoint_path: str, device: torch.device, verbose: bool = False):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 处理 DDP 包装的模型（移除 'module.' 前缀）
    model_state = checkpoint.get('model', checkpoint)
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}
    
    # 处理 MTP 包装不匹配的情况
    # 情况1：checkpoint 有 base_transformer 前缀，但当前模型没有
    if any(k.startswith('base_transformer.') for k in model_state.keys()) and not hasattr(model, 'base_transformer'):
        logger.warning("⚠️ 自动适配：移除MTP前缀（训练时use_mtp=True，推理时use_mtp=False）")
        new_state = {}
        for k, v in model_state.items():
            if k.startswith('base_transformer.'):
                new_key = k.replace('base_transformer.', '', 1)
                new_state[new_key] = v
            else:
                new_state[k] = v
        model_state = new_state
    
    # 情况2：checkpoint 没有 base_transformer 前缀，但当前模型有
    elif not any(k.startswith('base_transformer.') for k in model_state.keys()) and hasattr(model, 'base_transformer'):
        logger.warning("⚠️ 自动适配：添加MTP前缀（训练时use_mtp=False，推理时use_mtp=True）")
        new_state = {}
        mtp_keys = ['logits_to_embedding.weight', 'logits_to_hidden.weight']
        for k, v in model_state.items():
            if k in mtp_keys:
                # MTP特有的参数保持不变
                new_state[k] = v
            else:
                # 基础transformer的参数添加前缀
                new_state[f'base_transformer.{k}'] = v
        model_state = new_state
    
    # 加载权重（strict=False 允许加载部分权重）
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    
    # 只在verbose模式下显示详细的missing/unexpected keys
    if verbose:
        if missing_keys:
            logger.warning(f"⚠️ Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            logger.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)} keys (MTP相关)")
    
    # 简化的checkpoint信息
    epoch_info = f"E{checkpoint.get('epoch', '?')}"
    step_info = f"S{checkpoint.get('global_step', '?')}" if 'global_step' in checkpoint else ""
    loss_info = f"Loss={checkpoint.get('val_loss', checkpoint.get('train_loss', 0)):.4f}" if 'val_loss' in checkpoint or 'train_loss' in checkpoint else ""
    
    ckpt_name = checkpoint_path.split('/')[-1]
    info_str = " ".join(filter(None, [epoch_info, step_info, loss_info]))
    logger.info(f"✅ Checkpoint loaded: {ckpt_name} ({info_str})")
    
    # 移到设备并设置为评估模式
    model = model.to(device)
    
    # 注意：由于训练脚本中scaled_dot_product_attention函数硬编码了float32，
    # 推理时保持float32以避免dtype不匹配
    model = model.to(torch.float32)
    model.eval()
    
    return model


@torch.no_grad()
def translate_seq2seq(
    model,
    pt_tokenizer,
    en_tokenizer,
    input_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    verbose: bool = False,
):
    """
    使用Seq2Seq模型进行翻译（Encoder-Decoder结构）
    
    Args:
        model: Transformer模型（可能被MTP包装）
        pt_tokenizer: 葡萄牙语tokenizer
        en_tokenizer: 英语tokenizer
        input_text: 输入的葡萄牙语文本
        device: 设备
        max_new_tokens: 最大生成token数
        temperature: 采样温度（0.0表示贪婪解码）
        verbose: 是否显示详细日志
    """
    model.eval()
    
    # 如果模型被MTP包装，获取原始的Transformer
    if hasattr(model, 'base_transformer'):
        transformer = model.base_transformer
    else:
        transformer = model
    
    # 1. 编码输入文本（葡萄牙语）
    pt_ids = pt_tokenizer.encode(input_text, add_special_tokens=False)
    bos_id = pt_tokenizer.bos_token_id
    eos_id = pt_tokenizer.eos_token_id
    
    # 构建输入: [bos] + pt_ids + [eos]
    inp_ids = torch.tensor([[bos_id] + pt_ids + [eos_id]], dtype=torch.long, device=device)
    
    # 2. Encoder 推理（只需要运行一次）
    encoder_padding_mask, _, _ = create_masks(
        inp_ids, 
        torch.tensor([[en_tokenizer.bos_token_id]], device=device),
        src_pad_id=pt_tokenizer.pad_token_id,
        tgt_pad_id=en_tokenizer.pad_token_id,
    )
    
    # 运行encoder
    enc_output = transformer.encoder_model(inp_ids, src_mask=encoder_padding_mask)
    if isinstance(enc_output, tuple):
        enc_out = enc_output[0]  # 如果有router_logits，只取第一个
    else:
        enc_out = enc_output
    
    # 3. Decoder 自回归生成
    en_bos_id = en_tokenizer.bos_token_id
    en_eos_id = en_tokenizer.eos_token_id
    
    # 初始化：从 [bos] 开始
    generated_ids = [en_bos_id]
    
    for step in range(max_new_tokens):
        # 构建当前decoder输入
        tgt_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        
        # 创建decoder masks
        _, tgt_mask, enc_dec_mask = create_masks(
            inp_ids,
            tgt_ids,
            src_pad_id=pt_tokenizer.pad_token_id,
            tgt_pad_id=en_tokenizer.pad_token_id,
        )
        
        # Decoder前向传播
        dec_output = transformer.decoder_model(
            tgt_ids, 
            enc_out, 
            tgt_mask=tgt_mask, 
            enc_dec_mask=enc_dec_mask
        )
        
        # 解析输出
        if isinstance(dec_output, tuple):
            dec_out = dec_output[0]  # (dec_out, attention_weights) or (dec_out, attn, router_logits)
        else:
            dec_out = dec_output
        
        # 通过最终的输出层
        logits = transformer.final_layer(dec_out)  # [1, L, V]
        
        # 取最后一个token的logits
        next_token_logits = logits[0, -1, :]
        
        # 生成下一个token
        if temperature == 0.0:
            # 贪婪解码
            token_id = torch.argmax(next_token_logits, dim=-1).item()
        else:
            # 温度采样
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
        
        # 停止条件
        if token_id == en_eos_id:
            if verbose:
                logger.info(f"遇到 EOS token，停止生成（步骤 {step+1}）")
            break
        
        generated_ids.append(token_id)
        
        # 每隔20步打印一次当前结果（verbose模式下改为10步）
        log_interval = 10 if verbose else 20
        if verbose and (step + 1) % log_interval == 0:
            partial_text = en_tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
            logger.info(f"Step {step+1}: {partial_text}")
    
    # 解码生成的文本（跳过开头的bos）
    generated_text = en_tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # Tokenizer 配置
    pt_tokenizer_dir = "tok_pt"
    en_tokenizer_dir = "tok_en"
    pt_tokenizer_file = os.path.join(pt_tokenizer_dir, "tokenizer.json")
    en_tokenizer_file = os.path.join(en_tokenizer_dir, "tokenizer.json")
    
    # Checkpoint 路径
    checkpoint_path = "checkpoints_ddp_mla/mid_e16_s3552.pt"
    
    # 模型配置
    num_layers = 8
    d_model = 512
    num_heads = 8
    dff = 2048
    max_length = 64
    vocab_size = 2 ** 13  # 8192
    
    # 功能开关
    use_mla = True  # 使用MLA（Multi-head Latent Attention）
    use_moe = True  # 使用MoE
    use_mtp = False  # 使用MTP（Multi-Token Prediction）
    
    # 日志控制
    verbose = False  # 是否显示详细日志（加载过程、中间步骤等）
    # =================================================
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 简化的启动信息
    logger.info("=" * 80)
    logger.info("🚀 Portuguese → English Translation")
    logger.info("=" * 80)
    
    # 加载 tokenizers（静默）
    pt_tokenizer = load_tokenizer(pt_tokenizer_file, max_length, verbose=verbose)
    en_tokenizer = load_tokenizer(en_tokenizer_file, max_length, verbose=verbose)
    
    # 创建模型（静默）
    model = create_model(
        input_vocab_size=pt_tokenizer.vocab_size,
        target_vocab_size=en_tokenizer.vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_length=max_length,
        use_mla=use_mla,
        use_moe=use_moe,
        use_mtp=use_mtp,
        pt_pad_token_id=pt_tokenizer.pad_token_id,
        en_pad_token_id=en_tokenizer.pad_token_id,
        verbose=verbose,
    )
    
    # 加载检查点（只显示关键信息）
    model = load_checkpoint(model, checkpoint_path, device, verbose=verbose)
    
    # 测试样本
    logger.info("-" * 80)
    pt_text = "Eu gosto de aprender idiomas."
    logger.info(f"📥 Input:  {pt_text}")
    
    try:
        en_text = translate_seq2seq(
            model=model,
            pt_tokenizer=pt_tokenizer,
            en_tokenizer=en_tokenizer,
            input_text=pt_text,
            device=device,
            max_new_tokens=64,
            temperature=0.0,  # 贪婪解码
            verbose=verbose,
        )
        
        logger.success(f"📤 Output: {en_text}")
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


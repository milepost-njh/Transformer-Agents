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
def translate_with_mtp_speculative(
    model,
    pt_tokenizer,
    en_tokenizer,
    input_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    num_speculative_tokens: int = 3,
    verbose: bool = False,
):
    """
    使用MTP推测解码进行翻译（利用MTP预测头预测多个token）
    
    这是真正的MTP推理：利用MTP预测头一次预测未来多个token，
    然后一起验证，如果验证通过则接受多个token（加速推理）
    
    Args:
        model: MTP包装的Transformer模型
        num_speculative_tokens: 每次推测的token数量 (包含主模型预测的1个 + MTP预测的N-1个)
                                注意：不能超过训练时的 MTP 层数 + 1
    """
    model.eval()
    
    if not hasattr(model, 'base_transformer'):
        logger.warning("模型未被MTP包装，退回到标准解码")
        return translate_seq2seq(model, pt_tokenizer, en_tokenizer, input_text, device, max_new_tokens, temperature, verbose)
    
    transformer = model.base_transformer
    
    # 1. 编码输入（与标准方法相同）
    pt_ids = pt_tokenizer.encode(input_text, add_special_tokens=False)
    bos_id = pt_tokenizer.bos_token_id
    eos_id = pt_tokenizer.eos_token_id
    inp_ids = torch.tensor([[bos_id] + pt_ids + [eos_id]], dtype=torch.long, device=device)
    
    # 2. Encoder推理
    encoder_padding_mask, _, _ = create_masks(
        inp_ids, 
        torch.tensor([[en_tokenizer.bos_token_id]], device=device),
        src_pad_id=pt_tokenizer.pad_token_id,
        tgt_pad_id=en_tokenizer.pad_token_id,
    )
    
    enc_output = transformer.encoder_model(inp_ids, src_mask=encoder_padding_mask)
    if isinstance(enc_output, tuple):
        enc_out = enc_output[0]
    else:
        enc_out = enc_output
    
    # 3. MTP推测解码
    en_bos_id = en_tokenizer.bos_token_id
    en_eos_id = en_tokenizer.eos_token_id
    generated_ids = [en_bos_id]
    
    total_speculations = 0  # MTP推测次数
    accepted_speculations = 0  # 接受的MTP token总数（不包括主预测）
    total_speculated_tokens = 0  # MTP推测的token总数
    total_tokens_generated = 0  # 生成的token总数（包括主预测+MTP）
    
    step = 0
    while step < max_new_tokens:
        # 构建当前decoder输入
        tgt_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        
        _, tgt_mask, enc_dec_mask = create_masks(
            inp_ids, tgt_ids,
            src_pad_id=pt_tokenizer.pad_token_id,
            tgt_pad_id=en_tokenizer.pad_token_id,
        )
        
        # 完整的模型前向传播（包括MTP）
        model_output = model(inp_ids, tgt_ids, encoder_padding_mask, tgt_mask, enc_dec_mask)
        
        # 解析输出
        if isinstance(model_output, tuple) and len(model_output) >= 4:
            # 有MTP输出：(logits, attn, router_logits, mtp_logits)
            logits, _, _, mtp_logits = model_output[:4]
            has_mtp = mtp_logits is not None and len(mtp_logits) > 0
        elif isinstance(model_output, tuple):
            logits = model_output[0]
            has_mtp = False
        else:
            logits = model_output
            has_mtp = False
        
        # 获取下一个token（主预测）
        next_token_logits = logits[0, -1, :]
        if temperature == 0.0:
            main_token_id = torch.argmax(next_token_logits, dim=-1).item()
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            main_token_id = torch.multinomial(probs, num_samples=1).item()
        
        if verbose and has_mtp:
            top5_main_vals, top5_main_ids = torch.topk(next_token_logits, k=5)
            logger.debug(f"Step {step}: 主模型top5={top5_main_ids.tolist()}, 选择={main_token_id}")
        
        # 停止条件检查
        if main_token_id == en_eos_id:
            if verbose:
                logger.info(f"遇到 EOS，停止生成（步骤 {step+1}）")
            break
        
        # 如果有MTP预测头，尝试推测未来token（包括主预测）
        speculative_tokens = [main_token_id]  # 从主预测开始
        if has_mtp and step + num_speculative_tokens <= max_new_tokens:
            # 使用MTP预测头预测未来token
            for mtp_idx in range(min(num_speculative_tokens - 1, len(mtp_logits))):
                mtp_token_logits = mtp_logits[mtp_idx][0, -1, :]  # MTP预测的logits
                if temperature == 0.0:
                    mtp_token_id = torch.argmax(mtp_token_logits, dim=-1).item()
                else:
                    probs = torch.softmax(mtp_token_logits / temperature, dim=-1)
                    mtp_token_id = torch.multinomial(probs, num_samples=1).item()
                
                if verbose:
                    top5_vals, top5_ids = torch.topk(mtp_token_logits, k=5)
                    logger.debug(f"  MTP层{mtp_idx} top5预测: {top5_ids.tolist()}, 选择: {mtp_token_id}")
                
                if mtp_token_id == en_eos_id:
                    break
                speculative_tokens.append(mtp_token_id)
            
            # 验证推测的token（如果推测了多个token）
            if len(speculative_tokens) > 1:
                total_speculations += 1
                num_mtp_tokens = len(speculative_tokens) - 1  # MTP推测的token数（不含主预测）
                total_speculated_tokens += num_mtp_tokens
                
                # 将推测的所有token加入，然后用主模型验证
                test_ids = generated_ids + speculative_tokens
                test_tgt_ids = torch.tensor([test_ids], dtype=torch.long, device=device)
                
                _, test_tgt_mask, test_enc_dec_mask = create_masks(
                    inp_ids, test_tgt_ids,
                    src_pad_id=pt_tokenizer.pad_token_id,
                    tgt_pad_id=en_tokenizer.pad_token_id,
                )
                
                # 验证推测
                verify_output = transformer(inp_ids, test_tgt_ids, encoder_padding_mask, test_tgt_mask, test_enc_dec_mask)
                if isinstance(verify_output, tuple):
                    verify_logits = verify_output[0]
                else:
                    verify_logits = verify_output
                
                # 检查每个推测的token是否正确
                accepted_count = 0
                for i, spec_token in enumerate(speculative_tokens):
                    pos = len(generated_ids) + i - 1
                    if pos >= 0 and pos < verify_logits.shape[1]:
                        predicted_token = torch.argmax(verify_logits[0, pos, :], dim=-1).item()
                        if predicted_token == spec_token:
                            accepted_count += 1
                        else:
                            # 推测失败，只接受到这里
                            if verbose:
                                logger.debug(f"位置{i}: MTP预测={spec_token} vs 主模型={predicted_token} (不匹配)")
                            break
                    else:
                        if verbose:
                            logger.debug(f"位置{i}: pos={pos} 超出范围 {verify_logits.shape[1]}")
                
                # 接受验证通过的token
                accepted_tokens = speculative_tokens[:accepted_count] if accepted_count > 0 else [main_token_id]
                generated_ids.extend(accepted_tokens)
                total_tokens_generated += len(accepted_tokens)
                
                # 统计MTP接受的token数（减去主预测的1个）
                num_mtp_accepted = max(0, accepted_count - 1)
                accepted_speculations += num_mtp_accepted
                
                if accepted_count > 1:
                    if verbose:
                        logger.info(f"✅ 推测成功：接受 {accepted_count}/{len(speculative_tokens)} 个token (MTP: {num_mtp_accepted}/{num_mtp_tokens})")
                elif verbose:
                    logger.info(f"❌ 推测失败：只接受主预测token")
                
                step += len(accepted_tokens)
            else:
                # 只有主预测
                generated_ids.append(main_token_id)
                total_tokens_generated += 1
                step += 1
        else:
            # 没有MTP或已接近最大长度，使用标准解码
            generated_ids.append(main_token_id)
            total_tokens_generated += 1
            step += 1
        
        # 定期打印进度
        if verbose and step % 10 == 0:
            partial_text = en_tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
            logger.info(f"Step {step}: {partial_text}")
    
    # 解码生成的文本
    generated_text = en_tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
    
    # 统计信息
    stats = {
        "total_tokens": total_tokens_generated,
        "total_speculations": total_speculations,  # MTP推测次数
        "total_speculated_tokens": total_speculated_tokens,  # MTP推测的token总数
        "accepted_speculations": accepted_speculations,  # 接受的MTP token数
        "acceptance_rate": accepted_speculations / total_speculated_tokens if total_speculated_tokens > 0 else 0.0,
        "speculations_per_step": total_speculated_tokens / total_speculations if total_speculations > 0 else 0.0,
    }
    
    return generated_text, stats


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
    标准自回归解码（每次只预测1个token）
    
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


def run_inference_batch(
    model,
    pt_tokenizer,
    en_tokenizer,
    test_samples: List[str],
    device: torch.device,
    use_mtp_decoding: bool = False,
    verbose: bool = False,
):
    """
    对一批样本进行推理
    
    Args:
        model: 已加载的模型
        use_mtp_decoding: True=使用MTP推测解码, False=标准自回归解码
    """
    import time
    
    results = []
    all_stats = []
    total_time = 0.0
    
    for idx, pt_text in enumerate(test_samples, 1):
        logger.info("-" * 80)
        logger.info(f"Sample {idx}/{len(test_samples)}")
        logger.info(f"📥 Input:  {pt_text}")
        
        try:
            start_time = time.time()
            
            if use_mtp_decoding:
                # MTP推测解码
                # num_speculative_tokens 设为 3 (主预测 + 2个MTP预测)
                # 因为训练时只训练了 2 层 MTP，最多只能预测未来 2 个 token
                # 理论最大加速比 = 3/2 = 1.5x
                en_text, stats = translate_with_mtp_speculative(
                    model=model,
                    pt_tokenizer=pt_tokenizer,
                    en_tokenizer=en_tokenizer,
                    input_text=pt_text,
                    device=device,
                    max_new_tokens=64,
                    temperature=0.0,
                    num_speculative_tokens=3,
                    verbose=verbose,
                )
                all_stats.append(stats)
            else:
                # 标准自回归解码
                en_text = translate_seq2seq(
                    model=model,
                    pt_tokenizer=pt_tokenizer,
                    en_tokenizer=en_tokenizer,
                    input_text=pt_text,
                    device=device,
                    max_new_tokens=64,
                    temperature=0.0,
                    verbose=verbose,
                )
                all_stats.append(None)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            logger.success(f"📤 Output: {en_text}")
            logger.info(f"⏱️  Time: {elapsed_time:.3f}s")
            
            if use_mtp_decoding and stats:
                logger.info(f"📊 MTP Stats: {stats['accepted_speculations']}/{stats['total_speculated_tokens']} "
                           f"tokens accepted ({stats['acceptance_rate']:.1%}), "
                           f"{stats['total_speculations']} speculations")
            
            results.append(en_text)
            
        except Exception as e:
            logger.error(f"❌ Translation failed: {e}")
            results.append(f"[ERROR: {str(e)}]")
            all_stats.append(None)
            if verbose:
                import traceback
                traceback.print_exc()
    
    logger.info("=" * 80)
    logger.info(f"⏱️  Total time: {total_time:.3f}s")
    logger.info("")
    
    return results, all_stats, total_time


def main():
    """主函数 - MTP推理对比实验"""
    # ==================== 配置区域 ====================
    # Tokenizer 配置
    pt_tokenizer_dir = "tok_pt"
    en_tokenizer_dir = "tok_en"
    pt_tokenizer_file = os.path.join(pt_tokenizer_dir, "tokenizer.json")
    en_tokenizer_file = os.path.join(en_tokenizer_dir, "tokenizer.json")
    
    # Checkpoint 路径
    # 选项1: 原始checkpoint (mtp_loss_weight=0.1, MTP质量差)
    # checkpoint_path = "checkpoints_ddp_mla/mid_e16_s3552.pt"
    
    # 选项2: MTP强化训练的checkpoint (mtp_loss_weight=0.5, MTP质量好)
    checkpoint_path = "checkpoints_ddp_mla_mtp_strong/mid_e12_s2664.pt"  # 使用新训练的MTP模型
    
    # 模型配置
    num_layers = 8
    d_model = 512
    num_heads = 8
    dff = 2048
    max_length = 64
    use_mla = True  # 使用MLA
    use_moe = True  # 使用MoE
    
    # 日志控制
    verbose = False  # 是否显示详细日志（推荐先用False，如果需要调试再改为True）
    
    # 调试MTP: 设为True可以看到每步的预测细节
    debug_mtp = False  # 调试MTP预测过程
    # =================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 启动信息
    logger.info("=" * 80)
    logger.info("🚀 MTP推理对比实验: 标准解码 vs MTP推测解码")
    logger.info("=" * 80)
    logger.info(f"📦 Checkpoint: {checkpoint_path}")
    logger.info("")
    
    # 加载 tokenizers
    pt_tokenizer = load_tokenizer(pt_tokenizer_file, max_length, verbose=verbose)
    en_tokenizer = load_tokenizer(en_tokenizer_file, max_length, verbose=verbose)
    
    # 测试样本
    test_samples = [
        # 我喜欢学习语言
        "Eu gosto de aprender idiomas.",
        
        # 人工智能正在改变我们的生活方式，从医疗保健到交通运输
        "A inteligência artificial está mudando a forma como vivemos, desde a saúde até o transporte.",
        
        # 在这个美丽的夏日，孩子们在公园里快乐地玩耍
        "Neste lindo dia de verão, as crianças brincam felizes no parque.",
    ]
    
    logger.info(f"📝 测试样本数: {len(test_samples)}\n")
    
    # ============ 方法1: 标准自回归解码 ============
    logger.info("=" * 80)
    logger.info("1️⃣  标准自回归解码 (每次预测1个token)")
    logger.info("=" * 80)
    
    # 加载标准模型
    model_standard = create_model(
        input_vocab_size=pt_tokenizer.vocab_size,
        target_vocab_size=en_tokenizer.vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_length=max_length,
        use_mla=use_mla,
        use_moe=use_moe,
        use_mtp=False,  # 不使用MTP
        pt_pad_token_id=pt_tokenizer.pad_token_id,
        en_pad_token_id=en_tokenizer.pad_token_id,
        verbose=verbose,
    )
    model_standard = load_checkpoint(model_standard, checkpoint_path, device, verbose=verbose)
    
    # 推理
    results_standard, _, time_standard = run_inference_batch(
        model=model_standard,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        test_samples=test_samples,
        device=device,
        use_mtp_decoding=False,
        verbose=verbose,
    )
    
    # 释放显存
    del model_standard
    torch.cuda.empty_cache()
    
    # ============ 方法2: MTP推测解码 ============
    logger.info("=" * 80)
    logger.info("2️⃣  MTP推测解码 (预测多个token并验证)")
    logger.info("=" * 80)
    
    # 加载MTP模型
    model_mtp = create_model(
        input_vocab_size=pt_tokenizer.vocab_size,
        target_vocab_size=en_tokenizer.vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_length=max_length,
        use_mla=use_mla,
        use_moe=use_moe,
        use_mtp=True,  # 使用MTP
        pt_pad_token_id=pt_tokenizer.pad_token_id,
        en_pad_token_id=en_tokenizer.pad_token_id,
        verbose=verbose,
    )
    model_mtp = load_checkpoint(model_mtp, checkpoint_path, device, verbose=verbose)
    
    # 推理
    results_mtp, stats_mtp, time_mtp = run_inference_batch(
        model=model_mtp,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        test_samples=test_samples,
        device=device,
        use_mtp_decoding=True,
        verbose=verbose,
    )
    
    # ============ 对比结果 ============
    logger.info("=" * 80)
    logger.info("📊 对比结果")
    logger.info("=" * 80)
    
    for idx, pt_text in enumerate(test_samples, 1):
        logger.info(f"\n{'─' * 80}")
        logger.info(f"样本 {idx}:")
        logger.info(f"  🇵🇹 输入: {pt_text}")
        logger.info(f"  🇬🇧 标准解码: {results_standard[idx-1]}")
        logger.info(f"  🇬🇧 MTP解码:  {results_mtp[idx-1]}")
        
        if results_standard[idx-1] == results_mtp[idx-1]:
            logger.info(f"  ✅ 输出相同")
        else:
            logger.warning(f"  ⚠️  输出不同")
        
        if stats_mtp[idx-1]:
            stats = stats_mtp[idx-1]
            logger.info(f"  📊 MTP: {stats['accepted_speculations']}/{stats['total_speculated_tokens']} "
                       f"tokens被接受 (接受率 {stats['acceptance_rate']:.1%}), "
                       f"{stats['total_speculations']}次推测")
    
    # ============ 总体统计 ============
    logger.info("\n" + "=" * 80)
    logger.info("📈 总体统计")
    logger.info("=" * 80)
    
    logger.info(f"⏱️  时间对比:")
    logger.info(f"  - 标准解码总时间: {time_standard:.3f}s")
    logger.info(f"  - MTP解码总时间:  {time_mtp:.3f}s")
    if time_standard > 0:
        speedup = time_standard / time_mtp if time_mtp > 0 else 0
        logger.info(f"  - 加速比: {speedup:.2f}x")
    
    total_spec_count = sum(s['total_speculations'] for s in stats_mtp if s)
    total_spec_tokens = sum(s['total_speculated_tokens'] for s in stats_mtp if s)
    total_accepted = sum(s['accepted_speculations'] for s in stats_mtp if s)
    avg_acceptance = total_accepted / total_spec_tokens if total_spec_tokens > 0 else 0.0
    
    logger.info(f"\n📊 MTP统计:")
    logger.info(f"  - 总推测次数: {total_spec_count} 次")
    logger.info(f"  - 推测token总数: {total_spec_tokens} 个")
    logger.info(f"  - 接受token数: {total_accepted} 个")
    logger.info(f"  - 平均接受率: {avg_acceptance:.1%}")
    
    logger.info("\n" + "=" * 80)
    logger.info("💡 MTP推理原理与分析")
    logger.info("=" * 80)
    logger.info("标准解码: 每次预测1个token → 需要N次forward生成N个token")
    logger.info("MTP解码:  每次先用主模型预测1个token，再用MTP预测K个token → 验证MTP预测 → 接受或回退")
    
    # 计算平均每次MTP推测的token数
    avg_spec_tokens = total_spec_tokens / total_spec_count if total_spec_count > 0 else 0.0
    logger.info(f"         当前配置: 平均每次MTP推测 {avg_spec_tokens:.1f} 个token")
    logger.info(f"         MTP接受率: {avg_acceptance:.1%} (推测准确性)")
    logger.info(f"         理论加速比: 1 + {avg_spec_tokens:.1f} * {avg_acceptance:.1%} = {1 + avg_spec_tokens * avg_acceptance:.2f}x")
    logger.info(f"         实际加速比: {speedup:.2f}x (受验证开销、样本长度等因素影响)")
    logger.info("")
    
    if avg_acceptance < 0.1:
        logger.warning("⚠️  MTP接受率过低，可能的原因：")
        logger.warning("   1. MTP预测头训练不足（最可能）")
        logger.warning("      - 训练时mtp_loss_weight过小，MTP没学到有用模式")
        logger.warning("      - 建议：增加mtp_loss_weight（如0.3-0.5）重新训练")
        logger.warning("   2. MTP特征转换层质量差")
        logger.warning("      - logits_to_embedding/hidden层可能接近随机初始化")
        logger.warning("   3. 当前checkpoint可能主要优化了主任务，MTP是副产品")
        logger.warning("")
        logger.warning("💡 改进建议：")
        logger.warning("   - 如果要充分利用MTP加速，需要用更大的mtp_loss_weight重新训练")
        logger.warning("   - 或者使用专门训练的MTP checkpoint")
        logger.warning("   - 当前checkpoint更适合用标准解码")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


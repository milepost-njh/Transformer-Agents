#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真正的MLA vs 标准注意力 KV-cache 效率对比脚本
实现真正的KV-cache机制，测试自回归生成中的内存使用和性能

使用方法:
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths  64
"""

import os
import sys
import time
import torch
import torch.nn as nn
import psutil
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import json
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入必要的模块
from transformers import PreTrainedTokenizerFast
from train_tmp import (
    Transformer, MoEConfig, create_masks, get_device
)


def get_gpu_memory():
    """获取GPU显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class KVCacheTracker:
    """KV-cache追踪器，用于监控GPU显存使用"""
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, 
                 use_mla: bool = False, kv_lora_rank: int = None, 
                 qk_nope_head_dim: int = None, v_head_dim: int = None):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        
        # 记录每一步的KV-cache大小
        self.cache_sizes = []
    
    def calculate_cache_size(self, seq_len: int) -> float:
        """
        计算给定序列长度的KV-cache大小（MB）
        
        对于标准注意力:
            KV-cache = 2 * num_layers * num_heads * head_dim * seq_len * 2 bytes (bf16)
        
        对于MLA:
            K-cache = num_layers * (kv_lora_rank + qk_rope_head_dim) * seq_len * 2 bytes (bf16)
            V-cache = num_layers * num_heads * v_head_dim * seq_len * 2 bytes (bf16)
        """
        if self.use_mla:
            # MLA模式：压缩的KV-cache
            qk_rope_head_dim = self.head_dim // 2  # 默认值
            # K的压缩表示
            k_cache_params = self.kv_lora_rank + qk_rope_head_dim
            # V的表示
            v_cache_params = self.num_heads * self.v_head_dim
            total_params = (k_cache_params + v_cache_params) * seq_len * self.num_layers
        else:
            # 标准模式：完整的KV-cache
            # K和V都是 [num_heads, seq_len, head_dim]
            total_params = 2 * self.num_heads * self.head_dim * seq_len * self.num_layers
        
        # 转换为MB（bf16: 2字节）
        size_mb = total_params * 2 / (1024 * 1024)
        return size_mb
    
    def record_step(self, seq_len: int):
        """记录一步的KV-cache大小"""
        size = self.calculate_cache_size(seq_len)
        self.cache_sizes.append(size)
        return size
    
    def get_compression_ratio(self, standard_tracker: 'KVCacheTracker', seq_len: int) -> float:
        """计算相对于标准注意力的压缩比"""
        mla_size = self.calculate_cache_size(seq_len)
        standard_size = standard_tracker.calculate_cache_size(seq_len)
        if standard_size > 0:
            return (standard_size - mla_size) / standard_size * 100
        return 0.0


class KVCacheInferenceEngine:
    """支持KV-cache的推理引擎"""
    
    def __init__(self, model: nn.Module, pt_tokenizer, en_tokenizer, 
                 device: str, use_mla: bool = False, config: dict = None):
        self.model = model
        self.pt_tokenizer = pt_tokenizer
        self.en_tokenizer = en_tokenizer
        self.device = device
        self.use_mla = use_mla
        self.config = config or {}
        
        # 创建KV-cache追踪器
        num_layers = config.get('num_layers', 8)
        num_heads = config.get('num_heads', 8)
        d_model = config.get('d_model', 512)
        head_dim = d_model // num_heads
        
        self.kv_tracker = KVCacheTracker(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            use_mla=use_mla,
            kv_lora_rank=d_model // 4 if use_mla else None,
            qk_nope_head_dim=head_dim // 2 if use_mla else None,
            v_head_dim=head_dim if use_mla else None
        )
    
    def encode_input(self, text: str) -> torch.Tensor:
        """编码输入文本"""
        def encode_with_bos_eos(tokenizer, text: str):
            ids = tokenizer.encode(text, add_special_tokens=False)
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            if bos_id is None or eos_id is None:
                raise ValueError("Tokenizer must have bos_token and eos_token")
            return [bos_id] + ids + [eos_id]
        
        inp_ids = encode_with_bos_eos(self.pt_tokenizer, text)
        encoder_input = torch.tensor(inp_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return encoder_input
    
    def decode_output(self, token_ids: List[int]) -> str:
        """解码输出token IDs"""
        return self.en_tokenizer.decode(
            [i for i in token_ids if i < len(self.en_tokenizer)],
            skip_special_tokens=True
        )
    
    def generate_with_kv_cache(self, input_text: str, max_new_tokens: int = 64, 
                              use_real_cache: bool = True, verbose_tokens: bool = True) -> Dict:
        """
        使用KV-cache进行自回归生成
        
        Args:
            input_text: 输入文本
            max_new_tokens: 最大生成token数
            use_real_cache: 是否使用真正的KV-cache（True）还是模拟cache（False）
            verbose_tokens: 是否详细打印每个生成的token
        """
        start_time = time.time()
        self.model.eval()
        
        # 编码输入
        encoder_input = self.encode_input(input_text)
        
        # 初始化decoder输入
        start_id = self.en_tokenizer.bos_token_id
        end_id = self.en_tokenizer.eos_token_id
        decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=self.device)
        
        generated_tokens = []
        generated_texts = []  # 记录每个token的文本
        step_times = []
        
        with torch.no_grad():
            if use_real_cache:
                # ====== 真正的KV-cache实现 ======
                logger.info(f"  [真正Cache] 开始生成（Prefill + Decode两阶段）")
                
                # Prefill阶段：运行encoder一次，生成encoder cache
                enc_pad_mask, _, _ = create_masks(
                    encoder_input, decoder_input,
                    src_pad_id=self.pt_tokenizer.pad_token_id,
                    tgt_pad_id=self.en_tokenizer.pad_token_id,
                )
                
                # Encoder只运行一次（prefill）
                logger.info(f"  [Prefill阶段] Encoder输入长度: {encoder_input.size(1)} tokens")
                enc_output = self.model.encoder_model(
                    encoder_input, 
                    src_mask=enc_pad_mask,
                    use_cache=True
                )
                logger.info(f"  [Prefill阶段] Encoder运行完成，后续将复用encoder输出")
                
                # 处理encoder输出
                if isinstance(enc_output, tuple):
                    if len(enc_output) == 3:
                        encoder_outputs, _, encoder_cache = enc_output
                    else:
                        encoder_outputs, encoder_cache = enc_output
                else:
                    encoder_outputs = enc_output
                    encoder_cache = None
                
                # KV-cache: 只需要decoder的self-attention cache
                # encoder cache在encoder内部使用，cross-attention不需要cache（因为encoder输出固定）
                past_key_values = {"encoder": None, "decoder": None}
                
                # Decode阶段：逐token生成
                logger.info(f"  [Decode阶段] 开始逐token生成，目标生成 {max_new_tokens} tokens")
                for step in range(max_new_tokens):
                    step_start = time.time()
                    
                    # 记录KV-cache大小
                    current_seq_len = decoder_input.size(1)
                    cache_size = self.kv_tracker.record_step(current_seq_len)
                    
                    # 创建masks（只需要decoder的mask）
                    _, dec_mask, enc_dec_pad_mask = create_masks(
                        encoder_input, decoder_input,
                        src_pad_id=self.pt_tokenizer.pad_token_id,
                        tgt_pad_id=self.en_tokenizer.pad_token_id,
                    )
                    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                    
                    # 第一步：完整输入；后续步骤：只输入最后一个token
                    if step == 0:
                        # 第一步：输入完整的start token
                        current_decoder_input = decoder_input
                        if step % 20 == 0:
                            logger.info(f"    Step {step}: Decoder输入长度={current_decoder_input.size(1)} | 累积序列长度={decoder_input.size(1)}")
                    else:
                        # 后续步骤：只输入最后一个token，复用cache
                        current_decoder_input = decoder_input[:, -1:]
                        if step % 20 == 0:
                            logger.info(f"    Step {step}: Decoder输入长度={current_decoder_input.size(1)} ✅仅1个token | 累积序列长度={decoder_input.size(1)} | KV-cache: {cache_size:.2f}MB")
                        # 调整mask维度
                        dec_mask = dec_mask[:, :, -1:, :]
                        enc_dec_mask = enc_dec_mask[:, :, -1:, :]
                    
                    # 前向传播（使用真正的KV-cache）
                    model_output = self.model(
                        None,  # encoder_input不需要了，复用encoder_outputs
                        current_decoder_input,
                        src_mask=None,  # encoder不再运行
                        tgt_mask=dec_mask,
                        enc_dec_mask=enc_dec_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        encoder_outputs=encoder_outputs  # 复用encoder输出
                    )
                    
                    # 处理输出
                    if len(model_output) == 4:
                        logits, attn, present_key_values, router_logits = model_output
                    elif len(model_output) == 3:
                        logits, attn, present_key_values = model_output
                    else:
                        raise ValueError("Unexpected model output format")
                    
                    # 更新cache
                    past_key_values = present_key_values
                    
                    # 取最后一个token的logits
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    
                    # 检查是否结束
                    if next_token_id.item() == end_id:
                        break
                    
                    # 添加到生成序列
                    token_id = next_token_id.item()
                    generated_tokens.append(token_id)
                    
                    # 解码当前token的文本
                    token_text = self.en_tokenizer.decode([token_id], skip_special_tokens=False)
                    generated_texts.append(token_text)
                    
                    # 实时打印生成的token
                    if verbose_tokens:
                        cumulative_text = ''.join(generated_texts)
                        logger.info(f"    🔤 Token {step+1}: ID={token_id:5d} | 文本='{token_text}' | 累积='{cumulative_text}' | 耗时={step_time*1000:.1f}ms")
                    
                    decoder_input = torch.cat([decoder_input, next_token_id.unsqueeze(0)], dim=-1)
            else:
                # ====== 模拟KV-cache（旧实现，用于对比） ======
                logger.info(f"  [模拟Cache] 开始生成（每步重算整个序列）")
                for step in range(max_new_tokens):
                    step_start = time.time()
                    
                    # 记录KV-cache大小
                    current_seq_len = decoder_input.size(1)
                    cache_size = self.kv_tracker.record_step(current_seq_len)
                    
                    if step % 20 == 0:
                        logger.info(f"    Step {step}: Encoder输入长度={encoder_input.size(1)} | Decoder输入长度={decoder_input.size(1)} ❌每步重算")
                    
                    # 创建masks
                    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
                        encoder_input, decoder_input,
                        src_pad_id=self.pt_tokenizer.pad_token_id,
                        tgt_pad_id=self.en_tokenizer.pad_token_id,
                    )
                    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                    
                    # 前向传播（每步重新计算整个序列）
                    model_output = self.model(
                        encoder_input, decoder_input,
                        src_mask=enc_pad_mask,
                        tgt_mask=dec_mask,
                        enc_dec_mask=enc_dec_mask
                    )
                    
                    # 处理输出
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        logits, attn = model_output[0], model_output[1]
                    else:
                        raise ValueError("Unexpected model output format")
                    
                    # 取最后一个token的logits
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    
                    # 检查是否结束
                    if next_token_id.item() == end_id:
                        break
                    
                    # 添加到生成序列
                    token_id = next_token_id.item()
                    generated_tokens.append(token_id)
                    
                    # 解码当前token的文本
                    token_text = self.en_tokenizer.decode([token_id], skip_special_tokens=False)
                    generated_texts.append(token_text)
                    
                    # 实时打印生成的token
                    if verbose_tokens:
                        cumulative_text = ''.join(generated_texts)
                        logger.info(f"    🔤 Token {step+1}: ID={token_id:5d} | 文本='{token_text}' | 累积='{cumulative_text}' | 耗时={step_time*1000:.1f}ms")
                    
                    decoder_input = torch.cat([decoder_input, next_token_id.unsqueeze(0)], dim=-1)
        
        generation_time = time.time() - start_time
        
        return {
            'output': self.decode_output(generated_tokens),
            'generation_time': generation_time,
            'avg_step_time': sum(step_times) / len(step_times) if step_times else 0,
            'num_tokens_generated': len(generated_tokens),
            'kv_cache_sizes': self.kv_tracker.cache_sizes,
            'max_kv_cache_size': max(self.kv_tracker.cache_sizes) if self.kv_tracker.cache_sizes else 0,
            'final_kv_cache_size': self.kv_tracker.cache_sizes[-1] if self.kv_tracker.cache_sizes else 0,
            'step_times': step_times,
            'generated_tokens': generated_tokens,
            'generated_texts': generated_texts
        }


def load_model_and_tokenizers(checkpoint_path: str, use_mla: bool, device: str):
    """加载模型和tokenizer"""
    logger.info(f"加载模型: {'MLA' if use_mla else 'Standard'}")
    
    # 加载tokenizers
    pt_tokenizer_path = "tok_pt/tokenizer.json"
    en_tokenizer_path = "tok_en/tokenizer.json"
    
    if not os.path.exists(pt_tokenizer_path):
        raise FileNotFoundError(f"Portuguese tokenizer not found at {pt_tokenizer_path}")
    if not os.path.exists(en_tokenizer_path):
        raise FileNotFoundError(f"English tokenizer not found at {en_tokenizer_path}")
    
    pt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=pt_tokenizer_path)
    pt_tokenizer.pad_token = "<pad>"
    pt_tokenizer.unk_token = "<unk>"
    pt_tokenizer.bos_token = "<s>"
    pt_tokenizer.eos_token = "</s>"
    pt_tokenizer.mask_token = "<mask>"
    
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=en_tokenizer_path)
    en_tokenizer.pad_token = "<pad>"
    en_tokenizer.unk_token = "<unk>"
    en_tokenizer.bos_token = "<s>"
    en_tokenizer.eos_token = "</s>"
    en_tokenizer.mask_token = "<mask>"
    
    # 模型配置
    num_layers = 8
    d_model = 512
    num_heads = 8
    dff = 2048
    max_length = 512
    dropout_rate = 0.1
    
    # MoE配置
    moe_config = MoEConfig(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=d_model,
        intermediate_size=dff,
        hidden_act="silu",
        router_aux_loss_coef=0.005,
        use_moe=True,
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
    
    # 创建模型
    model = Transformer(
        num_layers=num_layers,
        input_vocab_size=len(pt_tokenizer),
        target_vocab_size=len(en_tokenizer),
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate,
        src_padding_idx=pt_tokenizer.pad_token_id,
        tgt_padding_idx=en_tokenizer.pad_token_id,
        use_rope=True,
        use_moe=True,
        moe_config=moe_config,
        use_mla=use_mla,
        q_lora_rank=d_model // 2 if use_mla else None,
        kv_lora_rank=d_model // 4 if use_mla else None,
    )
    
    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            # 处理DataParallel包装的模型
            if any(key.startswith('module.') for key in model_state.keys()):
                model_state = {k[7:] if k.startswith('module.') else k: v 
                             for k, v in model_state.items()}
            model.load_state_dict(model_state, strict=False)
            logger.info(f"✅ Checkpoint loaded from {checkpoint_path}")
    else:
        logger.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    model.to(device)
    
    config = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'd_model': d_model,
        'dff': dff,
        'max_length': max_length
    }
    
    return model, pt_tokenizer, en_tokenizer, config


def benchmark_model(checkpoint_path: str, use_mla: bool, test_input: str, 
                   max_new_tokens: int, device: str, use_real_cache: bool = True) -> Dict:
    """
    对单个模型进行基准测试
    
    Args:
        checkpoint_path: 模型checkpoint路径
        use_mla: 是否使用MLA
        test_input: 测试输入文本
        max_new_tokens: 最大生成token数
        device: 设备
        use_real_cache: 是否使用真正的KV-cache（False=模拟cache）
    """
    cache_type = "真实KV-cache" if use_real_cache else "模拟KV-cache"
    model_name = "MLA" if use_mla else "Standard"
    logger.info(f"\n{'='*60}")
    logger.info(f"测试 {model_name} 模型 ({cache_type}, 生成 {max_new_tokens} tokens)")
    logger.info(f"{'='*60}")
    
    # 清理内存
    clear_memory()
    
    # 加载模型
    model, pt_tokenizer, en_tokenizer, config = load_model_and_tokenizers(
        checkpoint_path, use_mla, device
    )
    
    logger.info(f"模型加载完成，GPU内存: {get_gpu_memory():.1f} MB")
    
    # 创建推理引擎
    engine = KVCacheInferenceEngine(
        model, pt_tokenizer, en_tokenizer, device, use_mla, config
    )
    
    # 执行生成
    result = engine.generate_with_kv_cache(test_input, max_new_tokens, use_real_cache=use_real_cache)
    
    # 打印关键结果
    logger.info(f"\n生成: {result['num_tokens_generated']} tokens | 时间: {result['generation_time']:.3f}s | KV-cache: {result['max_kv_cache_size']:.2f} MB")
    
    return {
        'model_name': model_name,
        'generation_time': result['generation_time'],
        'max_kv_cache_size': result['max_kv_cache_size'],
        'output': result['output'],
        'kv_cache_sizes': result['kv_cache_sizes'],
        'step_times': result['step_times']
    }


def compare_results(mla_result: Dict, standard_result: Dict):
    """比较两个模型的结果"""
    logger.info(f"\n{'='*60}")
    logger.info("📊 MLA vs Standard 对比结果")
    logger.info(f"{'='*60}")
    
    # KV-cache显存对比
    kv_diff = standard_result['max_kv_cache_size'] - mla_result['max_kv_cache_size']
    kv_improvement = (kv_diff / standard_result['max_kv_cache_size'] * 100) if standard_result['max_kv_cache_size'] > 0 else 0
    
    # 速度对比
    time_diff = standard_result['generation_time'] - mla_result['generation_time']
    time_improvement = (time_diff / standard_result['generation_time'] * 100) if standard_result['generation_time'] > 0 else 0
    
    logger.info(f"\n💾 KV-cache: {mla_result['max_kv_cache_size']:.2f} MB vs {standard_result['max_kv_cache_size']:.2f} MB | 节省 {kv_improvement:.1f}%")
    logger.info(f"⏱️  推理时间: {mla_result['generation_time']:.3f}s vs {standard_result['generation_time']:.3f}s | 慢 {abs(time_improvement):.1f}%")
    logger.info(f"\n✅ MLA节省 {kv_improvement:.1f}% KV-cache显存，但推理慢 {abs(time_improvement):.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MLA vs 标准注意力 KV-cache 效率对比")
    parser.add_argument("--mla_checkpoint", type=str, default="checkpoints/latest.pt",
                       help="MLA模型checkpoint路径")
    parser.add_argument("--no_mla_checkpoint", type=str, default="checkpoints_no_mla/latest.pt",
                       help="非MLA模型checkpoint路径")
    parser.add_argument("--test_input", type=str, 
                       default="O Tom está procurando uma segunda opinião sobre o tratamento médico.",
                       help="测试输入文本")
    parser.add_argument("--test_lengths", nargs='+', type=int, default=[32, 64, 128],
                       help="测试的生成长度列表")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 检查checkpoint
    if not os.path.exists(args.mla_checkpoint):
        logger.error(f"❌ MLA checkpoint not found: {args.mla_checkpoint}")
        return
    
    # 判断测试模式
    test_both_models = os.path.exists(args.no_mla_checkpoint)
    
    if test_both_models:
        logger.info(f"\n🚀 模式：MLA vs Standard 对比")
    else:
        logger.info(f"\n🚀 模式：真正KV-cache vs 模拟KV-cache 对比")
    
    logger.info(f"{'='*60}")
    
    all_results = []
    
    for max_new_tokens in args.test_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"测试: {max_new_tokens} tokens")
        logger.info(f"{'='*60}")
        
        if test_both_models:
            # 测试MLA模型
            mla_result = benchmark_model(
                args.mla_checkpoint, True, args.test_input, max_new_tokens, device
            )
            all_results.append(mla_result)
            
            # 清理内存
            clear_memory()
            time.sleep(1)
            
            # 测试标准模型
            standard_result = benchmark_model(
                args.no_mla_checkpoint, False, args.test_input, max_new_tokens, device
            )
            all_results.append(standard_result)
            
            # 对比结果
            compare_results(mla_result, standard_result)
            
            # 清理内存
            clear_memory()
            time.sleep(1)
        else:
            # 只有一个模型：对比真正cache vs 模拟cache
            logger.info("\n[1/2] 测试真正的KV-cache")
            real_cache_result = benchmark_model(
                args.mla_checkpoint, True, args.test_input, max_new_tokens, device, use_real_cache=True
            )
            all_results.append(real_cache_result)
            
            # 清理内存
            clear_memory()
            time.sleep(1)
            
            logger.info("\n[2/2] 测试模拟KV-cache（每步重算）")
            simulated_cache_result = benchmark_model(
                args.mla_checkpoint, True, args.test_input, max_new_tokens, device, use_real_cache=False
            )
            all_results.append(simulated_cache_result)
            
            # 对比真正cache vs 模拟cache
            logger.info(f"\n{'='*60}")
            logger.info("📊 真正Cache vs 不用Cache 对比结果")
            logger.info(f"{'='*60}")
            
            speedup = simulated_cache_result['generation_time'] / real_cache_result['generation_time']
            
            logger.info(f"\n🔍 关键区别：")
            logger.info(f"  真正Cache: Encoder运行1次 | Decoder每步输入1个token | 复用历史K/V")
            logger.info(f"  不用Cache: Encoder每步运行 | Decoder每步输入完整序列 | 重新计算所有K/V")
            logger.info(f"\n📈 性能对比：")
            logger.info(f"  ⚡ 加速比: {speedup:.2f}x")
            logger.info(f"  💾 KV-cache显存: {real_cache_result['max_kv_cache_size']:.2f} MB")
            logger.info(f"  ⏱️  真正Cache: {real_cache_result['generation_time']:.3f}s")
            logger.info(f"  🐌 不用Cache: {simulated_cache_result['generation_time']:.3f}s")
            logger.info(f"\n✅ 真正的KV-cache通过缓存历史K/V，避免重复计算，实现 {speedup:.2f}倍加速！")
            logger.info(f"   计算量: O(n) vs O(n²)  其中n={max_new_tokens}")
            
            # 对比生成的token差异
            logger.info(f"\n{'='*60}")
            logger.info("🔤 Token输出对比")
            logger.info(f"{'='*60}")
            
            real_tokens = real_cache_result.get('generated_tokens', [])
            sim_tokens = simulated_cache_result.get('generated_tokens', [])
            real_texts = real_cache_result.get('generated_texts', [])
            sim_texts = simulated_cache_result.get('generated_texts', [])
            
            # 检查输出是否一致
            if real_tokens == sim_tokens:
                logger.info("✅ 两种方法生成的token序列完全一致！")
                logger.info(f"📝 最终输出: {real_cache_result['output']}")
            else:
                logger.warning("⚠️ 两种方法生成的token序列存在差异！")
                logger.info(f"\n真正Cache输出 ({len(real_tokens)} tokens): {real_cache_result['output']}")
                logger.info(f"模拟Cache输出 ({len(sim_tokens)} tokens): {simulated_cache_result['output']}")
                
                # 详细对比差异
                logger.info(f"\n逐token对比：")
                max_len = max(len(real_tokens), len(sim_tokens))
                diff_count = 0
                for i in range(max_len):
                    real_token = real_tokens[i] if i < len(real_tokens) else None
                    sim_token = sim_tokens[i] if i < len(sim_tokens) else None
                    real_text = real_texts[i] if i < len(real_texts) else ""
                    sim_text = sim_texts[i] if i < len(sim_texts) else ""
                    
                    if real_token != sim_token:
                        diff_count += 1
                        status = "❌"
                    else:
                        status = "✅"
                    
                    logger.info(f"  {status} Token {i+1}: 真实={real_token}('{real_text}') | 模拟={sim_token}('{sim_text}')")
                
                logger.info(f"\n差异统计: {diff_count}/{max_len} tokens不同")
            
            # 清理内存
            clear_memory()
            time.sleep(1)
    
    # 保存结果
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'test_input': args.test_input,
            'test_lengths': args.test_lengths,
            'results': all_results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ 结果已保存到: {args.output}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ 测试完成")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()


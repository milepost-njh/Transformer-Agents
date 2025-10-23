# -*- coding: utf-8 -*-
"""
Transformer 推理脚本
支持普通推理模式、MLA推理模式、MTP推理模式

使用方法:
CUDA_VISIBLE_DEVICES=5 python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá, como você está?"

CUDA_VISIBLE_DEVICES=5 python inference.py --mode mla --checkpoint checkpoints/latest.pt --input "Olá, como você estä
CUDA_VISIBLE_DEVICES=5 python inference.py --mode mtp --checkpoint checkpoints/latest.pt --input "Olá, como você está?"
CUDA_VISIBLE_DEVICES=5 python inference.py --mode all --checkpoint checkpoints/latest.pt --input "Olá, como você estä
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
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
from core.models.modeling_deepseek import DeepseekV3MoE
from core.normalization import RMSNorm, LayerNorm
from core.models.deepseek_mtp import DeepSeekMTPConfig, add_mtp_to_transformer

# 从训练脚本导入必要的类和函数
from train_tmp import (
    Transformer, MoEConfig, MultiHeadAttention, EncoderLayer, DecoderLayer,
    EncoderModel, DecoderModel, create_masks, get_device, check_env
)


class InferenceConfig:
    """推理配置类"""
    def __init__(
        self,
        mode: str = "normal",  # normal, mla, mtp, all
        checkpoint_path: str = None,
        device: str = None,
        max_length: int = 64,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_beams: int = 1,
        do_sample: bool = True,
        early_stopping: bool = True,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        bos_token_id: int = 0,
        vocab_size: int = 8192,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        dff: int = 2048,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
        use_moe: bool = False,
        use_mla: bool = False,
        use_mtp: bool = False,
        moe_config: Optional[MoEConfig] = None,
        mtp_config: Optional[DeepSeekMTPConfig] = None,
        tokenizer_pt_path: str = "tok_pt",
        tokenizer_en_path: str = "tok_en",
        decode_method: str = "greedy",
    ):
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.device = device or get_device()
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.use_rope = use_rope
        self.use_moe = use_moe
        self.use_mla = use_mla
        self.use_mtp = use_mtp
        self.moe_config = moe_config
        self.mtp_config = mtp_config
        self.tokenizer_pt_path = tokenizer_pt_path
        self.tokenizer_en_path = tokenizer_en_path
        self.decode_method = decode_method


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.pt_tokenizer = None
        self.en_tokenizer = None
        self.model = None
        
    def load_tokenizers(self):
        """加载tokenizer"""
        try:
            # 加载葡萄牙语tokenizer
            pt_tokenizer_path = os.path.join(self.config.tokenizer_pt_path, "tokenizer.json")
            if not os.path.exists(pt_tokenizer_path):
                raise FileNotFoundError(f"Portuguese tokenizer not found at {pt_tokenizer_path}")
            
            self.pt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=pt_tokenizer_path)
            self.pt_tokenizer.pad_token = "<pad>"
            self.pt_tokenizer.unk_token = "<unk>"
            self.pt_tokenizer.bos_token = "<s>"
            self.pt_tokenizer.eos_token = "</s>"
            self.pt_tokenizer.mask_token = "<mask>"
            self.pt_tokenizer.model_max_length = self.config.max_length
            self.pt_tokenizer.padding_side = "right"
            
            # 加载英语tokenizer
            en_tokenizer_path = os.path.join(self.config.tokenizer_en_path, "tokenizer.json")
            if not os.path.exists(en_tokenizer_path):
                raise FileNotFoundError(f"English tokenizer not found at {en_tokenizer_path}")
            
            self.en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=en_tokenizer_path)
            self.en_tokenizer.pad_token = "<pad>"
            self.en_tokenizer.unk_token = "<unk>"
            self.en_tokenizer.bos_token = "<s>"
            self.en_tokenizer.eos_token = "</s>"
            self.en_tokenizer.mask_token = "<mask>"
            self.en_tokenizer.model_max_length = self.config.max_length
            self.en_tokenizer.padding_side = "right"
            
            logger.info(f"✅ Tokenizers loaded successfully")
            logger.info(f"   PT vocab size: {len(self.pt_tokenizer)}")
            logger.info(f"   EN vocab size: {len(self.en_tokenizer)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizers: {e}")
            raise
    
    def create_model(self):
        """创建模型"""
        try:
            # 根据配置创建模型
            model = Transformer(
                num_layers=self.config.num_layers,
                input_vocab_size=len(self.pt_tokenizer),
                target_vocab_size=len(self.en_tokenizer),
                max_length=self.config.max_length,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                rate=self.config.dropout_rate,
                src_padding_idx=self.pt_tokenizer.pad_token_id,
                tgt_padding_idx=self.en_tokenizer.pad_token_id,
                use_rope=self.config.use_rope,
                use_moe=self.config.use_moe,
                moe_config=self.config.moe_config,
                use_mla=self.config.use_mla,
                q_lora_rank=self.config.d_model // 2 if self.config.use_mla else None,
                kv_lora_rank=self.config.d_model // 4 if self.config.use_mla else None,
            )
            
            # 如果使用MTP，添加MTP功能
            if self.config.use_mtp and self.config.mtp_config:
                model = add_mtp_to_transformer(model, self.config.mtp_config)
                logger.info("✅ MTP module added to model")
            
            self.model = model
            logger.info(f"✅ Model created successfully")
            logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            raise
    
    def load_checkpoint(self):
        """加载检查点"""
        if not self.config.checkpoint_path or not os.path.exists(self.config.checkpoint_path):
            logger.warning("⚠️ No checkpoint provided or checkpoint not found, using random weights")
            return
        
        try:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.config.device)
            
            # 加载模型权重
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                # 处理DataParallel包装的模型
                if any(key.startswith('module.') for key in model_state.keys()):
                    # 移除'module.'前缀
                    model_state = {k[7:] if k.startswith('module.') else k: v for k, v in model_state.items()}
                
                self.model.load_state_dict(model_state, strict=False)
                logger.info(f"✅ Model weights loaded from {self.config.checkpoint_path}")
                
                # 打印检查点信息
                if 'epoch' in checkpoint:
                    logger.info(f"   Checkpoint epoch: {checkpoint['epoch']}")
                if 'step' in checkpoint:
                    logger.info(f"   Checkpoint step: {checkpoint['step']}")
            else:
                logger.warning("⚠️ No 'model' key found in checkpoint")
                
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def setup_model(self):
        """设置模型用于推理"""
        self.model.eval()
        self.model.to(self.config.device)
        
        # 禁用dropout和batch normalization的training模式
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
        
        logger.info(f"✅ Model setup for inference on {self.config.device}")


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.loader = ModelLoader(config)
        self.results = {}
        
    def initialize(self):
        """初始化推理引擎"""
        logger.info("🚀 Initializing inference engine...")
        
        # 加载tokenizers
        self.loader.load_tokenizers()
        
        # 创建模型
        self.loader.create_model()
        
        # 加载检查点
        self.loader.load_checkpoint()
        
        # 设置模型
        self.loader.setup_model()
        
        logger.info("✅ Inference engine initialized successfully")
    
    def encode_input(self, text: str) -> torch.Tensor:
        """编码输入文本"""
        def encode_with_bos_eos(tokenizer, text: str):
            ids = tokenizer.encode(text, add_special_tokens=False)
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            if bos_id is None or eos_id is None:
                raise ValueError("Tokenizer must have bos_token and eos_token")
            return [bos_id] + ids + [eos_id]
        
        inp_ids = encode_with_bos_eos(self.loader.pt_tokenizer, text)
        encoder_input = torch.tensor(inp_ids, dtype=torch.long, device=self.config.device).unsqueeze(0)
        return encoder_input
    
    def decode_output(self, token_ids: List[int]) -> str:
        """解码输出token IDs"""
        return self.loader.en_tokenizer.decode(
            [i for i in token_ids if i < len(self.loader.en_tokenizer)],
            skip_special_tokens=True
        )
    
    def greedy_decode(self, encoder_input: torch.Tensor, max_length: int = None) -> Tuple[List[int], Dict]:
        """贪心解码"""
        max_length = max_length or self.config.max_length
        device = encoder_input.device
        
        # 初始化decoder输入
        start_id = self.loader.en_tokenizer.bos_token_id
        end_id = self.loader.en_tokenizer.eos_token_id
        decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=device)
        
        attention_weights = {}
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                # 创建masks
                enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
                    encoder_input, decoder_input,
                    src_pad_id=self.loader.pt_tokenizer.pad_token_id,
                    tgt_pad_id=self.loader.en_tokenizer.pad_token_id,
                )
                enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                
                # 前向传播
                model_output = self.loader.model(
                    encoder_input, decoder_input,
                    src_mask=enc_pad_mask,
                    tgt_mask=dec_mask,
                    enc_dec_mask=enc_dec_mask,
                )
                
                # 处理不同模式的输出
                if isinstance(model_output, tuple) and len(model_output) == 4:
                    # MTP模式
                    logits, attn, router_logits, mtp_logits = model_output
                elif isinstance(model_output, tuple) and len(model_output) == 3:
                    # MoE模式
                    logits, attn, router_logits = model_output
                else:
                    # 标准模式
                    logits, attn = model_output
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                
                # 应用temperature
                if self.config.temperature != 1.0:
                    next_token_logits = next_token_logits / self.config.temperature
                
                # 贪心选择
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                
                # 检查是否结束
                if next_token_id.item() == end_id:
                    break
                
                # 添加到生成序列
                generated_tokens.append(next_token_id.item())
                decoder_input = torch.cat([decoder_input, next_token_id.unsqueeze(0)], dim=-1)
                attention_weights = attn
        
        return generated_tokens, attention_weights
    
    def sample_decode(self, encoder_input: torch.Tensor, max_length: int = None) -> Tuple[List[int], Dict]:
        """采样解码"""
        max_length = max_length or self.config.max_length
        device = encoder_input.device
        
        # 初始化decoder输入
        start_id = self.loader.en_tokenizer.bos_token_id
        end_id = self.loader.en_tokenizer.eos_token_id
        decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=device)
        
        attention_weights = {}
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                # 创建masks
                enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
                    encoder_input, decoder_input,
                    src_pad_id=self.loader.pt_tokenizer.pad_token_id,
                    tgt_pad_id=self.loader.en_tokenizer.pad_token_id,
                )
                enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                
                # 前向传播
                model_output = self.loader.model(
                    encoder_input, decoder_input,
                    src_mask=enc_pad_mask,
                    tgt_mask=dec_mask,
                    enc_dec_mask=enc_dec_mask,
                )
                
                # 处理不同模式的输出
                if isinstance(model_output, tuple) and len(model_output) == 4:
                    # MTP模式
                    logits, attn, router_logits, mtp_logits = model_output
                elif isinstance(model_output, tuple) and len(model_output) == 3:
                    # MoE模式
                    logits, attn, router_logits = model_output
                else:
                    # 标准模式
                    logits, attn = model_output
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                
                # 应用temperature
                if self.config.temperature != 1.0:
                    next_token_logits = next_token_logits / self.config.temperature
                
                # Top-k过滤
                if self.config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, self.config.top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p过滤
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                # 检查是否结束
                if next_token_id.item() == end_id:
                    break
                
                # 添加到生成序列
                generated_tokens.append(next_token_id.item())
                decoder_input = torch.cat([decoder_input, next_token_id], dim=-1)
                attention_weights = attn
        
        return generated_tokens, attention_weights
    
    def beam_search_decode(self, encoder_input: torch.Tensor, max_length: int = None) -> List[Tuple[List[int], float]]:
        """束搜索解码"""
        max_length = max_length or self.config.max_length
        device = encoder_input.device
        num_beams = self.config.num_beams
        
        # 初始化
        start_id = self.loader.en_tokenizer.bos_token_id
        end_id = self.loader.en_tokenizer.eos_token_id
        
        # 初始beam
        beams = [(torch.tensor([[start_id]], dtype=torch.long, device=device), 0.0)]
        completed_beams = []
        
        with torch.no_grad():
            for step in range(max_length):
                new_beams = []
                
                for decoder_input, score in beams:
                    # 创建masks
                    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
                        encoder_input, decoder_input,
                        src_pad_id=self.loader.pt_tokenizer.pad_token_id,
                        tgt_pad_id=self.loader.en_tokenizer.pad_token_id,
                    )
                    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                    
                    # 前向传播
                    model_output = self.loader.model(
                        encoder_input, decoder_input,
                        src_mask=enc_pad_mask,
                        tgt_mask=dec_mask,
                        enc_dec_mask=enc_dec_mask,
                    )
                    
                    # 处理不同模式的输出
                    if isinstance(model_output, tuple) and len(model_output) == 4:
                        # MTP模式
                        logits, attn, router_logits, mtp_logits = model_output
                    elif isinstance(model_output, tuple) and len(model_output) == 3:
                        # MoE模式
                        logits, attn, router_logits = model_output
                    else:
                        # 标准模式
                        logits, attn = model_output
                    
                    # 取最后一个token的logits
                    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                    
                    # 应用temperature
                    if self.config.temperature != 1.0:
                        next_token_logits = next_token_logits / self.config.temperature
                    
                    # 获取top-k候选
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, num_beams)
                    
                    for i in range(num_beams):
                        token_id = top_k_indices[0, i].item()
                        token_score = top_k_logits[0, i].item()
                        new_score = score + token_score
                        
                        if token_id == end_id:
                            # 完成一个beam
                            completed_beams.append((decoder_input.squeeze(0).tolist()[1:], new_score))
                        else:
                            # 继续beam
                            new_decoder_input = torch.cat([decoder_input, top_k_indices[0, i:i+1].unsqueeze(0)], dim=-1)
                            new_beams.append((new_decoder_input, new_score))
                
                # 选择top beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]
                
                if not beams:
                    break
        
        # 添加未完成的beams
        for decoder_input, score in beams:
            completed_beams.append((decoder_input.squeeze(0).tolist()[1:], score))
        
        return completed_beams
    
    def infer_single(self, input_text: str, decode_method: str = "greedy") -> Dict:
        """单句推理"""
        start_time = time.time()
        
        # 编码输入
        encoder_input = self.encode_input(input_text)
        
        # 解码
        if decode_method == "greedy":
            generated_tokens, attention_weights = self.greedy_decode(encoder_input)
            result = {
                "input": input_text,
                "output": self.decode_output(generated_tokens),
                "tokens": generated_tokens,
                "method": "greedy",
                "attention_weights": attention_weights,
                "inference_time": time.time() - start_time
            }
        elif decode_method == "sample":
            generated_tokens, attention_weights = self.sample_decode(encoder_input)
            result = {
                "input": input_text,
                "output": self.decode_output(generated_tokens),
                "tokens": generated_tokens,
                "method": "sample",
                "attention_weights": attention_weights,
                "inference_time": time.time() - start_time
            }
        elif decode_method == "beam":
            beam_results = self.beam_search_decode(encoder_input)
            result = {
                "input": input_text,
                "outputs": [self.decode_output(tokens) for tokens, score in beam_results],
                "beam_results": beam_results,
                "method": "beam_search",
                "inference_time": time.time() - start_time
            }
        else:
            raise ValueError(f"Unknown decode method: {decode_method}")
        
        return result
    
    def infer_batch(self, input_texts: List[str], decode_method: str = "greedy") -> List[Dict]:
        """批量推理"""
        results = []
        for text in input_texts:
            result = self.infer_single(text, decode_method)
            results.append(result)
        return results
    
    def compare_modes(self, input_text: str) -> Dict:
        """比较不同推理模式"""
        modes = []
        if self.config.mode == "all":
            modes = ["normal", "mla", "mtp"]
        else:
            modes = [self.config.mode]
        
        comparison_results = {}
        
        for mode in modes:
            # 临时修改配置
            original_mode = self.config.mode
            original_use_mla = self.config.use_mla
            original_use_mtp = self.config.use_mtp
            
            if mode == "normal":
                self.config.use_mla = False
                self.config.use_mtp = False
            elif mode == "mla":
                self.config.use_mla = True
                self.config.use_mtp = False
            elif mode == "mtp":
                self.config.use_mla = False
                self.config.use_mtp = True
            
            # 重新初始化模型（如果需要）
            if mode != original_mode:
                self.loader.create_model()
                self.loader.load_checkpoint()
                self.loader.setup_model()
            
            # 推理
            result = self.infer_single(input_text, "greedy")
            comparison_results[mode] = result
            
            # 恢复原始配置
            self.config.mode = original_mode
            self.config.use_mla = original_use_mla
            self.config.use_mtp = original_use_mtp
        
        return comparison_results


def create_moe_config(config: InferenceConfig) -> MoEConfig:
    """创建MoE配置"""
    return MoEConfig(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=config.d_model,
        intermediate_size=config.dff,
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
        moe_intermediate_size=config.dff,
    )


def create_mtp_config(config: InferenceConfig) -> DeepSeekMTPConfig:
    """创建MTP配置"""
    return DeepSeekMTPConfig(
        hidden_size=config.d_model,
        num_nextn_predict_layers=2,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_length,
        use_moe=config.use_moe,
        moe_config=config.moe_config,
        mtp_loss_weight=0.1,
    )


def main():
    parser = argparse.ArgumentParser(description="Transformer Inference Script")
    parser.add_argument("--mode", type=str, default="normal", 
                       choices=["normal", "mla", "mtp", "all"],
                       help="Inference mode: normal, mla, mtp, or all")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Input text to translate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (optional)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--max_length", type=int, default=64,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--decode_method", type=str, default="greedy",
                       choices=["greedy", "sample", "beam"],
                       help="Decoding method")
    parser.add_argument("--batch", action="store_true",
                       help="Process input as batch (one sentence per line)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare different modes")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # 创建配置
    config = InferenceConfig(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        decode_method=args.decode_method,
    )
    
    # 根据模式设置配置
    if args.mode in ["mla", "all"]:
        config.use_mla = True
    if args.mode in ["mtp", "all"]:
        config.use_mtp = True
        config.moe_config = create_moe_config(config)
        config.mtp_config = create_mtp_config(config)
    if args.mode == "normal":
        config.use_mla = False
        config.use_mtp = False
    
    # 创建推理引擎
    engine = InferenceEngine(config)
    
    try:
        # 初始化
        engine.initialize()
        
        # 处理输入
        if args.batch:
            # 批量处理
            with open(args.input, 'r', encoding='utf-8') as f:
                input_texts = [line.strip() for line in f if line.strip()]
            
            results = engine.infer_batch(input_texts, args.decode_method)
            
            # 输出结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"Input: {result['input']}\n")
                        f.write(f"Output: {result['output']}\n")
                        f.write(f"Time: {result['inference_time']:.3f}s\n\n")
            else:
                for result in results:
                    print(f"Input: {result['input']}")
                    print(f"Output: {result['output']}")
                    print(f"Time: {result['inference_time']:.3f}s\n")
        
        elif args.compare:
            # 比较模式
            comparison_results = engine.compare_modes(args.input)
            
            # 输出比较结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Input: {args.input}\n\n")
                    for mode, result in comparison_results.items():
                        f.write(f"Mode: {mode.upper()}\n")
                        f.write(f"Output: {result['output']}\n")
                        f.write(f"Time: {result['inference_time']:.3f}s\n\n")
            else:
                print(f"Input: {args.input}\n")
                for mode, result in comparison_results.items():
                    print(f"Mode: {mode.upper()}")
                    print(f"Output: {result['output']}")
                    print(f"Time: {result['inference_time']:.3f}s\n")
        
        else:
            # 单句推理
            result = engine.infer_single(args.input, args.decode_method)
            
            # 输出结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Input: {result['input']}\n")
                    f.write(f"Output: {result['output']}\n")
                    f.write(f"Method: {result['method']}\n")
                    f.write(f"Time: {result['inference_time']:.3f}s\n")
            else:
                print(f"Input: {result['input']}")
                print(f"Output: {result['output']}")
                print(f"Method: {result['method']}")
                print(f"Time: {result['inference_time']:.3f}s")
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

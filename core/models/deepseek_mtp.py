# -*- coding: utf-8 -*-
"""
DeepSeek MTP (Multi-Token Prediction) 模型实现
作为 Transformer 的辅助预测模块，用于多 token 预测和推测解码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from loguru import logger

from core.normalization import RMSNorm
from core.models.modeling_deepseek import DeepseekV3MoE


class DeepSeekMTPConfig:
    """MTP 配置类"""
    def __init__(
        self,
        hidden_size: int = 512,
        num_nextn_predict_layers: int = 2,
        vocab_size: int = 8192,
        max_position_embeddings: int = 1024,
        rms_norm_eps: float = 1e-6,
        use_moe: bool = False,
        moe_config: Optional[Any] = None,
        mtp_loss_weight: float = 0.1,  # MTP 损失权重
    ):
        self.hidden_size = hidden_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_config = moe_config
        self.mtp_loss_weight = mtp_loss_weight


class DeepSeekMTPLayer(nn.Module):
    """MTP 单层实现 - 用于多 token 预测"""
    def __init__(self, config: DeepSeekMTPConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 输入归一化
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 特征融合投影
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # 简化的处理层
        if config.use_moe and config.moe_config is not None:
            self.mtp_block = DeepseekV3MoE(config.moe_config)
        else:
            # 简单的 FFN
            self.mtp_block = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size)
            )
        
        # 每个MTP层只有一个预测头
        self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        返回:
            hidden_states: 处理后的隐藏状态
            mtp_logits: 多 token 预测的 logits 列表
        """
        # 掩码位置 0 的输入（MTP 不需要）
        inputs_embeds = inputs_embeds.clone()
        if positions is not None:
             inputs_embeds[positions == 0] = 0
        
        # 归一化
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        
        # 特征融合
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        
        # 通过 MTP 块
        if self.config.use_moe and self.config.moe_config is not None:
            mtp_output = self.mtp_block(hidden_states)
            if isinstance(mtp_output, tuple):
                hidden_states, _ = mtp_output
            else:
                hidden_states = mtp_output
        else:
            hidden_states = self.mtp_block(hidden_states)
        
        # 每个MTP层生成一个logits
        logits = self.mtp_head(hidden_states)
        
        return hidden_states, [logits]  # 包装成列表保持接口一致


class DeepSeekMTPPredictor(nn.Module):
    """MTP 预测器 - 作为 Transformer 的辅助预测模块"""
    def __init__(self, config: DeepSeekMTPConfig):
        super().__init__()
        self.config = config
        self.num_mtp_layers = config.num_nextn_predict_layers
        
        # MTP 层
        self.layers = nn.ModuleList([
            DeepSeekMTPLayer(config, layer_idx=i)
            for i in range(self.num_mtp_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        返回:
            hidden_states: 处理后的隐藏状态
            all_mtp_logits: 所有层的 MTP 预测 logits
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        
        hidden_states, mtp_logits = mtp_layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )
        
        return hidden_states, mtp_logits


class DeepSeekMTP(nn.Module):
    """DeepSeek MTP 模块 - 作为 Transformer 的辅助预测模块"""
    def __init__(self, config: DeepSeekMTPConfig):
        super().__init__()
        self.config = config
        self.predictor = DeepSeekMTPPredictor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        返回:
            hidden_states: 处理后的隐藏状态
            mtp_logits: MTP 预测的 logits 列表
        """
        hidden_states, mtp_logits = self.predictor(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states, mtp_logits


class DeepSeekMTPWrapper(nn.Module):
    """MTP 包装器 - 为现有 Transformer 添加 MTP 功能"""
    def __init__(self, base_transformer, mtp_config: DeepSeekMTPConfig):
        super().__init__()
        self.base_transformer = base_transformer
        self.mtp_module = DeepSeekMTP(mtp_config)
        self.mtp_config = mtp_config
        
        # 共享 Embedding：直接使用主模型的 embedding
        # 这样可以确保 MTP 的输入特征与主模型一致，且被正确训练
        self.embed_tokens = base_transformer.decoder_model.embedding
        
    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None, 
                past_key_values=None, use_cache=False, encoder_outputs=None):
        """
        增强的 forward 方法，支持 MTP 多 token 预测
        返回:
            logits: 主模型的 logits
            attention_weights: 注意力权重
            router_logits: MoE 路由 logits
            mtp_logits: MTP 预测的 logits 列表（如果启用）
        """
        # 主模型前向传播
        # 注意：这里假设 Transformer 已修改为返回 hidden_states
        transformer_output = self.base_transformer(
            inp_ids, tgt_ids, src_mask, tgt_mask, enc_dec_mask,
            past_key_values, use_cache, encoder_outputs
        )
        
        # 解析输出
        logits = None
        attention_weights = None
        router_logits = None
        hidden_states = None
        present_key_values = None
        
        # 根据 Transformer 返回值的长度进行解析
        if use_cache:
            # 训练时通常不使用 cache，这里主要处理推理场景
            if len(transformer_output) == 4:
                logits, attention_weights, present_key_values, hidden_states = transformer_output
            elif len(transformer_output) == 5:
                logits, attention_weights, present_key_values, router_logits, hidden_states = transformer_output
        else:
            # 训练场景
            if len(transformer_output) == 3:
                logits, attention_weights, hidden_states = transformer_output
            elif len(transformer_output) == 4:
                logits, attention_weights, router_logits, hidden_states = transformer_output
            else:
                # 兼容旧接口（虽然我们已经修改了 Transformer）
                logger.warning(f"Unexpected transformer output length: {len(transformer_output)}")
                if len(transformer_output) == 2:
                    logits, attention_weights = transformer_output
                elif len(transformer_output) == 3 and isinstance(transformer_output[2], list): # 假设第3个是router_logits
                    logits, attention_weights, router_logits = transformer_output
        
        # 如果启用 MTP，进行多 token 预测
        mtp_logits = None
        if hidden_states is not None and hasattr(self, 'mtp_module') and self.mtp_module is not None:
            batch_size, seq_len, _ = hidden_states.shape
            
            # 1. 准备输入 Embedding
            # MTP 需要下一个 token 的 embedding 作为输入的一部分
            # 对于训练：tgt_ids 是 decoder 的输入（已 shift），我们需要预测下一个 token
            # 这里我们使用 tgt_ids 对应的 embedding 作为 inputs_embeds
            # 注意：MTP 的设计通常是利用当前 hidden state 预测下一个 token，
            # 并利用下一个 token 的 embedding 和当前 hidden state 预测再下一个 token
            
            # 获取目标 embedding
            inputs_embeds = self.embed_tokens(tgt_ids) * self.base_transformer.decoder_model.scale
            
            # 2. 准备位置信息
            positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            
            # 3. 全序列 MTP 预测 (Whole Sequence Training)
            # 我们对序列中的每个位置都进行预测，而不仅仅是最后一个位置
            
            # 递归预测未来 K 个 token
            all_mtp_logits = []
            current_hidden_states = hidden_states
            
            # 循环预测每一层
            for i in range(self.mtp_config.num_nextn_predict_layers):
                try:
                    # 第一层 MTP 预测
                    # 输入: h_t, emb_t
                    # 输出: h'_{t+1}, logits_{t+1} (预测 x_{t+2})
                    
                    current_hidden_states, layer_mtp_logits = self.mtp_module(
                        input_ids=None,
                        positions=positions,
                        hidden_states=current_hidden_states,
                        inputs_embeds=inputs_embeds, # 使用当前输入的 embedding
                        spec_step_idx=i
                    )
                    
                    # layer_mtp_logits 是一个 list
                    if isinstance(layer_mtp_logits, list):
                        all_mtp_logits.extend(layer_mtp_logits)
                    else:
                        all_mtp_logits.append(layer_mtp_logits)
                    
                except Exception as e:
                    logger.error(f"MTP layer {i} forward error: {e}")
                    break
            
            mtp_logits = all_mtp_logits
            
            # 只在第一次执行时打印日志
            if not hasattr(DeepSeekMTPWrapper, '_first_forward_logged'):
                logger.info(f"✅ MTP全序列训练已修复: 输入形状={hidden_states.shape}, 预测层数={len(mtp_logits)}")
                DeepSeekMTPWrapper._first_forward_logged = True
        
        # 返回结果
        if use_cache:
             # 推理模式返回
            if mtp_logits is not None:
                 return logits, attention_weights, present_key_values, router_logits, mtp_logits
            elif router_logits is not None:
                 return logits, attention_weights, present_key_values, router_logits
            else:
                 return logits, attention_weights, present_key_values
        else:
            # 训练模式返回
            if mtp_logits is not None:
                return logits, attention_weights, router_logits, mtp_logits
            elif router_logits is not None:
                return logits, attention_weights, router_logits
            else:
                return logits, attention_weights


def compute_mtp_loss(mtp_logits_list: List[torch.Tensor], target_ids: torch.Tensor, mtp_loss_weight: float = 0.1) -> torch.Tensor:
    """
    计算 MTP 多 token 预测损失
    
    Args:
        mtp_logits_list: MTP 预测的 logits 列表，每个元素对应一个预测步骤
        target_ids: 目标 token IDs [batch_size, seq_len]
        mtp_loss_weight: MTP 损失权重
    
    Returns:
        MTP 损失值
    """
    if not mtp_logits_list or len(mtp_logits_list) == 0:
        return torch.tensor(0.0, device=target_ids.device)
    
    total_mtp_loss = 0.0
    num_predictions = len(mtp_logits_list)
    
    # target_ids 是主任务的 labels，即 x_{t+1}
    # MTP 第 1 层预测的是 x_{t+2}，所以需要 shift target_ids
    
    batch_size, seq_len = target_ids.shape
    
    for i, mtp_logits in enumerate(mtp_logits_list):
        # mtp_logits: [batch_size, seq_len, vocab_size]
        # 第 i 层 MTP 预测的是未来第 i+1 个 token (相对于主任务预测的 next token)
        # 主任务: h_t -> x_{t+1}
        # MTP_0:  h_t -> x_{t+2} (这里为了简化，假设 MTP_0 预测 +2 步)
        
        # 实际上，我们的实现中 MTP_0 是基于 h_t 预测 x_{t+1} 的辅助任务 (类似 Deep Seek V3 的 MTP 模块作为补充)
        # 或者 MTP_0 预测 x_{t+2}
        
        # 假设 MTP_k 预测 x_{t+k+2}
        shift = i + 1
        
        # 截取有效的 logits 和 targets
        # logits: [0, ..., L-1]
        # targets: [0, ..., L-1] (即 x_1, ..., x_L)
        
        # 我们需要预测 x_{t+1+shift}
        # valid length = seq_len - shift
        
        if seq_len > shift:
            valid_logits = mtp_logits[:, :-shift, :] # [B, L-shift, V]
            valid_targets = target_ids[:, shift:]     # [B, L-shift]
            
            loss = F.cross_entropy(
                valid_logits.reshape(-1, valid_logits.size(-1)),
                valid_targets.reshape(-1),
                reduction='mean'
            )
            total_mtp_loss += loss
    
    # 平均损失并应用权重
    avg_mtp_loss = total_mtp_loss / num_predictions if num_predictions > 0 else 0.0
    final_mtp_loss = avg_mtp_loss * mtp_loss_weight
    
    return final_mtp_loss


def add_mtp_to_transformer(transformer, mtp_config: DeepSeekMTPConfig):
    """
    为现有的 Transformer 添加 MTP 功能
    
    Args:
        transformer: 现有的 Transformer 模型
        mtp_config: MTP 配置
    
    Returns:
        增强后的模型，支持 MTP 多 token 预测
    """
    return DeepSeekMTPWrapper(transformer, mtp_config)

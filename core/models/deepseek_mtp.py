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
        
        # 多 token 预测头
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_nextn_predict_layers)
        ])

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
        
        # 生成多 token 预测 logits
        mtp_logits = []
        for head in self.mtp_heads:
            logits = head(hidden_states)
            mtp_logits.append(logits)
        
        return hidden_states, mtp_logits


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
        
    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        """
        增强的 forward 方法，支持 MTP 多 token 预测
        返回:
            logits: 主模型的 logits
            attention_weights: 注意力权重
            mtp_logits: MTP 预测的 logits 列表（如果启用）
        """
        # 主模型前向传播
        transformer_output = self.base_transformer(inp_ids, tgt_ids, src_mask, tgt_mask, enc_dec_mask)
        
        # 处理主模型输出
        if isinstance(transformer_output, tuple) and len(transformer_output) == 3:
            logits, attention_weights, router_logits = transformer_output
        elif isinstance(transformer_output, tuple) and len(transformer_output) == 2:
            logits, attention_weights = transformer_output
            router_logits = None
        else:
            logits = transformer_output
            attention_weights = None
            router_logits = None
        
        # 如果启用 MTP，进行多 token 预测
        mtp_logits = None
        if hasattr(self, 'mtp_module'):
            # 这里需要根据具体的 MTP 逻辑来实现
            # 暂时返回主模型的结果
            pass
        
        # 返回结果
        if mtp_logits is not None:
            return logits, attention_weights, router_logits, mtp_logits
        elif router_logits is not None:
            return logits, attention_weights, router_logits
        else:
            return logits, attention_weights


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
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
        
        # 用于将logits转换为embedding的线性层（延迟初始化）
        self.logits_to_embedding = None
        self.logits_to_hidden = None
        
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
        if hasattr(self, 'mtp_module') and self.mtp_module is not None:
            # 获取主模型的隐藏状态用于MTP预测
            # 这里需要从主模型中提取隐藏状态
            # 由于当前transformer输出结构，我们需要重新设计这部分
            
            # 临时解决方案：使用logits作为输入来生成MTP预测
            # 在实际应用中，应该从transformer的中间层获取隐藏状态
            batch_size, seq_len, vocab_size = logits.shape
            
            # 创建位置编码用于MTP
            positions = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(batch_size, -1)
            
            # 使用logits的最后一个token作为输入embedding的近似
            # 这里应该使用真正的embedding，但作为临时方案
            last_token_logits = logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            
            # 将logits转换为embedding-like表示
            # 使用线性变换将vocab_size维度映射到hidden_size
            if self.logits_to_embedding is None:
                self.logits_to_embedding = nn.Linear(vocab_size, self.mtp_config.hidden_size, bias=False).to(logits.device)
                # 将新创建的层注册为模块参数
                self.add_module('logits_to_embedding', self.logits_to_embedding)
            
            inputs_embeds = self.logits_to_embedding(last_token_logits)  # [batch_size, 1, hidden_size]
            
            # 使用主模型的输出作为previous_hidden_states
            # 这里需要重新设计，因为我们需要真正的隐藏状态
            if self.logits_to_hidden is None:
                self.logits_to_hidden = nn.Linear(vocab_size, self.mtp_config.hidden_size, bias=False).to(logits.device)
                # 将新创建的层注册为模块参数
                self.add_module('logits_to_hidden', self.logits_to_hidden)
            
            previous_hidden_states = self.logits_to_hidden(last_token_logits)  # [batch_size, 1, hidden_size]
            
            # 调用MTP模块进行多token预测
            try:
                _, mtp_logits = self.mtp_module(
                    input_ids=None,  # MTP不需要input_ids
                    positions=positions[:, -1:],  # 只使用最后一个位置
                    hidden_states=previous_hidden_states,
                    inputs_embeds=inputs_embeds,
                    spec_step_idx=0
                )
                # 只在第一次执行时打印日志（使用类级别标志避免DP模式重复打印）
                if not hasattr(DeepSeekMTPWrapper, '_first_forward_logged'):
                    logger.info(f"✅ MTP初始化成功: 预测层数={len(mtp_logits)}, 示例logits形状={mtp_logits[0].shape if mtp_logits else None}")
                    DeepSeekMTPWrapper._first_forward_logged = True
            except Exception as e:
                logger.error(f"MTP forward error: {e}")
                mtp_logits = None
        
        # 返回结果
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
    
    # 为每个预测步骤计算损失
    for i, mtp_logits in enumerate(mtp_logits_list):
        # mtp_logits: [batch_size, seq_len, vocab_size] 或 [batch_size, 1, vocab_size]
        # 我们需要预测目标序列中接下来的 token
        
        # 获取目标序列中对应位置的 token
        if i + 1 < target_ids.shape[1]:
            target_tokens = target_ids[:, i + 1:i + 2]  # [batch_size, 1]
            
            # 确保logits和targets的维度匹配
            if mtp_logits.dim() == 3:
                # 如果是 [batch_size, seq_len, vocab_size]，取最后一个时间步
                pred_logits = mtp_logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            else:
                pred_logits = mtp_logits  # [batch_size, 1, vocab_size]
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                pred_logits.squeeze(1),  # [batch_size, vocab_size]
                target_tokens.squeeze(1),  # [batch_size]
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
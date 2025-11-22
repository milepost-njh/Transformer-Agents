# -*- coding: utf-8 -*-
"""
DeepSeek MTP (Multi-Token Prediction) 模型实现
作为 Transformer 的辅助预测模块，用于多 token 预测和推测解码

直接复用 train_ddp_latest.py 中的完整版 EncoderLayer：
- 支持 RoPE（Rotary Position Embedding）
- 支持 MLA（Multi-head Latent Attention）
- 支持 MoE（Mixture of Experts）
- 使用 RMSNorm（而非 LayerNorm）
- 符合论文要求的 Transformer Block 结构（Self-Attention + FFN）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from loguru import logger

from core.normalization import RMSNorm
from core.models.modeling_deepseek import DeepseekV3MoE

# 直接导入 EncoderLayer（Self-Attention + FFN）
# 使用 train_ddp_latest.py 中的完整版本（支持 RoPE、MLA、MoE、RMSNorm）
try:
    from train_ddp_latest import EncoderLayer as AdvancedEncoderLayer
    _ENCODER_LAYER_AVAILABLE = True
    logger.info("✅ 成功导入 train_ddp_latest.EncoderLayer（支持 RoPE/MLA/MoE）")
except ImportError:
    _ENCODER_LAYER_AVAILABLE = False
    logger.warning("⚠️ 无法导入 train_ddp_latest.EncoderLayer，将使用 PyTorch 内置的 TransformerEncoderLayer")


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
        num_heads: int = 8,  # Transformer Block 的注意力头数
        dff: int = 2048,  # FFN 中间层维度
        dropout_rate: float = 0.1,  # Dropout 比率
    ):
        self.hidden_size = hidden_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_config = moe_config
        self.mtp_loss_weight = mtp_loss_weight
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate


class DeepSeekMTPLayer(nn.Module):
    """MTP 单层实现 - 用于多 token 预测"""
    def __init__(self, config: DeepSeekMTPConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 输入归一化
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 特征融合投影（将拼接后的 2d 维度映射回 d 维度）
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # MTP Transformer Block（论文中的单层 Transformer）
        # 直接使用 train_ddp_latest.py 中的完整版 EncoderLayer
        if _ENCODER_LAYER_AVAILABLE:
            # 使用完整版 EncoderLayer（支持 RoPE、MLA、MoE、RMSNorm）
            logger.info(f"MTP Layer {layer_idx} 使用 AdvancedEncoderLayer (RoPE/MLA/MoE/RMSNorm)")
            self.mtp_block = AdvancedEncoderLayer(
                d_model=config.hidden_size,
                num_heads=config.num_heads,
                dff=config.dff,
                rate=config.dropout_rate,
                use_rope=False,      # MTP 内部不需要 RoPE（主模型已应用）
                use_moe=config.use_moe,
                moe_config=config.moe_config,
                use_mla=False,       # MTP 使用标准注意力即可
            )
        elif config.use_moe and config.moe_config is not None:
            # 如果配置了 MoE，使用 MoE Block（实验性）
            logger.warning(f"MTP Layer {layer_idx} 使用 MoE Block（实验性，不符合论文标准实现）")
            self.mtp_block = DeepseekV3MoE(config.moe_config)
        else:
            # 备用方案：使用 PyTorch 内置 TransformerEncoderLayer
            logger.info(f"MTP Layer {layer_idx} 使用 PyTorch 内置 TransformerEncoderLayer")
            self.mtp_block = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.dff,
                dropout=config.dropout_rate,
                activation='relu',
                batch_first=True,
                norm_first=False,  # Post-norm（先残差后归一化）
            )
        
        # 每个MTP层只有一个预测头（参数共享）
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
        实现论文中的 MTP 前向传播：
        
        公式：
            h'_i^k = M_k[RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]
            h_{1:T-k}^k = TRM_k(h'_{1:T-k})
            P_{i+k+1}^k = OutHead(h_i^k)
        
        对应图片中的流程：
            1. h^{k-1} 和 Emb(t_{i+k}) 分别经过 RMSNorm
            2. 拼接后经过 Linear Projection (M_k)
            3. 通过 Transformer Block (TRM_k)
            4. 通过 Output Head 生成预测
        
        Args:
            input_ids: 输入 token IDs（可选）
            positions: 位置信息 [batch_size, seq_len]
            previous_hidden_states: 上一层的隐藏状态 h^{k-1} [batch_size, seq_len, hidden_size]
                                   （因果链传递：来自主模型或上一个 MTP Module）
            inputs_embeds: 当前输入的 embedding Emb(t_{i+k}) [batch_size, seq_len, hidden_size]
                          （Teacher Forcing：使用 ground truth token 的 embedding）
            spec_step_index: 推测步骤索引 k
        
        Returns:
            hidden_states: 处理后的隐藏状态 h^k（传递给下一个 MTP Module，形成因果链）
            mtp_logits: 多 token 预测的 logits 列表 P^k
        """
        # 掩码位置 0 的输入（MTP 不需要）
        inputs_embeds = inputs_embeds.clone()
        if positions is not None:
             inputs_embeds[positions == 0] = 0
        
        # 步骤1: 归一化 - RMSNorm(h^{k-1}) 和 RMSNorm(Emb(t_{i+k}))
        inputs_embeds_norm = self.enorm(inputs_embeds)
        previous_hidden_states_norm = self.hnorm(previous_hidden_states)
        
        # 步骤2: 拼接 - [RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))]
        concat_features = torch.cat([previous_hidden_states_norm, inputs_embeds_norm], dim=-1)
        
        # 步骤3: 线性投影 M_k - 将 2d 维度映射回 d 维度
        hidden_states = self.eh_proj(concat_features)
        
        # 步骤4: 通过 Transformer Block - TRM_k(h')
        if _ENCODER_LAYER_AVAILABLE:
            # 使用 train_ddp_latest.py 的 AdvancedEncoderLayer
            # forward(x, src_mask, past_key_value, use_cache)
            # 返回: out2 或 (out2, router_logits) 或 (out2, present_key_value) 等
            mtp_output = self.mtp_block(hidden_states, src_mask=None, use_cache=False)
            
            # 处理返回值（可能包含 router_logits）
            if isinstance(mtp_output, tuple):
                hidden_states = mtp_output[0]  # 第一个总是 hidden states
            else:
                hidden_states = mtp_output
        elif self.config.use_moe and self.config.moe_config is not None:
            # MoE 分支（实验性，不推荐）
            mtp_output = self.mtp_block(hidden_states)
            if isinstance(mtp_output, tuple):
                hidden_states, _ = mtp_output
            else:
                hidden_states = mtp_output
        else:
            # PyTorch 内置 TransformerEncoderLayer
            hidden_states = self.mtp_block(hidden_states, src_mask=None)
        
        # 步骤5: 输出预测 - OutHead(h^k) -> P^k
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
            
            # ============================================================
            # 📌 Causal Chain（因果链）实现
            # ============================================================
            # 根据 DeepSeek V3 论文，MTP 模块形成因果链：
            # Main Model → MTP Module 1 → MTP Module 2 → ...
            #    h⁰            h¹              h²
            #              (Next² Token)  (Next³ Token)
            #
            # 每个 MTP Module 的输出 h^k 会传递给下一个 Module 作为输入
            # 形成递归的预测链，实现多步预测
            # ============================================================
            
            # 递归预测未来 K 个 token
            all_mtp_logits = []
            # ✅ 因果链起点：主模型的最后一层输出（图中"从最后一层传出来"）
            current_hidden_states = hidden_states
            
            # 循环预测每一层（因果链传递）
            for i in range(self.mtp_config.num_nextn_predict_layers):
                try:
                    # ============================================================
                    # 📌 MTP Module k 的处理
                    # ============================================================
                    # 输入：
                    #   - hidden_states: h^{k-1}（上一层的输出，因果链传递）
                    #   - inputs_embeds: Emb(t_{i+k})（当前位置的 embedding）
                    # 
                    # 输出：
                    #   - current_hidden_states: h^k（传递给下一层，因果链延续）
                    #   - layer_mtp_logits: P^k（预测 t_{i+k+1}）
                    #
                    # 公式：h^k = TRM_k([RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))])
                    # ============================================================
                    
                    current_hidden_states, layer_mtp_logits = self.mtp_module(
                        input_ids=None,
                        positions=positions,
                        hidden_states=current_hidden_states,  # ← 因果链：上一层的输出
                        inputs_embeds=inputs_embeds,          # ← Teacher Forcing 的 ground truth
                        spec_step_idx=i
                    )
                    # ✅ current_hidden_states 被更新为当前层的输出
                    # ✅ 将在下一次循环中作为输入，实现因果链传递
                    
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

#!/usr/bin/env python3
"""MTP 调试脚本 - PyCharm 打断点调试"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from loguru import logger

def test_mtp_layer():
    """测试 MTP Layer - 在这里打断点调试"""
    from core.models.deepseek_mtp import DeepSeekMTPConfig, DeepSeekMTPLayer
    
    logger.info("🔍 MTP Layer 调试")
    
    # 创建配置和模型
    config = DeepSeekMTPConfig(hidden_size=512, vocab_size=8192)
    layer = DeepSeekMTPLayer(config, layer_idx=0)
    layer.eval()
    
    # 准备输入数据
    real_data_path = 'test/debugs/mtp_real_data.pt'
    if os.path.exists(real_data_path):
        # 使用服务器保存的真实数据
        logger.info(f"✅ 加载真实数据: {real_data_path}")
        data = torch.load(real_data_path, map_location='cpu')
        previous_hidden_states = data['previous_hidden_states']
        inputs_embeds = data['inputs_embeds']
        positions = data['positions']
    else:
        # 使用随机模拟数据
        logger.warning(f"⚠️  未找到真实数据，使用随机数据")
        batch_size, seq_len = 1, 5
        previous_hidden_states = torch.randn(batch_size, seq_len, 512) * 0.6
        inputs_embeds = torch.randn(batch_size, seq_len, 512) * 11
        positions = torch.arange(seq_len).unsqueeze(0)
    
    logger.info(f"输入形状:")
    logger.info(f"  previous_hidden_states: {previous_hidden_states.shape}")
    logger.info(f"  inputs_embeds: {inputs_embeds.shape}")
    logger.info(f"  positions: {positions.shape}")
    
    # 🔴 在这行打断点，查看输入变量
    with torch.no_grad():
        hidden_states, mtp_logits = layer(
            input_ids=None,
            positions=positions,
            previous_hidden_states=previous_hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_index=0
        )
    
    # 🔴 在这行打断点，查看输出变量
    logger.info(f"输出形状:")
    logger.info(f"  hidden_states: {hidden_states.shape}")
    logger.info(f"  mtp_logits: {mtp_logits[0].shape}")
    
    # 查看 Top-5 预测
    last_logits = mtp_logits[0][0, -1, :]
    top5_tokens = torch.topk(last_logits, k=5).indices
    logger.info(f"  Top-5 tokens: {top5_tokens.tolist()}")
    
    return hidden_states, mtp_logits


def test_mtp_causal_chain():
    """测试 MTP 因果链 - 在这里打断点调试多层传递"""
    from core.models.deepseek_mtp import DeepSeekMTPConfig, DeepSeekMTP
    
    logger.info("🔗 MTP 因果链调试")
    
    config = DeepSeekMTPConfig(
        hidden_size=512,
        num_nextn_predict_layers=2,
        vocab_size=8192
    )
    mtp = DeepSeekMTP(config)
    mtp.eval()
    
    # 准备输入
    real_data_path = 'test/debugs/mtp_real_data.pt'
    if os.path.exists(real_data_path):
        logger.info(f"✅ 加载真实数据: {real_data_path}")
        data = torch.load(real_data_path, map_location='cpu')
        current_hidden = data['previous_hidden_states']
        inputs_embeds = data['inputs_embeds']
        positions = data['positions']
    else:
        logger.warning(f"⚠️  未找到真实数据，使用随机数据")
        batch_size, seq_len = 1, 5
        current_hidden = torch.randn(batch_size, seq_len, 512) * 0.6
        inputs_embeds = torch.randn(batch_size, seq_len, 512) * 11
        positions = torch.arange(seq_len).unsqueeze(0)
    
    logger.info(f"初始 hidden: {current_hidden.shape}")
    
    # 因果链传递：h⁰ → MTP₀ → h¹ → MTP₁ → h²
    for i in range(config.num_nextn_predict_layers):
        logger.info(f"\nMTP Layer {i}:")
        
        # 🔴 在这行打断点，查看每一层的输入
        with torch.no_grad():
            current_hidden, mtp_logits = mtp(
                input_ids=None,
                positions=positions,
                hidden_states=current_hidden,  # ← 上一层的输出作为输入
                inputs_embeds=inputs_embeds,
                spec_step_idx=i
            )
        
        # 🔴 在这行打断点，查看每一层的输出
        logger.info(f"  输出 hidden: {current_hidden.shape}")
        logger.info(f"  输出 logits: {mtp_logits[0].shape}")
    
    logger.success("✅ 因果链测试完成")
    return current_hidden, mtp_logits


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MTP 调试脚本 - 直接运行或在 PyCharm 中打断点")
    logger.info("=" * 60)
    
    # 测试1: 单层 MTP
    logger.info("\n📍 测试 1: MTP Layer")
    hidden_states, mtp_logits = test_mtp_layer()
    
    # 测试2: 因果链
    logger.info("\n📍 测试 2: MTP 因果链")
    final_hidden, final_logits = test_mtp_causal_chain()
    
    logger.success("\n✅ 所有测试完成")


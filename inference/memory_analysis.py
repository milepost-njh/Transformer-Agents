# -*- coding: utf-8 -*-
"""
模型显存需求分析脚本
分析不同配置下的显存使用情况
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, List
from loguru import logger


def calculate_model_parameters(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dff: int,
    max_length: int,
    use_moe: bool = False,
    num_experts: int = 8,
    use_mla: bool = False,
    use_mtp: bool = False
) -> Dict[str, int]:
    """计算模型参数数量"""
    
    params = {}
    
    # 1. Embedding层
    params['embedding'] = vocab_size * d_model * 2  # 源语言和目标语言
    
    # 2. 位置编码 (如果不使用RoPE)
    params['position_embedding'] = max_length * d_model * 2
    
    # 3. Transformer层
    layer_params = 0
    
    # 3.1 注意力层
    if use_mla:
        # MLA模式：低秩分解
        q_lora_rank = d_model // 2
        kv_lora_rank = d_model // 4
        q_head_dim = d_model // num_heads
        v_head_dim = d_model // num_heads
        
        # Q投影
        layer_params += d_model * q_lora_rank  # q_a_proj
        layer_params += q_lora_rank * num_heads * q_head_dim  # q_b_proj
        
        # KV投影
        layer_params += d_model * (kv_lora_rank + q_head_dim)  # kv_a_proj_with_mqa
        layer_params += kv_lora_rank * num_heads * (q_head_dim + v_head_dim)  # kv_b_proj
        
        # 输出投影
        layer_params += num_heads * v_head_dim * d_model  # out_proj
    else:
        # 标准注意力
        layer_params += d_model * d_model * 3  # WQ, WK, WV
        layer_params += d_model * d_model  # out_proj
    
    # 3.2 前馈网络
    if use_moe:
        # MoE: 8个专家，每个专家2层
        expert_params = d_model * dff + dff * d_model  # 每个专家的参数
        layer_params += num_experts * expert_params
        layer_params += d_model * num_experts  # 路由器
    else:
        # 标准FFN
        layer_params += d_model * dff + dff * d_model
    
    # 3.3 归一化层
    layer_params += d_model * 2  # 两个RMSNorm层
    
    # 编码器和解码器各有num_layers层
    params['encoder_layers'] = num_layers * layer_params
    params['decoder_layers'] = num_layers * layer_params
    
    # 4. 最终输出层
    params['final_layer'] = d_model * vocab_size
    
    # 5. MTP模块 (如果使用)
    if use_mtp:
        mtp_params = 0
        # MTP层参数
        mtp_params += d_model * 2 * d_model  # eh_proj
        mtp_params += d_model * d_model * 4  # mtp_block (简化)
        mtp_params += d_model * vocab_size  # mtp_head
        params['mtp_module'] = mtp_params * 2  # 2个MTP层
    
    # 总参数
    total_params = sum(params.values())
    params['total'] = total_params
    
    return params


def calculate_memory_usage(
    params: Dict[str, int],
    batch_size: int = 1,
    seq_length: int = 64,
    precision: str = "fp16"
) -> Dict[str, float]:
    """计算显存使用量"""
    
    # 精度设置
    bytes_per_param = 2 if precision == "fp16" else 4  # fp16: 2 bytes, fp32: 4 bytes
    
    memory = {}
    
    # 1. 模型参数显存
    total_params = params['total']
    memory['model_params'] = total_params * bytes_per_param / (1024**3)  # GB
    
    # 2. 梯度显存 (训练时)
    memory['gradients'] = total_params * bytes_per_param / (1024**3)  # GB
    
    # 3. 优化器状态显存 (AdamW)
    memory['optimizer'] = total_params * bytes_per_param * 2 / (1024**3)  # GB (momentum + variance)
    
    # 4. 激活值显存
    d_model = 512  # 从配置中获取
    num_layers = 8
    num_heads = 8
    
    # 4.1 注意力激活值
    attention_activations = batch_size * seq_length * d_model * num_layers * 2  # 编码器和解码器
    memory['attention_activations'] = attention_activations * bytes_per_param / (1024**3)
    
    # 4.2 FFN激活值
    ffn_activations = batch_size * seq_length * 2048 * num_layers * 2  # dff=2048
    memory['ffn_activations'] = ffn_activations * bytes_per_param / (1024**3)
    
    # 4.3 其他激活值
    other_activations = batch_size * seq_length * d_model * num_layers * 4  # 各种中间状态
    memory['other_activations'] = other_activations * bytes_per_param / (1024**3)
    
    # 总激活值
    memory['total_activations'] = (
        memory['attention_activations'] + 
        memory['ffn_activations'] + 
        memory['other_activations']
    )
    
    # 5. 总显存需求
    memory['inference_total'] = memory['model_params'] + memory['total_activations']
    memory['training_total'] = (
        memory['model_params'] + 
        memory['gradients'] + 
        memory['optimizer'] + 
        memory['total_activations']
    )
    
    return memory


def analyze_memory_requirements():
    """分析不同配置下的显存需求"""
    
    # 基础配置 (来自train_tmp.py)
    base_config = {
        'vocab_size': 2**13,  # 8192
        'd_model': 512,
        'num_layers': 8,
        'num_heads': 8,
        'dff': 2048,
        'max_length': 64,
        'use_moe': True,
        'num_experts': 8,
        'use_mla': True,
        'use_mtp': True
    }
    
    # 不同配置场景
    scenarios = {
        'current_config': base_config,
        'small_config': {**base_config, 'd_model': 256, 'num_layers': 4, 'dff': 1024},
        'medium_config': base_config,
        'large_config': {**base_config, 'd_model': 1024, 'num_layers': 12, 'dff': 4096},
        'inference_only': {**base_config, 'use_moe': False, 'use_mla': False, 'use_mtp': False},
        'mla_only': {**base_config, 'use_moe': False, 'use_mtp': False},
        'moe_only': {**base_config, 'use_mla': False, 'use_mtp': False},
    }
    
    results = {}
    
    for scenario_name, config in scenarios.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"分析场景: {scenario_name}")
        logger.info(f"{'='*60}")
        
        # 计算参数
        params = calculate_model_parameters(**config)
        
        # 计算显存
        memory_fp16 = calculate_memory_usage(params, batch_size=1, seq_length=64, precision="fp16")
        memory_fp32 = calculate_memory_usage(params, batch_size=1, seq_length=64, precision="fp32")
        
        # 训练时显存 (batch_size=32)
        memory_training = calculate_memory_usage(params, batch_size=32, seq_length=64, precision="fp16")
        
        results[scenario_name] = {
            'params': params,
            'memory_fp16': memory_fp16,
            'memory_fp32': memory_fp32,
            'memory_training': memory_training
        }
        
        # 打印结果
        logger.info(f"模型参数数量: {params['total']:,}")
        logger.info(f"推理显存需求 (FP16): {memory_fp16['inference_total']:.2f} GB")
        logger.info(f"推理显存需求 (FP32): {memory_fp32['inference_total']:.2f} GB")
        logger.info(f"训练显存需求 (FP16, batch=32): {memory_training['training_total']:.2f} GB")
        
        # 详细分解
        logger.info(f"\n详细显存分解 (FP16):")
        logger.info(f"  模型参数: {memory_fp16['model_params']:.2f} GB")
        logger.info(f"  激活值: {memory_fp16['total_activations']:.2f} GB")
        logger.info(f"    - 注意力激活: {memory_fp16['attention_activations']:.2f} GB")
        logger.info(f"    - FFN激活: {memory_fp16['ffn_activations']:.2f} GB")
        logger.info(f"    - 其他激活: {memory_fp16['other_activations']:.2f} GB")
    
    return results


def recommend_gpu_configurations():
    """推荐GPU配置"""
    
    logger.info(f"\n{'='*60}")
    logger.info("GPU配置推荐")
    logger.info(f"{'='*60}")
    
    recommendations = {
        'RTX 4090 (24GB)': {
            'inference': '支持所有配置的推理',
            'training': '支持小到中等配置的训练 (batch_size ≤ 16)',
            'note': '推荐用于开发和实验'
        },
        'RTX 3090 (24GB)': {
            'inference': '支持所有配置的推理',
            'training': '支持小到中等配置的训练 (batch_size ≤ 8)',
            'note': '性价比高，适合训练'
        },
        'RTX 4080 (16GB)': {
            'inference': '支持所有配置的推理',
            'training': '仅支持小配置训练 (batch_size ≤ 4)',
            'note': '推理性能好，训练受限'
        },
        'RTX 4070 (12GB)': {
            'inference': '支持小到中等配置推理',
            'training': '仅支持小配置训练 (batch_size ≤ 2)',
            'note': '入门级选择'
        },
        'A100 (40GB)': {
            'inference': '支持所有配置的推理',
            'training': '支持大配置训练 (batch_size ≤ 64)',
            'note': '专业级，适合大规模训练'
        },
        'V100 (32GB)': {
            'inference': '支持所有配置的推理',
            'training': '支持中等配置训练 (batch_size ≤ 32)',
            'note': '数据中心级'
        }
    }
    
    for gpu, specs in recommendations.items():
        logger.info(f"\n{gpu}:")
        logger.info(f"  推理: {specs['inference']}")
        logger.info(f"  训练: {specs['training']}")
        logger.info(f"  备注: {specs['note']}")


def memory_optimization_tips():
    """显存优化建议"""
    
    logger.info(f"\n{'='*60}")
    logger.info("显存优化建议")
    logger.info(f"{'='*60}")
    
    tips = [
        "1. 使用混合精度训练 (FP16/BF16)",
        "2. 启用梯度检查点 (Gradient Checkpointing)",
        "3. 使用数据并行 (DataParallel/DistributedDataParallel)",
        "4. 减少批处理大小",
        "5. 使用序列长度截断",
        "6. 启用内存池优化",
        "7. 使用模型并行 (对于超大模型)",
        "8. 考虑使用CPU卸载 (CPU Offloading)",
        "9. 使用DeepSpeed ZeRO优化器",
        "10. 启用Flash Attention (如果支持)"
    ]
    
    for tip in tips:
        logger.info(tip)


def main():
    """主函数"""
    logger.info("🚀 开始分析模型显存需求...")
    
    # 分析显存需求
    results = analyze_memory_requirements()
    
    # 推荐GPU配置
    recommend_gpu_configurations()
    
    # 显存优化建议
    memory_optimization_tips()
    
    logger.info(f"\n{'='*60}")
    logger.info("分析完成!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

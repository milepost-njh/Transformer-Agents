#!/usr/bin/env python3
"""
真正的MLA vs 标准注意力KV-cache效率对比脚本
实现真正的KV-cache机制，测试自回归生成中的内存使用

使用方法:
CUDA_VISIBLE_DEVICES=5 python inference/compare_mla_kv_cache_real.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_lengths 32 64 128
"""

import os
import sys
import time
import torch
import psutil
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 使用新的对比脚本
from compare_kv_cache_mla import (
    benchmark_model, compare_results, clear_memory, get_gpu_memory
)


def main():
    """主函数 - 简化版，调用新的对比脚本"""
    parser = argparse.ArgumentParser(description="真正的MLA vs 标准注意力KV-cache效率对比")
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
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("🚀 真正的MLA vs 标准注意力KV-cache效率对比")
    logger.info("=" * 60)
    logger.info("基于DeepSeek MLA技术原理")
    logger.info("测试自回归生成中的KV-cache压缩效果")
    
    # 检查checkpoint文件
    if not os.path.exists(args.mla_checkpoint):
        logger.error(f"❌ MLA checkpoint not found: {args.mla_checkpoint}")
        return
    
    if not os.path.exists(args.no_mla_checkpoint):
        logger.error(f"❌ Standard checkpoint not found: {args.no_mla_checkpoint}")
        return
    
    logger.info(f"✅ Found checkpoints:")
    logger.info(f"   MLA: {args.mla_checkpoint}")
    logger.info(f"   Standard: {args.no_mla_checkpoint}")
    logger.info(f"\n📝 测试输入: {args.test_input}")
    
    all_results = []
    
    for max_new_tokens in args.test_lengths:
        logger.info(f"\n{'='*20} 生成长度: {max_new_tokens} {'='*20}")
        
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
    
    logger.info("\n✅ 真正的KV-cache对比完成!")

if __name__ == "__main__":
    main()

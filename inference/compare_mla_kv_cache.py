#!/usr/bin/env python3
"""
MLA vs 标准注意力KV-cache效率对比脚本
重点测试长序列下的KV-cache内存使用和推理速度
基于DeepSeek MLA技术原理实现
"""

import os
import sys
import time
import torch
import psutil
import gc
import argparse
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference.inference import InferenceEngine, InferenceConfig

def get_gpu_memory():
    """获取GPU内存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_long_test_input(base_text, target_length=200):
    """创建长序列测试输入"""
    words = base_text.split()
    if len(words) >= target_length:
        return " ".join(words[:target_length])
    
    # 重复扩展文本以达到目标长度
    extended_text = base_text
    while len(extended_text.split()) < target_length:
        extended_text += " " + base_text
    
    return " ".join(extended_text.split()[:target_length])

def benchmark_kv_cache(config, checkpoint_path, test_input, model_name, max_length=512):
    """专门测试KV-cache的基准测试"""
    print(f"\n🔍 测试 {model_name} 模型 (序列长度: {max_length})...")
    
    # 清理内存
    clear_memory()
    initial_gpu_memory = get_gpu_memory()
    
    # 创建推理引擎
    config.checkpoint_path = checkpoint_path
    config.max_length = max_length
    engine = InferenceEngine(config)
    
    # 初始化模型
    start_time = time.time()
    engine.initialize()
    init_time = time.time() - start_time
    
    # 记录模型加载后内存
    model_gpu_memory = get_gpu_memory()
    
    # 长序列推理测试
    print(f"   开始长序列推理测试...")
    start_time = time.time()
    result = engine.infer_single(test_input, "greedy")
    inference_time = time.time() - start_time
    
    # 记录推理后内存
    final_gpu_memory = get_gpu_memory()
    
    # 计算KV-cache相关指标
    kv_cache_memory = final_gpu_memory - model_gpu_memory
    
    print(f"   推理时间: {inference_time:.3f}s")
    print(f"   输出长度: {len(result['tokens'])} tokens")
    print(f"   KV-cache内存: {kv_cache_memory:.1f} MB")
    
    return {
        'model_name': model_name,
        'init_time': init_time,
        'inference_time': inference_time,
        'output_length': len(result['tokens']),
        'initial_gpu_memory': initial_gpu_memory,
        'model_gpu_memory': model_gpu_memory,
        'final_gpu_memory': final_gpu_memory,
        'kv_cache_memory': kv_cache_memory,
        'output': result['output']
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MLA vs 标准注意力KV-cache效率对比")
    parser.add_argument("--mla_checkpoint", type=str, default="checkpoints/latest.pt",
                       help="MLA模型checkpoint路径")
    parser.add_argument("--no_mla_checkpoint", type=str, default="checkpoints/latest.pt",
                       help="非MLA模型checkpoint路径")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--test_lengths", nargs='+', type=int, default=[128, 256, 512],
                       help="测试的序列长度列表")
    
    args = parser.parse_args()
    
    print("🚀 MLA vs 标准注意力KV-cache效率对比")
    print("=" * 60)
    print("基于DeepSeek MLA技术原理")
    print("重点测试KV-cache压缩效果")
    
    # 检查checkpoint文件
    if not os.path.exists(args.mla_checkpoint):
        print(f"❌ MLA checkpoint not found: {args.mla_checkpoint}")
        return
    
    # 查找非MLA checkpoint
    import glob
    no_mla_files = glob.glob(args.no_mla_checkpoint)
    if not no_mla_files:
        print(f"❌ No-MLA checkpoint not found: {args.no_mla_checkpoint}")
        return
    
    no_mla_checkpoint = no_mla_files[0]
    print(f"✅ Found checkpoints:")
    print(f"   MLA: {args.mla_checkpoint}")
    print(f"   No-MLA: {no_mla_checkpoint}")
    
    # 检查模型训练程度
    print(f"\n⚠️  重要提醒：")
    print(f"   请确保两个模型训练到相同的epoch，否则对比结果不可信！")
    print(f"   建议：")
    print(f"   1. 都用最新模型：--mla_checkpoint latest.pt --no_mla_checkpoint latest.pt")
    print(f"   2. 都用相同epoch：--mla_checkpoint mid_e8_s*.pt --no_mla_checkpoint mid_e8_s*.pt")
    print(f"   3. 训练两个模型到相同程度后再对比")
    
    # 创建长序列测试输入
    base_text = "O Tom está procurando uma segunda opinião sobre o tratamento médico que recebeu. Ele quer entender melhor as opções disponíveis e os possíveis efeitos colaterais. A tecnologia médica está evoluindo rapidamente, oferecendo novas possibilidades de diagnóstico e tratamento. O aprendizado de máquina e a inteligência artificial estão revolucionando a área da saúde, permitindo análises mais precisas e personalizadas. O futuro da medicina parece promissor, com avanços em terapia genética, medicina regenerativa e telemedicina."
    
    long_test_input = create_long_test_input(base_text, 150)
    print(f"\n📝 测试输入长度: {len(long_test_input.split())} words")
    print(f"   输入预览: {long_test_input[:100]}...")
    
    all_results = []
    
    for max_length in args.test_lengths:
        print(f"\n{'='*20} 序列长度: {max_length} {'='*20}")
        
        # 测试MLA模型
        mla_config = InferenceConfig(
            use_mla=True,
            use_mtp=True,
            use_moe=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        mla_result = benchmark_kv_cache(mla_config, args.mla_checkpoint, long_test_input, "MLA", max_length)
        all_results.append(mla_result)
        
        # 清理内存
        clear_memory()
        
        # 测试非MLA模型
        no_mla_config = InferenceConfig(
            use_mla=False,
            use_mtp=True,
            use_moe=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        no_mla_result = benchmark_kv_cache(no_mla_config, no_mla_checkpoint, long_test_input, "No-MLA", max_length)
        all_results.append(no_mla_result)
        
        # 当前长度对比
        print(f"\n📊 序列长度 {max_length} 对比结果:")
        print(f"   推理时间 - MLA: {mla_result['inference_time']:.3f}s, No-MLA: {no_mla_result['inference_time']:.3f}s")
        print(f"   KV-cache内存 - MLA: {mla_result['kv_cache_memory']:.1f}MB, No-MLA: {no_mla_result['kv_cache_memory']:.1f}MB")
        
        if no_mla_result['kv_cache_memory'] > 0:
            kv_savings = (no_mla_result['kv_cache_memory'] - mla_result['kv_cache_memory']) / no_mla_result['kv_cache_memory'] * 100
            print(f"   MLA KV-cache节省: {kv_savings:.1f}%")
        
        clear_memory()
    
    # 总结对比
    print(f"\n{'='*60}")
    print("📈 总结对比 (MLA vs No-MLA)")
    print(f"{'='*60}")
    
    mla_results = [r for r in all_results if r['model_name'] == 'MLA']
    no_mla_results = [r for r in all_results if r['model_name'] == 'No-MLA']
    
    print(f"{'序列长度':<10} {'推理时间对比':<20} {'KV-cache内存对比':<25} {'MLA优势':<15}")
    print("-" * 70)
    
    for i, length in enumerate(args.test_lengths):
        mla = mla_results[i]
        no_mla = no_mla_results[i]
        
        time_diff = no_mla['inference_time'] - mla['inference_time']
        time_improvement = (time_diff / no_mla['inference_time'] * 100) if no_mla['inference_time'] > 0 else 0
        
        kv_diff = no_mla['kv_cache_memory'] - mla['kv_cache_memory']
        kv_improvement = (kv_diff / no_mla['kv_cache_memory'] * 100) if no_mla['kv_cache_memory'] > 0 else 0
        
        advantage = "✅" if kv_improvement > 0 else "❌"
        
        print(f"{length:<10} {time_improvement:+.1f}% ({time_diff:+.3f}s){'':<5} {kv_improvement:+.1f}% ({kv_diff:+.1f}MB){'':<5} {advantage}")
    
    print(f"\n💡 结论:")
    print(f"   - MLA的核心优势是KV-cache压缩，通过低秩分解减少存储")
    print(f"   - 序列越长，MLA的KV-cache节省效果越明显")
    print(f"   - 推理速度可能因计算复杂度而有所差异")
    print(f"   - 这符合DeepSeek MLA技术原理：空间换时间")
    
    print("\n✅ 对比完成!")

if __name__ == "__main__":
    main()

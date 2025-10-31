#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内存诊断脚本：逐步测试每个阶段的内存占用
"""

import os
import sys
import gc
import psutil

# 禁用TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6,7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_memory():
    """打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    available_gb = vm.available / (1024 ** 3)
    used_gb = vm.used / (1024 ** 3)
    
    print(f"进程内存: {mem_gb:.2f} GB | 系统内存: {used_gb:.2f}/{total_gb:.2f} GB (可用: {available_gb:.2f} GB)")

print("\n" + "=" * 80)
print("🔍 内存诊断开始")
print("=" * 80)

print("\n📌 阶段0: 导入基础库之前")
print_memory()

# 阶段1: 导入PyTorch
print("\n📌 阶段1: 导入PyTorch")
import torch
import torch.nn as nn
print_memory()

# 阶段2: 导入transformers
print("\n📌 阶段2: 导入transformers")
from transformers import PreTrainedTokenizerFast
print_memory()

# 阶段3: 导入Kimi模型
print("\n📌 阶段3: 导入Kimi模型")
try:
    from core.models.kimi_linear.modeling_kimi import KimiLinearForCausalLM
    from core.models.kimi_linear.configuration_kimi import KimiLinearConfig
    print_memory()
    print("✅ Kimi模型导入成功")
except Exception as e:
    print(f"❌ Kimi模型导入失败: {e}")
    print_memory()
    sys.exit(1)

# 阶段4: 加载tokenizer
print("\n📌 阶段4: 加载tokenizer")
try:
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
    print_memory()
    print("✅ Tokenizer加载成功")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    print_memory()
    sys.exit(1)

# 阶段5: 加载数据缓存
print("\n📌 阶段5: 加载数据缓存")
try:
    import pickle
    with open("data_cache/train_kimi.pkl", 'rb') as f:
        train_sequences = pickle.load(f)
    print(f"训练集样本数: {len(train_sequences)}")
    del train_sequences
    gc.collect()
    print_memory()
    print("✅ 数据缓存加载成功")
except Exception as e:
    print(f"❌ 数据缓存加载失败: {e}")
    print_memory()
    sys.exit(1)

# 阶段6: 创建小模型配置
print("\n📌 阶段6: 创建小模型配置（测试）")
try:
    kimi_config = KimiLinearConfig(
        vocab_size=8192,
        hidden_size=256,  # 很小的模型
        head_dim=32,
        intermediate_size=512,
        num_hidden_layers=2,  # 只有2层
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    print_memory()
    print("✅ 配置创建成功")
except Exception as e:
    print(f"❌ 配置创建失败: {e}")
    print_memory()
    sys.exit(1)

# 阶段7: 初始化小模型
print("\n📌 阶段7: 初始化小模型")
try:
    model = KimiLinearForCausalLM(kimi_config)
    print_memory()
    print("✅ 小模型初始化成功")
    
    # 移动到GPU
    if torch.cuda.is_available():
        print("\n📌 阶段8: 移动模型到GPU")
        model = model.cuda()
        print_memory()
        print("✅ 模型已移动到GPU")
        
        # 检查GPU内存
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: 已用 {gpu_mem:.2f} GB, 预留 {gpu_reserved:.2f} GB")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    print_memory()
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有阶段测试通过！")
print("=" * 80)
print("\n💡 建议：")
print("1. 如果上面所有阶段都成功，说明小模型没问题")
print("2. 可能是train_tmp.py中的模型配置太大了")
print("3. 尝试减小: num_layers, d_model, dff, num_heads")
print("=" * 80)


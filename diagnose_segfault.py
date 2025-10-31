#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
细粒度Segmentation Fault诊断脚本
"""

import os
import sys

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6,7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关键：禁用Flash Attention等可能导致段错误的扩展
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("\n" + "=" * 80)
print("🔍 Segmentation Fault 诊断")
print("=" * 80)

print("\n步骤1: 导入 sys 和 os")
print("✅ sys, os 导入成功")

print("\n步骤2: 导入 torch")
try:
    import torch
    print(f"✅ torch 导入成功，版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   可用GPU数量: {torch.cuda.device_count()}")
except Exception as e:
    print(f"❌ torch 导入失败: {e}")
    sys.exit(1)

print("\n步骤3: 导入 torch.nn")
try:
    import torch.nn as nn
    print("✅ torch.nn 导入成功")
except Exception as e:
    print(f"❌ torch.nn 导入失败: {e}")
    sys.exit(1)

print("\n步骤4: 导入 transformers")
try:
    import transformers
    print(f"✅ transformers 导入成功，版本: {transformers.__version__}")
except Exception as e:
    print(f"❌ transformers 导入失败: {e}")
    sys.exit(1)

print("\n步骤5: 导入 transformers.PreTrainedModel")
try:
    from transformers import PreTrainedModel
    print("✅ PreTrainedModel 导入成功")
except Exception as e:
    print(f"❌ PreTrainedModel 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤6: 导入 transformers 常用组件")
try:
    from transformers import PretrainedConfig
    print("✅ PretrainedConfig 导入成功")
except Exception as e:
    print(f"❌ PretrainedConfig 导入失败: {e}")
    sys.exit(1)

print("\n步骤7: 测试 tiktoken（可能的问题源）")
try:
    import tiktoken
    print(f"✅ tiktoken 导入成功，版本: {tiktoken.__version__}")
except Exception as e:
    print(f"⚠️  tiktoken 导入失败: {e}")
    print("   这可能是问题所在！")

print("\n步骤8: 导入 KimiLinearConfig")
try:
    from core.models.kimi_linear.configuration_kimi import KimiLinearConfig
    print("✅ KimiLinearConfig 导入成功")
except Exception as e:
    print(f"❌ KimiLinearConfig 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤9: 导入 Kimi Tokenizer（可能触发tiktoken问题）")
try:
    from core.models.kimi_linear.tokenization_kimi import KimiTokenizer
    print("✅ KimiTokenizer 类导入成功")
except Exception as e:
    print(f"❌ KimiTokenizer 导入失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 这可能是问题所在 - tiktoken相关")
    sys.exit(1)

print("\n步骤10: 尝试初始化 Tokenizer（最危险的操作）")
try:
    tokenizer_path = "./core/models/kimi_linear"
    print(f"   Tokenizer 路径: {tokenizer_path}")
    # 不实际加载，只测试到这里
    print("   ⏭️  跳过实际加载以避免段错误")
except Exception as e:
    print(f"❌ 准备加载 tokenizer 失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有基础导入测试通过！")
print("=" * 80)
print("\n💡 如果在步骤9之前崩溃，问题在基础库")
print("   如果在步骤9-10崩溃，问题在 tiktoken/tokenizer")
print("=" * 80)


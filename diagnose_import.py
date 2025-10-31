#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入诊断脚本：逐个测试Kimi模型的导入
"""

import os
import sys

# 禁用TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6,7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("\n" + "=" * 80)
print("🔍 Kimi模型导入诊断")
print("=" * 80)

print("\n✅ 基础库导入...")
import torch
import torch.nn as nn
from transformers import PreTrainedModel
print("✅ 基础库导入成功")

print("\n📌 测试1: 导入 KimiLinearConfig")
try:
    from core.models.kimi_linear.configuration_kimi import KimiLinearConfig
    print("✅ KimiLinearConfig 导入成功")
except Exception as e:
    print(f"❌ KimiLinearConfig 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n📌 测试2: 逐个导入 Kimi 模型组件")

components = [
    "KimiRMSNorm",
    "KimiDynamicCache",
    "KimiMLAAttention",
    "KimiDeltaAttention",
    "KimiMLP",
    "KimiBlockSparseMLP",
    "KimiMoEGate",
    "KimiSparseMoeBlock",
    "KimiDecoderLayer",
    "KimiPreTrainedModel",
    "KimiLinearModel",
    "KimiLinearForCausalLM",
]

for component in components:
    print(f"\n  尝试导入: {component}")
    try:
        exec(f"from core.models.kimi_linear.modeling_kimi import {component}")
        print(f"  ✅ {component} 导入成功")
    except Exception as e:
        print(f"  ❌ {component} 导入失败!")
        print(f"     错误: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💥 导入在 {component} 处失败")
        sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有组件导入成功！")
print("=" * 80)
print("\n💡 说明：如果这个脚本成功了，说明导入本身没问题")
print("   可能是在实际使用时触发了问题")
print("=" * 80)


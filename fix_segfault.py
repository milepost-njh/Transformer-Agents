#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentation Fault 修复脚本
提供多个可能的解决方案
"""

import os
import sys

print("\n" + "=" * 80)
print("🔧 Segmentation Fault 修复方案")
print("=" * 80)

print("\n📋 可能的原因和解决方案：")
print("\n1️⃣  tiktoken 库问题（最常见）")
print("   解决方案：")
print("   - 重新安装 tiktoken: pip uninstall tiktoken && pip install tiktoken")
print("   - 或使用特定版本: pip install tiktoken==0.5.1")

print("\n2️⃣  Flash Attention 2 兼容性问题")
print("   解决方案：")
print("   - 禁用 Flash Attention: 设置环境变量")
print("     export KIMI_DISABLE_FLASH_ATTN=1")

print("\n3️⃣  transformers 库的加速特性")
print("   解决方案：")
print("   - 降级 transformers: pip install transformers==4.36.0")
print("   - 或禁用加速: export TRANSFORMERS_NO_ADVISORY_WARNINGS=1")

print("\n4️⃣  CUDA/PyTorch 版本不兼容")
print("   解决方案：")
print("   - 检查版本兼容性")
print("   - 重新安装匹配的 PyTorch 版本")

print("\n5️⃣  多进程tokenizer问题")
print("   解决方案：")
print("   - 已设置 TOKENIZERS_PARALLELISM=false")

print("\n" + "=" * 80)
print("🔍 正在检测环境...")
print("=" * 80)

# 检查已安装的包
try:
    import subprocess
    
    print("\n📦 检查关键包版本：")
    packages = ["torch", "transformers", "tiktoken"]
    
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        version = line.split(":")[1].strip()
                        print(f"   {package}: {version}")
                        break
            else:
                print(f"   {package}: ❌ 未安装")
        except Exception as e:
            print(f"   {package}: ⚠️  无法检测 ({e})")
            
except Exception as e:
    print(f"⚠️  无法运行检测: {e}")

print("\n" + "=" * 80)
print("💡 建议的修复步骤（按顺序尝试）：")
print("=" * 80)

fix_commands = """
# 步骤1: 更新 tiktoken（最可能的解决方案）
pip uninstall -y tiktoken
pip install tiktoken==0.5.1

# 步骤2: 如果步骤1无效，尝试禁用优化
export TOKENIZERS_PARALLELISM=false
export KIMI_DISABLE_FLASH_ATTN=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# 步骤3: 如果还不行，降级 transformers
pip install transformers==4.36.0

# 步骤4: 检查 PyTorch 和 CUDA 兼容性
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 步骤5: 如果都不行，尝试在 CPU 模式下运行
export CUDA_VISIBLE_DEVICES=""
"""

print(fix_commands)

print("\n" + "=" * 80)
print("🚀 自动尝试修复方案1：更新 tiktoken")
print("=" * 80)

user_input = input("\n是否尝试自动修复？(y/n): ")

if user_input.lower() == 'y':
    print("\n正在卸载旧版本 tiktoken...")
    os.system(f"{sys.executable} -m pip uninstall -y tiktoken")
    
    print("\n正在安装 tiktoken==0.5.1...")
    os.system(f"{sys.executable} -m pip install tiktoken==0.5.1")
    
    print("\n✅ 修复完成！请重新运行诊断脚本：")
    print("   python diagnose_import.py")
else:
    print("\n请手动执行上述命令")


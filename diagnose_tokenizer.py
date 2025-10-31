#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门测试 tokenizer 导入问题
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6,7"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("\n" + "=" * 80)
print("🔍 Tokenizer 导入测试")
print("=" * 80)

print("\n步骤1: 测试 tiktoken 导入")
try:
    import tiktoken
    print(f"✅ tiktoken 导入成功，版本: {tiktoken.__version__}")
except Exception as e:
    print(f"❌ tiktoken 导入失败: {e}")
    print("\n💡 解决方案:")
    print("   pip uninstall -y tiktoken")
    print("   pip install tiktoken==0.5.1")
    sys.exit(1)

print("\n步骤2: 测试 tiktoken.load")
try:
    from tiktoken.load import load_tiktoken_bpe
    print("✅ tiktoken.load 导入成功")
except Exception as e:
    print(f"❌ tiktoken.load 导入失败: {e}")
    sys.exit(1)

print("\n步骤3: 测试加载 tiktoken model 文件")
try:
    model_path = "./core/models/kimi_linear/tiktoken.model"
    print(f"   模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"   ❌ 文件不存在: {model_path}")
        sys.exit(1)
    
    print(f"   文件大小: {os.path.getsize(model_path)} bytes")
    
    # 危险操作 - 实际加载文件
    print("   正在加载 tiktoken model...")
    mergeable_ranks = load_tiktoken_bpe(model_path)
    print(f"   ✅ 加载成功！词表大小: {len(mergeable_ranks)}")
    
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 这是段错误的常见原因！")
    print("   解决方案:")
    print("   1. 重新安装 tiktoken: pip install --force-reinstall tiktoken==0.5.1")
    print("   2. 检查模型文件是否损坏")
    print("   3. 尝试从官方重新下载模型文件")
    sys.exit(1)

print("\n步骤4: 测试创建 tiktoken Encoding")
try:
    pat_str = r"[^\r\n\p{L}\p{N}]+"  # 简化的模式
    
    special_tokens = {
        "[BOS]": len(mergeable_ranks),
        "[EOS]": len(mergeable_ranks) + 1,
    }
    
    print("   正在创建 Encoding...")
    encoding = tiktoken.Encoding(
        name="test_encoding",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    print(f"   ✅ Encoding 创建成功！词表大小: {encoding.n_vocab}")
    
except Exception as e:
    print(f"   ❌ Encoding 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤5: 测试简单编码/解码")
try:
    test_text = "Hello, world!"
    tokens = encoding.encode(test_text)
    decoded = encoding.decode(tokens)
    print(f"   原文: {test_text}")
    print(f"   Token数: {len(tokens)}")
    print(f"   解码: {decoded}")
    print("   ✅ 编码/解码成功")
except Exception as e:
    print(f"   ❌ 编码/解码失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有 tiktoken 测试通过！")
print("=" * 80)
print("\n💡 如果上述测试都通过，问题可能在于:")
print("   1. transformers 和 tiktoken 的集成")
print("   2. PreTrainedTokenizer 的初始化")
print("   3. 其他模型组件的导入")
print("=" * 80)


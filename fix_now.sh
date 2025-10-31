#!/bin/bash
# 最简单的一键修复脚本

echo "🔧 正在修复 Segmentation Fault 问题..."
echo ""

# 方案1: 重新安装 tiktoken (最常见的解决方案)
echo "步骤1: 重新安装 tiktoken..."
pip uninstall -y tiktoken 2>/dev/null
pip install tiktoken==0.5.1 --quiet

# 测试
echo ""
echo "步骤2: 测试导入..."
if python diagnose_import.py 2>&1 | grep -q "所有组件导入成功"; then
    echo ""
    echo "✅ 修复成功！"
    exit 0
else
    echo ""
    echo "❌ 简单修复失败，尝试高级修复..."
    chmod +x auto_fix_segfault.sh
    ./auto_fix_segfault.sh
fi


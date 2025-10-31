#!/bin/bash
# -*- coding: utf-8 -*-
# 自动修复 Segmentation Fault 脚本

set -e

echo "================================================================================"
echo "🔧 Segmentation Fault 自动修复脚本"
echo "================================================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 测试函数
test_import() {
    log_info "测试导入..."
    if python diagnose_import.py > /tmp/test_import.log 2>&1; then
        log_info "✅ 导入测试通过！"
        return 0
    else
        log_error "❌ 导入测试失败"
        return 1
    fi
}

echo ""
echo "步骤1: 检查当前环境"
echo "--------------------------------------------------------------------------------"

log_info "Python 版本："
python --version

log_info "关键包版本："
pip show torch transformers tiktoken 2>/dev/null | grep -E "Name:|Version:" || log_warn "某些包未安装"

echo ""
echo "步骤2: 尝试修复方案1 - 更新 tiktoken"
echo "--------------------------------------------------------------------------------"

log_info "卸载旧版本 tiktoken..."
pip uninstall -y tiktoken 2>/dev/null || log_warn "tiktoken 未安装"

log_info "安装 tiktoken 0.5.1..."
pip install tiktoken==0.5.1 --no-cache-dir

log_info "验证安装..."
python -c "import tiktoken; print(f'tiktoken 版本: {tiktoken.__version__}')"

if test_import; then
    echo ""
    log_info "🎉 修复成功！问题已解决。"
    exit 0
fi

echo ""
echo "步骤3: 尝试修复方案2 - 尝试最新版 tiktoken"
echo "--------------------------------------------------------------------------------"

log_info "安装最新版 tiktoken..."
pip install --upgrade tiktoken --no-cache-dir

if test_import; then
    echo ""
    log_info "🎉 修复成功！问题已解决。"
    exit 0
fi

echo ""
echo "步骤4: 尝试修复方案3 - 更新 transformers"
echo "--------------------------------------------------------------------------------"

log_info "更新 transformers..."
pip install --upgrade transformers --no-cache-dir

if test_import; then
    echo ""
    log_info "🎉 修复成功！问题已解决。"
    exit 0
fi

echo ""
echo "步骤5: 尝试修复方案4 - 降级到稳定版本"
echo "--------------------------------------------------------------------------------"

log_info "降级到稳定版本组合..."
pip install transformers==4.36.0 tiktoken==0.5.1 --no-cache-dir

if test_import; then
    echo ""
    log_info "🎉 修复成功！问题已解决。"
    exit 0
fi

echo ""
echo "步骤6: 尝试修复方案5 - 从源码重新安装 tiktoken"
echo "--------------------------------------------------------------------------------"

log_info "从源码安装 tiktoken..."
pip uninstall -y tiktoken
pip install --no-binary tiktoken tiktoken==0.5.1

if test_import; then
    echo ""
    log_info "🎉 修复成功！问题已解决。"
    exit 0
fi

echo ""
echo "================================================================================"
log_error "❌ 所有自动修复方案均失败"
echo "================================================================================"
echo ""
echo "建议手动排查："
echo "  1. 运行: python diagnose_segfault.py"
echo "  2. 运行: python diagnose_tokenizer.py"
echo "  3. 查看: SEGFAULT_FIX_GUIDE.md"
echo ""
echo "可能需要："
echo "  - 检查 CUDA 版本兼容性"
echo "  - 重新安装 PyTorch"
echo "  - 检查系统库依赖"
echo ""
log_info "详细日志已保存到: /tmp/test_import.log"

exit 1


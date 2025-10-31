# Segmentation Fault 问题修复指南

## 问题描述
运行 `python diagnose_import.py` 时发生段错误（Segmentation fault），程序在"基础库导入"之后崩溃。

## 最可能的原因

### 1. tiktoken 库问题（90%可能性）
`tiktoken` 是一个 C 扩展库，在某些环境下会导致段错误。

**诊断方法：**
```bash
python diagnose_tokenizer.py
```

**解决方案：**
```bash
# 方案A: 重新安装特定版本
pip uninstall -y tiktoken
pip install tiktoken==0.5.1

# 方案B: 使用最新版本
pip install --force-reinstall tiktoken

# 方案C: 从源码安装
pip install --no-binary tiktoken tiktoken
```

### 2. Flash Attention 兼容性问题（5%可能性）
某些 CUDA 版本与 Flash Attention 不兼容。

**解决方案：**
```bash
# 设置环境变量禁用 Flash Attention
export KIMI_DISABLE_FLASH_ATTN=1
export CUDA_LAUNCH_BLOCKING=1
```

### 3. PyTorch/CUDA 版本不兼容（3%可能性）

**检查版本：**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

**常见兼容组合：**
- PyTorch 2.0.x + CUDA 11.7/11.8
- PyTorch 2.1.x + CUDA 11.8/12.1
- PyTorch 2.2.x + CUDA 12.1

### 4. transformers 库问题（2%可能性）

**解决方案：**
```bash
# 降级到稳定版本
pip install transformers==4.36.0

# 或升级到最新版本
pip install --upgrade transformers
```

## 逐步排查流程

### 第1步：运行细粒度诊断
```bash
python diagnose_segfault.py
```

这会告诉你在哪一步崩溃：
- 如果在步骤2-5崩溃：PyTorch/CUDA问题
- 如果在步骤6-7崩溃：transformers问题
- 如果在步骤8-10崩溃：tiktoken/tokenizer问题

### 第2步：专门测试 tiktoken
```bash
python diagnose_tokenizer.py
```

如果这一步崩溃，确认是 tiktoken 问题。

### 第3步：应用修复
根据上述诊断结果选择对应的解决方案。

### 第4步：使用安全版本（临时方案）
如果无法修复 tiktoken，可以暂时跳过 tokenizer：

```python
# 在 train_tmp.py 中
# 不导入 tokenizer
# from core.models.kimi_linear import KimiTokenizer  # 注释掉

# 只导入模型相关的类
from core.models.kimi_linear.configuration_kimi import KimiLinearConfig
from core.models.kimi_linear.modeling_kimi import (
    KimiLinearForCausalLM,
    KimiLinearModel,
    # 其他需要的类
)
```

## 快速修复命令（按优先级）

```bash
# 🔧 修复1: 更新 tiktoken（推荐）
pip uninstall -y tiktoken && pip install tiktoken==0.5.1

# 🔧 修复2: 设置环境变量
export TOKENIZERS_PARALLELISM=false
export KIMI_DISABLE_FLASH_ATTN=1
export CUDA_LAUNCH_BLOCKING=1

# 🔧 修复3: 更新所有相关库
pip install --upgrade torch transformers tiktoken tokenizers

# 🔧 修复4: 降级到稳定版本
pip install transformers==4.36.0 tiktoken==0.5.1

# 🔧 修复5: CPU模式测试（排查 CUDA 问题）
CUDA_VISIBLE_DEVICES="" python diagnose_import.py
```

## 验证修复

修复后运行：
```bash
python diagnose_import.py
```

应该看到：
```
================================================================================
✅ 所有组件导入成功！
================================================================================
```

## 预防措施

在 `train_tmp.py` 或其他训练脚本开头添加：

```python
import os
# 在导入任何库之前设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KIMI_DISABLE_FLASH_ATTN"] = "1"  # 如果有 Flash Attention 问题
```

## 相关文件

- `diagnose_import.py` - 基础导入诊断
- `diagnose_segfault.py` - 细粒度段错误诊断
- `diagnose_tokenizer.py` - tiktoken 专项测试
- `fix_segfault.py` - 自动修复脚本
- `tokenization_kimi_safe.py` - 安全版本的 tokenizer（延迟导入）

## 联系支持

如果上述方法都无效，请提供：
1. `python diagnose_segfault.py` 的完整输出
2. `python diagnose_tokenizer.py` 的完整输出
3. 系统信息：`uname -a`
4. Python 版本：`python --version`
5. 包版本：`pip list | grep -E "torch|transformers|tiktoken"`


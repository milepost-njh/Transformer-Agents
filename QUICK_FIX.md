# 🚨 Segmentation Fault 快速修复

## 问题：导入 Kimi 模型时发生段错误

## 🔧 快速解决方案（90%有效）

### 方案1：一键自动修复（推荐）

```bash
chmod +x auto_fix_segfault.sh
./auto_fix_segfault.sh
```

### 方案2：手动修复 tiktoken

```bash
# 重新安装 tiktoken
pip uninstall -y tiktoken
pip install tiktoken==0.5.1

# 测试
python diagnose_import.py
```

### 方案3：如果方案1-2无效，运行诊断

```bash
# 步骤1：细粒度诊断（找出在哪一步崩溃）
python diagnose_segfault.py

# 步骤2：tiktoken 专项测试
python diagnose_tokenizer.py

# 步骤3：根据输出结果查看完整指南
cat SEGFAULT_FIX_GUIDE.md
```

## 📝 临时绕过方案

如果急需训练但无法修复导入问题，可以临时不使用 tokenizer：

修改 `train_tmp.py`：

```python
# 注释掉 tokenizer 相关导入
# from core.models.kimi_linear import KimiTokenizer

# 只导入必要的模型组件
from core.models.kimi_linear.configuration_kimi import KimiLinearConfig
from core.models.kimi_linear.modeling_kimi import (
    KimiLinearForCausalLM,
    KimiLinearModel,
)

# 使用预先处理好的数据（已经 tokenized）
# 跳过 tokenizer 初始化步骤
```

## 🔍 了解更多

- **完整修复指南**：`SEGFAULT_FIX_GUIDE.md`
- **诊断脚本**：`diagnose_*.py`
- **自动修复脚本**：`auto_fix_segfault.sh`
- **安全版 tokenizer**：`core/models/kimi_linear/tokenization_kimi_safe.py`

## ✅ 验证修复

成功后应该看到：

```
================================================================================
✅ 所有组件导入成功！
================================================================================
```


# Kimi 模型训练诊断工具

本文件夹包含用于诊断和调试 Kimi 翻译模型训练问题的工具脚本。

## 📁 文件分类

### 训练数据检查
- **check_training_data.py** - 检查训练数据的格式和内容
- **debug_training_format.py** - 深度分析训练数据格式、tokenizer行为、loss计算范围
- **debug_token_225.py** - 调查特定token（225）的身份，比较分开编码vs一起编码的结果

### Labels 对齐诊断
- **debug_labels_shift.py** - 分析Causal LM中labels的shift机制
- **diagnose_labels_alignment.py** - 诊断labels对齐问题
- **verify_accuracy_bug.py** - 验证token_accuracy函数的bug和double-shift问题
- **verify_labels_fix.py** - 验证labels构造修复的正确性

### 模型测试
- **test_simple_inference.py** - 测试模型的基本预测能力（能否预测EOS、能否预测第一个英文词）
- **test_teacher_forcing.py** - 在teacher forcing条件下评估模型性能（关键诊断工具）

## 🔍 使用场景

这些脚本主要用于：
1. **调试训练问题** - 当模型训练出现异常时，使用这些工具定位问题
2. **验证修复** - 修复训练代码后，使用验证脚本确认修复正确
3. **理解机制** - 帮助理解Causal LM的训练机制和tokenizer行为

## 📝 历史背景

这些诊断工具是在修复Kimi模型训练过程中的关键bug时创建的：
- **Labels shift问题** - 发现并修复了labels构造错误
- **Token accuracy bug** - 发现并修复了accuracy计算错误
- **训练-推理不匹配** - 发现tokenizer编码不一致的问题

## ⚠️ 注意

这些是**诊断工具**，不是生产环境的推理代码。真正的推理代码在 `inference/` 文件夹下。


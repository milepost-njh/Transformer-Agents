# Kimi 模型训练与推理问题诊断与解决

本文档按时间顺序记录了在训练和推理 Kimi 翻译模型过程中遇到的关键问题及其解决方案。

## 📋 问题时间线

### 阶段一：训练初期问题发现（2025-11-02）

**现象**：
- 训练准确率极低（0.0000-0.07%），Loss 很高（9.0+）
- 大量梯度爆炸警告（Gradient norm > 10）
- 模型无法生成有意义的翻译结果

**诊断工具**：
- `01_check_training_data.py` - 检查训练数据格式
- `02_debug_training_format.py` - 深度分析训练数据格式、tokenizer行为

**发现的问题**：
1. 训练数据格式正确，但模型学习效果极差
2. 需要进一步检查模型训练机制

---

### 阶段二：Tokenizer 编码不一致问题（2025-11-03）

**现象**：
- 推理输出为重复的乱码（如 "make make make", "shy shy shy"）
- 即使训练准确率看起来不错，推理结果仍然很差

**诊断工具**：
- `03_debug_token_225.py` - 调查特定token编码问题

**发现的问题**：
- **训练-推理不匹配**：训练时分开编码 `"English: "` 和 `"I"` 然后拼接，与推理时直接编码 `"English: I"` 的结果不同
- Tokenizer 对空格和换行的处理导致 token 序列不一致

**解决方案**：
- 修改推理脚本，使用与训练时完全一致的编码方式（`add_special_tokens=False` + 手动添加 BOS）

---

### 阶段三：Labels 对齐问题（2025-11-08）

**现象**：
- 训练准确率虚高（显示100%），但模型实际没有学到任何东西
- Teacher forcing 测试显示 0% 准确率
- 模型无法预测第一个英文词

**诊断工具**：
- `04_debug_labels_shift.py` - 分析 Causal LM 中 labels 的 shift 机制
- `05_diagnose_labels_alignment.py` - 诊断 labels 对齐问题
- `06_test_teacher_forcing.py` - 在 teacher forcing 条件下测试模型（关键诊断工具）

**发现的问题**：
1. **Labels 构造错误**：
   ```python
   # 错误的构造方式
   labels = [-100] * prompt_len + en_ids + [eos_id]
   # 导致 labels[i] = input_ids[i]，模型无法学习从 prompt 预测第一个英文词
   ```

2. **Token Accuracy 函数 Bug**：
   - 没有考虑 Causal LM 中的 shift（`logits[i]` 预测 `token[i+1]`）
   - 导致准确率计算错误

**解决方案**：
1. 修复 labels 构造：向左 shift 一位
   ```python
   labels_unshifted = [-100] * prompt_len + en_ids + [eos_id]
   labels = labels_unshifted[1:] + [-100]  # 向左shift一位
   ```

2. 修复 token_accuracy 函数：由于 labels 已正确 shift，直接比较即可

**验证工具**：
- `07_verify_accuracy_bug.py` - 验证 accuracy bug 和 double-shift 问题
- `08_verify_labels_fix.py` - 验证 labels 构造修复的正确性

**修复效果**：
- 训练准确率从虚高的 100% 降至真实的 46%（Epoch 1）
- 模型开始真正学习翻译任务
- 验证集准确率逐步提升

---

### 阶段四：训练稳定性问题（2025-11-08）

**现象**：
- Epoch 4 开始准确率大幅下降（从 76% 降至 25%）
- 出现梯度爆炸警告

**诊断**：
- 分析训练日志发现学习率过大（2e-4）导致训练后期不稳定

**解决方案**：
- 降低学习率：从 `2e-4` → `5e-5`
- 增强梯度监控：阈值从 `5.0` → `3.0`

**修复效果**：
- 训练过程更稳定
- 准确率持续提升至 97%+（Epoch 11）

---

### 阶段五：推理验证（2025-11-09）

**现象**：
- 训练准确率达到 97%+，验证集准确率 82%+
- 推理结果基本正确，但部分长句仍有问题

**验证结果**：
```
Example 1: ✅ "I enjoy learning languages." (完美)
Example 2: ⚠️ "Tom is looking for a second opinion of the desk of the second child." (部分正确)
Example 3: ❌ "Which day. Is you are?" (错误)
```

**结论**：
- 模型已成功学习翻译任务
- 短句翻译质量高，长句仍需改进
- 训练和推理流程已基本正确

---

## 📁 诊断工具文件说明

### 数据检查类
- **`01_check_training_data.py`** - 检查训练数据格式和内容
- **`02_debug_training_format.py`** - 深度分析训练数据格式、tokenizer行为、loss计算范围

### Tokenizer 问题诊断
- **`03_debug_token_225.py`** - 调查 tokenizer 编码不一致问题（分开编码 vs 一起编码）

### Labels 对齐诊断
- **`04_debug_labels_shift.py`** - 分析 Causal LM 中 labels 的 shift 机制
- **`05_diagnose_labels_alignment.py`** - 诊断 labels 对齐问题
- **`07_verify_accuracy_bug.py`** - 验证 token_accuracy 函数的 bug
- **`08_verify_labels_fix.py`** - 验证 labels 构造修复的正确性

### 模型能力测试
- **`06_test_teacher_forcing.py`** - 在 teacher forcing 条件下评估模型性能（关键诊断工具）
- **`09_test_simple_inference.py`** - 测试模型的基本预测能力

---

## 🔧 关键修复总结

### 1. Labels 构造修复
**位置**：`train_kimi.py` 第 517-533 行

**修复前**：
```python
labels = [-100] * prompt_len + en_ids + [eos_id]
```

**修复后**：
```python
labels_unshifted = [-100] * prompt_len + en_ids + [eos_id]
labels = labels_unshifted[1:] + [-100]  # 向左shift一位
```

**原因**：在 Causal LM 中，`logits[i]` 预测的是 `input_ids[i+1]`，所以 `labels[i]` 应该等于 `input_ids[i+1]`。

### 2. 学习率调整
**位置**：`train_kimi.py` 第 2627 行

**修复前**：`learning_rate = 2e-4`

**修复后**：`learning_rate = 5e-5`

**原因**：学习率过大导致训练后期梯度爆炸，准确率崩溃。

### 3. 推理编码方式修复
**位置**：`inference/inference_kimi.py`

**修复**：使用与训练时完全一致的编码方式
```python
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = torch.tensor([[bos_id] + prompt_ids], ...)
```

---

## 📊 训练效果对比

| 阶段 | Epoch 1 Acc | Epoch 3 Acc | Epoch 11 Acc | 状态 |
|------|-------------|-------------|--------------|------|
| **修复前** | 100% (虚假) | 100% (虚假) | 崩溃 | ❌ 模型未学习 |
| **修复后** | 46.97% | 76.41% | 97.66% | ✅ 正常学习 |

---

## 🎯 最佳实践

1. **训练时**：
   - 使用正确的 labels shift（向左 shift 一位）
   - 使用保守的学习率（5e-5）
   - 监控梯度范数，及时调整

2. **推理时**：
   - 使用与训练时完全一致的 tokenizer 编码方式
   - 使用 `add_special_tokens=False` + 手动添加 BOS
   - 确保 attention_mask 正确

3. **调试时**：
   - 使用 teacher forcing 测试验证模型是否真正学习
   - 检查 labels 和 logits 的对齐关系
   - 验证训练-推理编码一致性

---

## ⚠️ 注意事项

1. 这些诊断工具是**临时调试工具**，不是生产代码
2. 真正的推理代码在 `inference/inference_kimi.py`
3. 所有关键修复已合并到主训练脚本 `train_kimi.py`

---

## 📚 参考资料

- Causal Language Model 训练机制
- PyTorch CrossEntropyLoss 文档
- Transformers Tokenizer 编码规范

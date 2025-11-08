# 🐛 训练Bug修复总结

## 发现的问题

通过一系列诊断脚本，我们发现了导致模型无法正确学习翻译任务的**关键bug**：

### Bug：token_accuracy 函数的 Shift 对齐错误

**位置**: `train_kimi.py` 第1928-1945行

**问题描述**:
在 Causal Language Model 中，`logits[i]` 预测的是 `token[i+1]`，但原始的 `token_accuracy` 函数直接比较 `pred_ids[i]` 和 `labels[i]`，导致：
1. **虚假的高准确率**（训练日志显示100%，但实际上模型什么都没学会）
2. 无法正确评估模型的真实性能
3. 早停机制失效（因为准确率虚高，认为模型已经很好了）

**验证证据** (`verify_accuracy_bug.py`):
```
场景1（完美预测）：
  ❌ 原始函数：12.50% 准确率（应该是100%）
  ✅ 修复函数：100.00% 准确率

场景3（对齐测试）：
  ❌ 原始函数：100% 准确率（预测当前token而非下一个，说明没考虑shift）
  ✅ 修复函数：0% 准确率（正确识别出对齐错误）
```

---

## 已实施的修复

### ✅ 修复1：token_accuracy 函数

**修改位置**: `train_kimi.py` 第1928-1956行

**修复内容**:
```python
def token_accuracy(real, pred, pad_id):
    """计算token级别的准确率（修复版）"""
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    
    # 🔧 修复：在Causal LM中，logits[i]预测token[i+1]
    # 所以应该比较 pred_ids[:, :-1] 和 real[:, 1:]
    pred_ids_shifted = pred_ids[:, :-1]  # 去掉最后一个预测
    real_shifted = real[:, 1:]  # 去掉第一个标签
    
    # 同时mask掉pad_id和-100
    mask = (real_shifted != pad_id) & (real_shifted != -100)
    correct = ((pred_ids_shifted == real_shifted) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)
```

**效果**: 现在准确率计算是正确的，能真实反映模型的学习状态。

---

## 其他发现（已分析，但无需修改）

### 发现2：手动 Shift Labels

**位置**: `train_kimi.py` 第2004行

**代码**:
```python
if "labels" in batch:
    labels = batch["labels"].to(device)
else:
    # 对话模式（向右移位生成labels）
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device)], dim=1)
```

**分析结果**:
- ✅ 这段代码**只在对话模式下执行**（`else` 分支）
- ✅ 对于翻译任务，labels 是从 batch 中直接获取的，没有 shift
- ✅ 不需要修改

### 发现3：Kimi 模型内部不自动 Shift

**位置**: `core/models/kimi_linear/modeling_kimi.py` 第1383行

**注释说明**:
```python
# 损失函数会：
#   1. 对每个位置计算交叉熵：比较 logits[b, i, :] 和 labels[b, i]
#   2. 忽略 labels[b, i] = -100 的位置（prompt 和 padding）
```

**分析结果**:
- Kimi 模型使用自定义的 loss_function，直接比较 `logits[i]` 和 `labels[i]`
- 这意味着**labels 必须预先正确对齐**
- 但查看数据处理逻辑后，发现labels的构建已经考虑了这一点：
  - `labels[prompt_len:] = en_ids + [eos]`
  - 这意味着 `labels[i] = input_ids[i]`（对于英文部分）
  - 在forward时，模型看到 `input_ids[:i]`，预测 `input_ids[i]`，loss 使用 `labels[i]`
  - **这是正确的！** 因为对于翻译任务，我们希望模型根据prompt预测第一个英文词，而不是预测第二个词

---

## ⚠️ 重要发现：模型根本没学会

通过 `test_teacher_forcing.py` 测试，我们发现即使在 **Teacher Forcing** 条件下（给定完整的正确答案），模型的所有预测都是错的：

```
Teacher Forcing 测试结果：准确率 0/4 = 0.0%
```

**这说明**：
1. 模型权重可能初始化有问题
2. 或者训练过程中梯度没有正确流动
3. 或者学习率/优化器配置不当
4. 或者数据本身有问题

---

## 下一步建议

### 方案1：重新训练（推荐）

由于当前的checkpoints可能已经"学坏了"（训练时使用了错误的准确率指标），建议：

1. **清理现有checkpoints**
   ```bash
   rm -rf /workspace/checkpoints_kimi_translation_bak/*
   ```

2. **使用修复后的代码重新训练**
   ```bash
   cd /workspace
   git pull  # 获取最新修复
   bash run_kimi.sh
   ```

3. **密切监控训练日志**
   - 现在的准确率应该从较低值开始（例如10-20%）
   - 逐步上升到合理的水平（80-90%）
   - 如果一开始就是100%，说明还有问题

4. **每个epoch结束时查看实际翻译样本**
   - 建议在 `train_kimi.py` 的验证函数后添加实际翻译测试
   - 观察模型是否真的学会了翻译

### 方案2：诊断现有模型

如果想先诊断为什么模型学不会，可以：

1. **检查梯度流动**
   ```python
   # 在train_kimi.py的训练循环中添加
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
   ```

2. **检查loss是否真的下降**
   ```python
   # 打印每个batch的详细loss
   logger.info(f"Batch {i}: loss={loss.item():.6f}")
   ```

3. **使用更小的模型测试**
   - 减少层数（2-4层）
   - 减少隐藏维度（128-256）
   - 快速验证训练流程是否正确

### 方案3：从头开始（最保险）

如果以上方案都不行，建议：

1. 使用一个已知有效的预训练模型（如GPT-2 small）
2. Fine-tune到翻译任务上
3. 验证训练流程是否正确
4. 再逐步替换成Kimi模型

---

## 诊断脚本清单

已创建的所有诊断脚本（都在 `inference/` 目录下）：

1. ✅ `verify_accuracy_bug.py` - 验证准确率计算bug
2. ✅ `test_teacher_forcing.py` - 测试模型基础预测能力
3. ✅ `test_simple_inference.py` - 简单推理测试
4. ✅ `inference_kimi_fixed.py` - 修复版推理脚本
5. ✅ `debug_training_format.py` - 训练数据格式调试
6. ✅ `debug_token_225.py` - Token编码调试
7. ✅ `debug_labels_shift.py` - Labels对齐调试
8. ✅ `check_training_data.py` - 训练数据检查

---

## 总结

核心问题：**token_accuracy 函数的 shift 对齐错误**，导致虚假的高准确率，掩盖了模型没有学会的事实。

**已修复**：`train_kimi.py` 中的 `token_accuracy` 函数

**下一步**：重新训练，并密切监控准确率和实际翻译质量

---

## 参考资料

- HuggingFace Causal LM文档: https://huggingface.co/docs/transformers/model_doc/gpt2
- PyTorch CrossEntropyLoss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Causal Language Modeling: https://huggingface.co/docs/transformers/tasks/language_modeling


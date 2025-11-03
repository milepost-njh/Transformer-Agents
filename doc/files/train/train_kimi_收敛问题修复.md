# Train_Kimi 收敛问题诊断与修复

## 问题现象

训练 `train_kimi.py` 时出现以下异常：
- ✅ 训练 Loss 正常下降：Epoch 1: 9.2→4.2, Epoch 2: 2.0→1.56
- ❌ 验证 Loss 反而上升：9.3→9.6 (说明过拟合)
- ❌ 准确率异常低：训练准确率 0.0040 (0.4%)，验证准确率 0.1306 (13.06%)

对比参考脚本 `train_rope_moe_data_parallel.py` 能够稳定收敛。

## 根因分析

通过代码对比发现了**3个关键Bug**：

### Bug 1: `loss_object` 的 `ignore_index` 设置错误

**位置**: 第 2913 行

**错误代码**:
```python
PAD_ID_TGT = tokenizer.pad_token_id  # = 1
loss_object = nn.CrossEntropyLoss(
    reduction="none", 
    ignore_index=PAD_ID_TGT,  # ❌ 错误：应该是 -100
    label_smoothing=label_smoothing
)
```

**问题说明**:
- 在翻译任务中，`labels` 使用 **-100** 来 mask prompt 部分（见 `build_filtered_translation_sequences` 函数第 495 行）
- 但 `loss_object` 设置了 `ignore_index=PAD_ID_TGT` (即 1)，导致无法正确忽略 prompt 部分
- PyTorch 的标准做法是使用 `-100` 作为 ignore_index

**修复**:
```python
loss_object = nn.CrossEntropyLoss(
    reduction="none", 
    ignore_index=-100,  # ✅ 标准做法
    label_smoothing=label_smoothing
)
```

### Bug 2: `token_accuracy` 函数没有 mask -100

**位置**: 第 1796 行

**错误代码**:
```python
@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    pred_ids = pred.argmax(dim=-1)
    mask = (real != pad_id)  # ❌ 只 mask pad_id，没有 mask -100
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)
```

**问题说明**:
- `labels` 中 prompt 部分都是 -100，应该被忽略
- 但只检查了 `pad_id`，导致计算准确率时把 prompt 部分的错误预测也算进去了
- 这就是为什么准确率异常低的原因

**修复**:
```python
@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    pred_ids = pred.argmax(dim=-1)
    # ✅ 同时 mask pad_id 和 -100
    mask = (real != pad_id) & (real != -100)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)
```

### Bug 3: 验证函数 `evaluate_on_val` 使用了错误的 labels

**位置**: 第 2238-2239 行

**错误代码**:
```python
for batch in val_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # ❌ 错误：使用对话模式的 labels（简单右移）
    labels = input_ids.clone()
    labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device)], dim=1)
```

**问题说明**:
- 验证集是翻译数据集，batch 中已经包含了正确的 `labels`（prompt 部分被 mask 为 -100）
- 但代码忽略了 batch 中的 labels，自己重新生成了一个简单右移的 labels
- 这导致验证时使用了**完全错误的labels**，验证 Loss 自然会很高

**修复**:
```python
for batch in val_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # ✅ 翻译模式：使用 batch 中提供的 labels
    if "labels" in batch:
        labels = batch["labels"].to(device)
    else:
        # 对话模式：如果 batch 没有 labels，则生成（向右移位）
        labels = input_ids.clone()
        labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device)], dim=1)
```

## 修复效果预期

修复这3个Bug后，应该能看到：

1. **训练 Loss 继续正常下降**（这个本来就是正常的）
2. **验证 Loss 显著降低**（因为使用了正确的 labels）
3. **准确率大幅提升**（因为正确 mask 了 -100 位置）
4. **模型能够收敛**（验证 Loss 应该下降而不是上升）

## 额外建议

### 1. 过拟合问题

从日志看，训练 Loss 下降很快但验证 Loss 上升，说明有过拟合趋势。建议：

- ✅ 已经使用了 `label_smoothing=0.1`
- 可以考虑增加 Dropout（目前 MoE 没有 dropout）
- 可以考虑减少训练轮数（从 20 降到 10）
- 可以考虑使用 Early Stopping

### 2. 梯度裁剪

日志显示大量梯度裁剪警告：
```
Large gradient norm detected: 14.6916
Applied additional gradient clipping to 0.1
```

当前策略：
- 第一次裁剪：`max_norm=0.5`
- 如果 `grad_norm > 5.0`，再次裁剪到 `max_norm=0.1`

这个策略是合理的，但可以考虑：
- 减少第二次裁剪的阈值（从 5.0 降到 3.0）
- 或者直接使用更小的第一次裁剪值（从 0.5 降到 0.3）

### 3. 学习率

当前使用：
- 初始学习率：`2e-4`
- Warmup steps：4000
- Scheduler：Cosine with warmup

这些设置看起来是合理的。

## 对比参考脚本

`train_rope_moe_data_parallel.py` 为什么能收敛？因为：

1. ✅ 使用了 Encoder-Decoder 架构，不需要 mask prompt
2. ✅ `token_accuracy` 函数只需要 mask `pad_id`（没有 -100 的情况）
3. ✅ 验证函数直接使用 `tar_real`（不会出错）

所以 `train_kimi.py` 需要特别处理 Decoder-Only 架构的翻译任务，这就是为什么会出现这些 Bug。

## 总结

核心问题是**Decoder-Only 架构做翻译任务时的特殊处理**：
- 需要用 -100 mask prompt 部分
- `ignore_index` 必须设为 -100
- `token_accuracy` 必须 mask -100
- 验证时必须使用 batch 中的 labels

这些在 Encoder-Decoder 架构中都不需要考虑，所以参考脚本没有这些问题。


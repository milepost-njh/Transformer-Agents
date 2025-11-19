# MTP 对比实验说明

## 问题总结

你遇到的问题是：推理时设置 `use_mtp=False` 会导致输出乱码，而 `use_mtp=True` 输出正常。

### 根本原因

1. **训练时** 使用了 `use_mtp=True`
2. **MTP 包装器** 会将原始 Transformer 保存为 `base_transformer` 属性
3. **Checkpoint** 中所有参数都带有 `base_transformer.` 前缀
4. **推理时** 如果 `use_mtp=False`，创建的是普通 Transformer（无前缀）
5. **结果**：参数名不匹配，权重加载失败，输出是随机的

## 修复方案

### 已修改的代码

**修改文件：** `inference/inference_ddp.py`

**修改内容：** 在 `load_checkpoint` 函数中添加了自动处理前缀不匹配的逻辑：
- 如果 checkpoint 有 `base_transformer.` 前缀但模型没有 → 自动移除前缀
- 如果 checkpoint 没有前缀但模型有 → 自动添加前缀

### 现在的行为

现在你可以：
- 设置 `use_mtp=False` 时，代码会自动移除前缀，正确加载权重
- 但会显示警告：**这不是真正的"非MTP效果"对比！**

## 关键理解

### ❌ 错误理解
"推理时改变 `use_mtp` 参数可以对比 MTP 的效果"

### ✅ 正确理解
- `use_mtp` 参数只控制**推理时是否创建 MTP 包装器**
- **不能**控制模型是否具有 MTP 训练的能力
- MTP 训练的模型已经学会了多步预测，这是内在的

### 为什么必须重新训练？

| 对比项 | MTP 训练 | 普通训练 |
|--------|----------|----------|
| **损失函数** | 同时预测多个未来 token | 只预测下一个 token |
| **优化目标** | L_total = L_next + α × L_future | L = L_next |
| **学到的能力** | 多步预测 | 单步预测 |
| **参数优化** | 受多步损失影响 | 只受单步损失影响 |

**结论：** 不能用 MTP 训练的模型测试"非 MTP 效果"！

## 正确的对比实验方案

### 方案 1：训练两个模型（推荐）

```python
# 训练配置 A：使用 MTP
use_mtp = True
checkpoint_a = "checkpoints/model_with_mtp.pt"

# 训练配置 B：不使用 MTP
use_mtp = False
checkpoint_b = "checkpoints/model_without_mtp.pt"
```

### 方案 2：对比推理速度/内存

如果你只是想测试 MTP 包装器的开销：

```python
# 都使用 MTP 训练的模型
checkpoint = "checkpoints_ddp_mla/mid_e16_s3552.pt"

# 测试 1：带 MTP 包装器推理
use_mtp = True
# ... 记录推理时间和内存

# 测试 2：不带 MTP 包装器推理（仅推理）
use_mtp = False  
# ... 记录推理时间和内存
```

**注意：** 方案 2 只能对比推理开销，不能对比翻译质量！

## 实验建议

### 如果你想对比翻译质量

**必须重新训练！** 步骤：

1. **训练 MTP 模型**（已完成）
   ```bash
   # train_ddp_latest.py 中设置
   use_mtp = True
   ```

2. **训练非 MTP 模型**（需要做）
   ```bash
   # train_ddp_latest.py 中设置
   use_mtp = False
   # 重新训练
   ```

3. **分别推理对比**
   ```bash
   # 推理 MTP 模型
   use_mtp = True
   checkpoint = "checkpoints/mtp_model.pt"
   
   # 推理非 MTP 模型
   use_mtp = False
   checkpoint = "checkpoints/non_mtp_model.pt"
   ```

### 如果你只想测试推理性能

可以直接使用现在的代码：

```python
# inference/inference_ddp.py
use_mtp = False  # 现在可以正常运行了
```

但记住：**翻译质量对比无意义！** 因为模型还是 MTP 训练的。

## 总结

### 当前状态
- ✅ 代码已修复，`use_mtp=False` 可以正常运行
- ⚠️ 但这不是真正的对比实验

### 下一步
- 如果要对比**翻译质量** → 必须重新训练一个 `use_mtp=False` 的模型
- 如果只是测试**推理性能** → 可以直接用当前代码

---

**问题？** 如有疑问，请随时提问！


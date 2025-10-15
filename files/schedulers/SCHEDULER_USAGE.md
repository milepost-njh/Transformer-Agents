# Scheduler使用说明

## 概述

现在 `train_moe_paraller_mla.py` 支持多种学习率调度器，可以通过修改 `scheduler_type` 参数来切换不同的调度策略。

## 可用的Scheduler类型

### 1. `"transformers_cosine"` (默认)
- **描述**: 使用transformers库的get_cosine_schedule_with_warmup
- **特点**: 线性warmup + 余弦衰减
- **适用场景**: 标准Transformer训练

### 2. `"cosine_warmup"`
- **描述**: 自定义实现的余弦warmup调度器
- **特点**: 与transformers版本类似，但使用本地实现
- **适用场景**: 需要自定义修改的场景

### 3. `"onecycle"`
- **描述**: OneCycle学习率调度器
- **特点**: 先上升后下降，收敛更快
- **适用场景**: 快速训练，MoE模型推荐

### 4. `"moe_cosine"`
- **描述**: 专门为MoE模型设计的余弦调度器
- **特点**: 针对MoE训练优化
- **适用场景**: MoE模型训练

### 5. `"reduce_on_plateau"`
- **描述**: 基于验证loss的自适应调度器
- **特点**: 当验证loss不再下降时自动降低学习率
- **适用场景**: 需要自适应调整的场景

## 如何切换Scheduler

在 `train_moe_paraller_mla.py` 中找到以下代码段：

```python
# Scheduler configuration
scheduler_type = "transformers_cosine"  # 修改这里
warmup_ratio = 0.15  # 15% of training steps for warmup
```

将 `scheduler_type` 修改为上述任一选项即可。

## 推荐配置

### 针对MoE模型梯度爆炸问题：

```python
scheduler_type = "onecycle"  # 推荐
learning_rate = 3e-4  # 降低学习率
warmup_ratio = 0.2  # 增加warmup比例
```

### 针对稳定训练：

```python
scheduler_type = "moe_cosine"  # MoE专用
learning_rate = 1e-4  # 保守学习率
warmup_ratio = 0.25  # 更长warmup
```

### 针对自适应调整：

```python
scheduler_type = "reduce_on_plateau"  # 自适应
learning_rate = 5e-4  # 初始学习率
warmup_ratio = 0.15  # 标准warmup
```

## 参数说明

- `learning_rate`: 最大学习率
- `warmup_ratio`: warmup阶段占总训练步数的比例
- `num_cycles`: 余弦衰减的周期数（仅对cosine类调度器有效）
- `div_factor`: OneCycleLR的初始学习率除数
- `final_div_factor`: OneCycleLR的最终学习率除数
- `patience`: ReduceLROnPlateau的耐心值
- `factor`: ReduceLROnPlateau的学习率衰减因子

## 日志输出

训练时会输出详细的scheduler配置信息：

```
Scheduler configuration:
  Type: onecycle
  Learning rate: 3e-4
  Warmup ratio: 0.2
  Total training steps: 27660
  Warmup steps: 5532
```

## 注意事项

1. **ReduceLROnPlateau**: 需要验证loss，会在每个epoch结束后调用scheduler.step(validation_loss)
2. **OneCycleLR**: 学习率变化较大，建议降低初始learning_rate
3. **MoE模型**: 建议使用更长的warmup_ratio (0.2-0.25)
4. **梯度爆炸**: 如果遇到梯度爆炸，建议使用OneCycleLR或降低learning_rate

## 实验建议

1. 先尝试 `"onecycle"` 解决梯度爆炸问题
2. 如果仍有问题，尝试 `"moe_cosine"` 并增加warmup_ratio
3. 对于长期训练，可以考虑 `"reduce_on_plateau"`
4. 记录不同scheduler的训练日志进行对比

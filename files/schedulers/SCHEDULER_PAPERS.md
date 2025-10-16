# 学习率调度器论文汇总

## 按时间顺序排列

### 1. **ReduceLROnPlateau** (2012年)
- **论文**: "Improving neural networks by preventing co-adaptation of feature detectors"
- **作者**: Geoffrey Hinton, et al.
- **会议**: 2012年提出
- **URL**: https://arxiv.org/abs/1207.0580
- **核心思想**: 基于验证指标的自适应学习率调整
- **代码类名**: `ReduceLROnPlateauWrapper`
- **实现位置**: `schedulers.py` 第128-163行

### 2. **Cosine Annealing** (2017年)
- **论文**: "SGDR: Stochastic Gradient Descent with Warm Restarts"
- **作者**: Ilya Loshchilov, Frank Hutter
- **会议**: ICLR 2017
- **URL**: https://arxiv.org/abs/1608.03983
- **核心思想**: 使用余弦函数进行学习率衰减，支持热重启
- **代码类名**: `CosineWarmupLR` (基于此论文改进)
- **实现位置**: `schedulers.py` 第16-62行

### 3. **Linear Warmup + Cosine Decay** (2017年)
- **论文**: "Attention Is All You Need" (Transformer论文)
- **作者**: Ashish Vaswani, et al.
- **会议**: NIPS 2017
- **URL**: https://arxiv.org/abs/1706.03762
- **核心思想**: 线性预热 + 余弦衰减，成为Transformer标准配置
- **代码类名**: `TransformersCosineWarmup`
- **实现位置**: `schedulers.py` 第207-236行

### 4. **OneCycleLR** (2017年)
- **论文**: "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- **作者**: Leslie N. Smith, Nicholay Topin
- **会议**: ICML 2018 (但2017年提出)
- **URL**: https://arxiv.org/abs/1708.07120
- **核心思想**: 单周期学习率调度，先上升后下降
- **代码类名**: `OneCycleLRCustom`
- **实现位置**: `schedulers.py` 第65-125行

### 5. **MoE-specific Schedulers** (2020年+)
- **论文**: "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
- **作者**: William Fedus, et al.
- **会议**: 2021年
- **URL**: https://arxiv.org/abs/2101.03961
- **核心思想**: 针对MoE模型的特殊学习率调度需求
- **代码类名**: `MoEWarmupCosineLR`
- **实现位置**: `schedulers.py` 第166-204行

## 代码实现说明

### 工厂函数
- **函数名**: `create_scheduler`
- **实现位置**: `schedulers.py` 第239-315行
- **功能**: 统一创建不同类型调度器的工厂函数

### 支持的调度器类型
```python
scheduler_types = [
    "transformers_cosine",  # TransformersCosineWarmup
    "cosine_warmup",       # CosineWarmupLR  
    "onecycle",            # OneCycleLRCustom
    "reduce_on_plateau",   # ReduceLROnPlateauWrapper
    "moe_cosine",          # MoEWarmupCosineLR
]
```

## 使用示例

### 在训练脚本中的使用
```python
# 文件: train_moe_paraller_mla.py
# 位置: 第2048-2055行

# 调度器配置
scheduler_type = "moe_cosine"  # 选项: "transformers_cosine", "cosine_warmup", "onecycle", "reduce_on_plateau", "moe_cosine"
warmup_ratio = 0.15  # 15%的训练步数用于预热

# 创建调度器
scheduler = create_scheduler(
    scheduler_type=scheduler_type,
    optimizer=optimizer,
    num_training_steps=num_training_steps,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    num_cycles=0.5,
    # 特定调度器的额外参数
    div_factor=25.0,  # OneCycleLR使用
    final_div_factor=1e4,  # OneCycleLR使用
    patience=3,  # ReduceLROnPlateau使用
    factor=0.5,  # ReduceLROnPlateau使用
    min_lr=1e-6,  # 最小学习率
)
```

## 推荐使用场景

### 1. **MoE模型训练** (推荐)
```python
scheduler_type = "moe_cosine"
# 专门为MoE模型设计，解决梯度爆炸问题
```

### 2. **快速收敛**
```python
scheduler_type = "onecycle"
# 基于OneCycleLR论文，收敛速度快
```

### 3. **稳定训练**
```python
scheduler_type = "transformers_cosine"
# 基于Transformer论文，稳定可靠
```

### 4. **自适应调整**
```python
scheduler_type = "reduce_on_plateau"
# 基于验证loss自动调整学习率
```

## 技术细节

### 学习率计算公式

#### Cosine Annealing
```
lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(π * t / T)) / 2
```

#### Linear Warmup + Cosine Decay
```
if t < warmup_steps:
    lr = lr_max * t / warmup_steps
else:
    lr = lr_max * cos(π * (t - warmup_steps) / (total_steps - warmup_steps))
```

#### OneCycleLR
```
# 阶段1: 线性上升 (0 → max_lr)
# 阶段2: 余弦下降 (max_lr → min_lr)
```

## 参考文献

1. Hinton, G. E., et al. (2012). "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580.

2. Loshchilov, I., & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017.

3. Vaswani, A., et al. (2017). "Attention Is All You Need." NIPS 2017.

4. Smith, L. N., & Topin, N. (2017). "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." ICML 2018.

5. Fedus, W., et al. (2021). "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." arXiv preprint arXiv:2101.03961.

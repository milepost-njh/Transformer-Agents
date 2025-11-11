# KV-cache 使用教程

> **更新日期**: 2025-10-28 | **状态**: ✅ 完整实现

## 概述

本教程介绍如何在Transformer模型中使用KV-cache进行高效推理。

## ✅ 好消息：无需重新训练！

您现有的模型（如 `checkpoints/mid_e1_s222.pt`）可以**直接**使用KV-cache进行推理，无需任何重新训练。

### 为什么不需要重新训练？

1. **KV-cache是推理优化**：只改变计算方式，不改变模型权重
2. **向后兼容**：添加的参数都有默认值（`use_cache=False`）
3. **训练行为未变**：训练时的计算逻辑与之前完全一样

## 快速开始（3步）

### 步骤1: 确认环境

确保您的环境已安装：
- PyTorch ≥ 1.12
- transformers
- loguru

```bash
# 检查必要文件
ls checkpoints/mid_e1_s222.pt  # 模型checkpoint
ls tok_pt/tokenizer.json       # 葡语tokenizer
ls tok_en/tokenizer.json       # 英语tokenizer
```

### 步骤2: 运行推理

```bash
# 单模型测试（真正cache vs 模拟cache，自动验证正确性）
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --test_lengths 128
```

### 步骤3: 查看性能提升

从输出可以看到真正的KV-cache相比模拟cache的加速效果：
- 推理速度提升：2-5x（序列越长提升越大）
- MLA显存节省：约34%

---

## 什么是KV-cache？

KV-cache（Key-Value cache）是一种推理优化技术，通过缓存历史token的K和V向量，避免重复计算，从而：

- **提升推理速度**：长序列生成可提速 2-5x
- **节省显存**：MLA模式下可节省 34% 显存
- **保持准确性**：输出结果与不使用cache完全一致

### Prefill vs Decode

推理分为两个阶段：

1. **Prefill阶段**（第一步）：
   - 输入完整prompt
   - 一次性计算所有token的K/V
   - 生成初始cache

2. **Decode阶段**（后续步骤）：
   - 每步只输入1个新token
   - 复用历史K/V cache
   - 拼接新K/V到cache

## 快速开始

### 1. 基本推理

```python
from inference.compare_kv_cache_mla import KVCacheInferenceEngine, load_model_and_tokenizers

# 加载模型
model, pt_tokenizer, en_tokenizer = load_model_and_tokenizers(
    checkpoint_path="checkpoints/mid_e1_s222.pt",
    use_mla=True,  # 使用MLA
    device="cuda"
)

# 创建推理引擎
engine = KVCacheInferenceEngine(
    model=model,
    pt_tokenizer=pt_tokenizer,
    en_tokenizer=en_tokenizer,
    device="cuda"
)

# 生成文本（使用真正的KV-cache）
result = engine.generate_with_kv_cache(
    input_text="O Tom está procurando uma segunda opinião.",
    max_new_tokens=64,
    use_real_cache=True  # 启用真正的KV-cache
)

print(f"输出: {result['output']}")
print(f"生成时间: {result['generation_time']:.3f}s")
print(f"速度: {result['tokens_per_second']:.2f} tokens/s")
print(f"KV-cache显存: {result['total_cache_memory_mb']:.2f} MB")
```

### 2. 对比测试（真正cache vs 模拟cache）

```python
# 模拟cache（无加速）
result_simulated = engine.generate_with_kv_cache(
    input_text="输入文本",
    max_new_tokens=64,
    use_real_cache=False  # 模拟cache
)

# 真正的cache（有加速）
result_real = engine.generate_with_kv_cache(
    input_text="输入文本",
    max_new_tokens=64,
    use_real_cache=True  # 真正的cache
)

# 计算加速比
speedup = result_simulated['generation_time'] / result_real['generation_time']
print(f"加速比: {speedup:.2f}x")
```

## Cache类型选择指南

### DynamicCache - 推荐用于大多数推理场景

**优点**：
- 自动增长，无需预设最大长度
- 使用简单，内存占用随序列增长
- 适合序列长度不确定的场景

**缺点**：
- 不支持torch.compile优化
- 频繁拼接可能略慢

**使用场景**：
- 交互式对话
- 长文本生成
- 序列长度变化大的任务

```python
from core.models.kv_cache import DynamicCache

# 自动管理，无需配置
cache = DynamicCache()
```

### StaticCache - 用于最快速度

**优点**：
- 预分配显存，速度最快
- 支持torch.compile优化
- 使用index_copy_高效更新

**缺点**：
- 需要预设最大长度
- 浪费未使用的显存

**使用场景**：
- 固定长度生成
- 追求极致性能
- 配合torch.compile使用

```python
from core.models.kv_cache import StaticCache

cache = StaticCache(
    num_layers=8,           # 模型层数
    num_heads=8,            # 注意力头数
    head_dim=64,            # 头维度
    max_batch_size=4,       # 最大批次
    max_cache_len=512,      # 最大序列长度
    device="cuda",
    dtype=torch.bfloat16
)
```

### MLACache - MLA模型专用

**优点**：
- 压缩KV表示，节省34%显存
- 支持动态和静态两种模式
- 保持MLA的高效性

**使用场景**：
- 使用MLA模型时必选
- 超长序列生成
- 显存受限场景

```python
from core.models.kv_cache import MLACache

# 动态模式
cache = MLACache(
    num_layers=8,
    kv_lora_rank=128,       # KV压缩维度
    qk_rope_head_dim=32,    # RoPE维度
    num_heads=8,
    v_head_dim=64,
    dynamic=True            # 动态增长
)

# 静态模式（更快）
cache = MLACache(
    num_layers=8,
    kv_lora_rank=128,
    qk_rope_head_dim=32,
    num_heads=8,
    v_head_dim=64,
    max_batch_size=4,
    max_cache_len=512,
    device="cuda",
    dtype=torch.bfloat16,
    dynamic=False           # 预分配
)
```

## 性能优化建议

### 1. 选择合适的dtype

```python
# 推荐使用bfloat16（如果硬件支持）
cache = StaticCache(..., dtype=torch.bfloat16)

# 或使用float16
cache = StaticCache(..., dtype=torch.float16)

# float32精度最高但慢且占用显存
cache = StaticCache(..., dtype=torch.float32)
```

### 2. 批量推理

```python
# 增大batch_size以充分利用GPU
cache = StaticCache(
    ...,
    max_batch_size=8  # 根据显存调整
)
```

### 3. 配合torch.compile（需PyTorch 2.0+）

```python
import torch

# 使用StaticCache
cache = StaticCache(...)

# 编译模型
compiled_model = torch.compile(model, mode="reduce-overhead")

# 推理时会自动优化
```

### 4. 预热（Warmup）

```python
# 首次推理前进行预热
for _ in range(3):
    _ = engine.generate("预热文本", max_new_tokens=10, use_cache=True)
    
# 正式推理
result = engine.generate("正式输入", max_new_tokens=64, use_cache=True)
```

## 性能测试

### 单模型测试

```bash
# 测试MLA模型，对比真正cache vs 模拟cache
python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --test_lengths 64 128
```

### MLA vs 标准注意力对比

```bash
# 对比MLA和标准注意力的KV-cache效率
python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths 32 64 128
```

## 常见问题

### Q1: 使用cache后输出是否会改变？

**A**: 不会。使用cache和不使用cache的输出在数值上完全一致（浮点误差在1e-5以内）。

```bash
# 验证正确性：运行对比测试
python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --test_lengths 64
# 会自动对比真正cache和模拟cache的输出
```

### Q2: 什么时候使用cache最有效？

**A**: 序列越长，cache的效果越明显：
- 16 tokens: 加速约 1.2x
- 64 tokens: 加速约 2-3x
- 128+ tokens: 加速约 3-5x

### Q3: cache会占用多少显存？

**A**: 标准注意力的KV-cache大小计算：
```
显存 = 2 × num_layers × num_heads × head_dim × seq_len × 2 bytes (bf16)
```

示例（8层，8头，64维，128序列）：
```
2 × 8 × 8 × 64 × 128 × 2 bytes = 2.1 MB
```

MLA模式下节省约34%。

### Q4: 训练时可以使用cache吗？

**A**: 可以，但通常不推荐：
- 训练时有完整序列，不需要cache
- 使用`use_cache=False`（默认）即可
- cache主要用于自回归推理

### Q5: 如何调试cache问题？

**A**: 运行对比测试验证正确性：

```bash
# 对比真正cache vs 模拟cache
python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --test_lengths 64

# 输出会显示两种方式的结果是否一致
```

## 最佳实践总结

1. **推理时始终启用cache**（use_real_cache=True）
2. **根据场景选择cache类型**：
   - 不确定长度 → DynamicCache（默认）
   - 固定长度 → StaticCache
   - MLA模型 → MLACache（自动）
3. **使用bf16精度**（如果硬件支持）
4. **预热后再正式测试**性能
5. **监控显存使用**，避免OOM
6. **定期运行对比测试**确保正确性

## 相关文档

本文档所在位置：`doc/files/inference/KV-cache使用教程.md`

同一目录下的相关文档：
- [README](README.md) - 📖 文档索引和导航
- [KV-cache实现指南](KV-cache实现指南.md) - 🔧 技术细节和实现原理
- [KV-cache存储量计算](KV-cache存储量计算.md) - 🧮 显存占用计算公式
- [精度统一说明](精度统一说明.md) - 🎯 bf16/fp16精度说明

## 技术支持

遇到问题？

1. 查看[实现指南](KV-cache实现指南.md)的FAQ部分
2. 运行`compare_kv_cache_mla.py`对比测试验证环境
3. 检查模型checkpoint是否正确加载
4. 确认PyTorch版本 ≥ 1.12（推荐 2.0+）


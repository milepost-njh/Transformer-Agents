# KV-cache 实现说明

> **类型**: 技术说明 | **难度**: 高级

## 概述

KV-cache 是推理优化的核心技术。本文档说明当前项目的实现方式。

## 当前实现

### 核心思想

在自回归生成中，每个新 token 的注意力计算只需要当前 token 的 Q，但需要所有历史 token 的 K/V。通过缓存 K/V，可以避免重复计算。

**术语**:
- `L` = Transformer 层数
- `H` = 注意力头数
- `d_h` = 头维度
- `S` = 序列长度

**标准注意力 (MHA)**:
- 每 step 缓存: K/V 各 `[B, H, S, d_h]`
- 单token显存: `2 × H × d_h × L × 2bytes`

**MLA 优化**:
- 每 step 缓存: 压缩的 K/V
- 单token显存: `(d_kv_compressed + d_rope + H×v_head_dim) × L × 2bytes`
- 压缩比: 34.0%（64 token 场景）

### 代码实现

**MultiHeadAttention forward 方法**（`train_tmp.py` 759-916行）:

```python
def forward(self, q, k, v, mask=None, return_attn=True, 
            past_key_value=None, use_cache=False):
    """
    支持 KV-cache 的前向传播
    
    Args:
        past_key_value: 缓存的 (past_k, past_v) 元组
        use_cache: 是否返回当前的 KV-cache
    """
    # ... 注意力计算 ...
    
    # KV-cache 处理
    if past_key_value is not None:
        past_k, past_v = past_key_value
        key_states = torch.cat([past_k, key_states], dim=2)  # 沿序列维拼接
        v_states = torch.cat([past_v, v_states], dim=2)
    
    # 返回当前 KV 用于下一步
    present_key_value = (key_states, v_states) if use_cache else None
    
    # 返回值
    if use_cache:
        if return_attn:
            return output, attn_weights, present_key_value
        return output, present_key_value
    else:
        if return_attn:
            return output, attn_weights
        return output
```

### 自回归生成流程

```python
def generate_with_kv_cache(self, input_text: str, max_new_tokens: int = 64):
    """自回归生成，使用 KV-cache"""
    
    kv_cache = None  # 初始化缓存
    step_times = []
    kv_cache_sizes = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            start = time.time()
            
            # 前向传播（传入历史 KV-cache）
            output, kv_cache = self.model(
                query, key, value,
                past_key_value=kv_cache,
                use_cache=True
            )
            
            # 记录显存
            kv_size = self._calculate_cache_size(kv_cache)
            kv_cache_sizes.append(kv_size)
            step_times.append(time.time() - start)
    
    return {
        'kv_cache_sizes': kv_cache_sizes,
        'max_kv_cache_size': max(kv_cache_sizes),
        'step_times': step_times
    }
```

## 实现状态说明

### 阶段一：模拟KV-cache（显存测量工具）

**目的**：测量和对比 MLA vs 标准注意力的显存节省效果

#### 已实现功能（阶段一）

| 功能 | 状态 | 说明 |
|------|------|------|
| **底层注意力层支持** | ✅ 已实现 | `MultiHeadAttention` 支持 `past_key_value` 和 `use_cache` |
| **显存计算** | ✅ 已实现 | `KVCacheTracker` 可准确计算理论显存占用 |
| **对比测试** | ✅ 已实现 | 可对比 MLA vs 标准注意力的显存差异（34.4% 节省） |

#### 限制（阶段一）

| 功能 | 状态 | 说明 |
|------|------|------|
| **上层模型传递** | ❌ 未实现（阶段一） | 当时`Transformer`/`Decoder`/`DecoderLayer` 未传递 KV-cache 参数 |
| **计算复用** | ❌ 未实现（阶段一） | 每步重新计算整个序列，未复用历史 K/V |
| **真正的推理加速** | ❌ 未实现（阶段一） | 无法节省计算时间，只能测量显存 |

**阶段一成果**：成功验证MLA相比标准注意力可节省 **34.4%** KV-cache显存。

---

### 阶段二：真正的KV-cache（推理加速）✅

**更新日期**: 2025-10-28  
**状态**: ✅ 已完整实现

**目的**：实现真正的推理加速和显存优化，支持生产环境使用

#### 已实现功能（阶段二）

| 功能 | 状态 | 说明 |
|------|------|------|
| **Cache基础设施** | ✅ 已实现 | `DynamicCache`, `StaticCache`, `MLACache` 三种Cache类型 |
| **底层注意力层支持** | ✅ 已实现 | `MultiHeadAttention` 完整支持 `past_key_value` 和 `use_cache` |
| **上层模型传递** | ✅ 已实现 | `Transformer`/`Encoder`/`Decoder`/各Layer 完整支持 KV-cache 参数传递 |
| **计算复用** | ✅ 已实现 | Prefill阶段运行一次encoder，Decode阶段逐token复用历史KV |
| **真正的推理加速** | ✅ 已实现 | 节省计算时间和显存，长序列提速2-5x |
| **显存计算** | ✅ 已实现 | `KVCacheTracker` 可准确计算理论显存占用 |
| **对比测试** | ✅ 已实现 | 可对比 MLA vs 标准注意力的显存差异 |
| **性能对比测试** | ✅ 已实现 | `compare_kv_cache_mla.py` 真正cache vs 模拟cache对比 |

#### 特性支持（阶段二）

| 特性 | 支持情况 |
|------|----------|
| **标准注意力** | ✅ 完全支持 |
| **MLA (Multi-head Latent Attention)** | ✅ 完全支持（压缩KV格式） |
| **RoPE位置编码** | ✅ 完全支持 |
| **MoE (Mixture of Experts)** | ✅ 完全支持 |
| **训练模式** | ✅ 支持（use_cache=False） |
| **推理模式** | ✅ 支持（use_cache=True） |
| **批量推理** | ✅ 支持 |

#### 重要说明：模型兼容性

**✅ 无需重新训练！** 

使用阶段一代码训练的模型（如 `checkpoints/mid_e1_s222.pt`）可以直接用于阶段二的KV-cache推理：

- ✅ 模型权重完全兼容
- ✅ KV-cache只是推理优化，不改变模型参数
- ✅ 训练时的行为（use_cache=False）与之前完全一样
- ✅ 只需加载checkpoint，使用新的推理脚本即可

**原因**：
- KV-cache是**推理时的计算优化**，不是模型结构的改变
- 添加的`use_cache`参数默认为False，保持向后兼容
- 训练代码的修改只是为了接口统一，实际行为未变

## 使用示例

### 基本使用

```python
from inference.compare_kv_cache_mla import KVCacheInferenceEngine, load_model_and_tokenizers

# 加载模型
model, pt_tokenizer, en_tokenizer = load_model_and_tokenizers(
    checkpoint_path="checkpoints/mid_e1_s222.pt",
    use_mla=True,  # 使用MLA模型
    device="cuda"
)

# 创建推理引擎
engine = KVCacheInferenceEngine(model, pt_tokenizer, en_tokenizer, device="cuda")

# 使用真正的KV-cache生成
result = engine.generate_with_kv_cache(
    input_text="O Tom está procurando uma opinião médica.",
    max_new_tokens=64,
    use_real_cache=True  # True=真正的cache，False=模拟cache
)

print(f"输出: {result['output']}")
print(f"速度: {result['tokens_per_second']:.2f} tokens/s")
print(f"KV-cache显存: {result['total_cache_memory_mb']:.2f} MB")
```

### 性能对比测试

```bash
# 单模型测试：对比真实KV-cache vs 模拟KV-cache
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/latest.pt \
    --test_lengths 128

# MLA vs 标准注意力对比
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths 128
```

### Cache类型选择

**DynamicCache** - 推荐用于推理：
```python
from core.models.kv_cache import DynamicCache

cache = DynamicCache()
# 自动增长，使用简单，适合大多数推理场景
```

**StaticCache** - 用于torch.compile优化：
```python
from core.models.kv_cache import StaticCache

cache = StaticCache(
    num_layers=8,
    num_heads=8,
    head_dim=64,
    max_batch_size=4,
    max_cache_len=512,
    device="cuda",
    dtype=torch.bfloat16
)
# 预分配显存，支持torch.compile，速度最快
```

**MLACache** - MLA模型专用：
```python
from core.models.kv_cache import MLACache

cache = MLACache(
    num_layers=8,
    kv_lora_rank=128,
    qk_rope_head_dim=32,
    num_heads=8,
    v_head_dim=64,
    dynamic=True  # 或 False (静态模式)
)
# 压缩格式，节省34%显存
```

---

## 显存效率 (bf16 精度)

**64 token 生成对比** (L=8, H=8, d_h=64):

| 模型 | 单token | S=64 | 推理时间 | 每步时间 |
|------|--------|------|---------|---------|
| 标准 | 16 KB | 1.00 MB | 2.344s | 36.57ms |
| MLA | 10.8 KB | 0.66 MB | 2.667s | 41.61ms |
| **节省** | **-32.5%** | **-34.0%** | **-13.8%** | **-13.8%** |

## 关键实现细节

1. **past_key_value 格式**: 元组 `(past_k, past_v)` 存储缓存
2. **拼接维度**: `dim=2` 对应序列维
3. **精度统一**: bf16（2 bytes/参数）
4. **梯度处理**: 推理使用 `torch.no_grad()`

## 测试脚本

运行 KV-cache 对比测试:
```bash
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths 64
```

## 相关文档

本文档所在位置：`doc/files/inference/KV-cache实现指南.md`

同一目录下的相关文档：
- [README](README.md) - 📖 文档索引和导航
- [KV-cache使用教程](KV-cache使用教程.md) - 📚 详细的使用指南（推荐先看）
- [KV-cache存储量计算](KV-cache存储量计算.md) - 🧮 显存占用计算公式详解
- [精度统一说明](精度统一说明.md) - 🎯 bf16/fp16精度说明

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

## 实现状态说明 ⚠️

### 当前实现：显存测量（模拟 KV-cache）

**重要**：`compare_kv_cache_mla.py` 中的实现是**显存测量工具**，而非完整的 KV-cache 推理优化。

#### 已实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| **底层注意力层支持** | ✅ 已实现 | `MultiHeadAttention` 支持 `past_key_value` 和 `use_cache` |
| **显存计算** | ✅ 已实现 | `KVCacheTracker` 可准确计算理论显存占用 |
| **对比测试** | ✅ 已实现 | 可对比 MLA vs 标准注意力的显存差异 |

#### 未实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| **上层模型传递** | ❌ 未实现 | `Transformer`/`Decoder`/`DecoderLayer` 未传递 KV-cache 参数 |
| **计算复用** | ❌ 未实现 | 每步重新计算整个序列，未复用历史 K/V |
| **真正的推理加速** | ❌ 未实现 | 无法节省计算时间，只能测量显存 |

#### 真正的 KV-cache vs 当前的模拟 KV-cache

**真正的 KV-cache**（未实现）：
```python
# 第一步：计算 token 0 的 K, V
past_kv = compute_kv(token_0)  # 缓存起来

# 第二步：只计算 token 1 的 K, V
new_kv = compute_kv(token_1)
full_kv = concat(past_kv, new_kv)  # ✅ 复用历史，节省计算

# 第三步：只计算 token 2 的 K, V
new_kv = compute_kv(token_2)
full_kv = concat(full_kv, new_kv)  # ✅ 继续复用
```

**优势**：
- ✅ 每步只计算 1 个新 token 的 K/V
- ✅ 复用历史 token 的 K/V
- ✅ 节省**计算时间**和**显存**

---

**当前的模拟 KV-cache**（已实现）：
```python
# 第一步：计算 token 0 的 K, V
kv = compute_kv([token_0])

# 第二步：重新计算 token 0-1 的 K, V（全部重算）
kv = compute_kv([token_0, token_1])  # ❌ 没有复用

# 第三步：重新计算 token 0-2 的 K, V（全部重算）
kv = compute_kv([token_0, token_1, token_2])  # ❌ 没有复用
```

**现状**：
- ❌ 每步重新计算整个序列的 K/V
- ✅ 但通过 `KVCacheTracker` **理论计算** KV-cache 应该占用的显存
- ✅ 用于对比 MLA vs 标准注意力的**显存占用差异**（34.4% 节省）

#### 为什么 encoder_cache 和 decoder_cache 没被用到？

在 `compare_kv_cache_mla.py` 第 180-184 行：
```python
# 编码器的KV-cache（只需要计算一次）
encoder_cache = None

# 解码器的KV-cache（每步更新）
decoder_cache = None
```

这些变量是**预留的**，但因为：
1. `Transformer.forward()` 不接受 `past_key_value` 参数（第 1236 行）
2. `Decoder.forward()` 不接受 `past_key_value` 参数（第 1157 行）
3. `DecoderLayer.forward()` 未传递 KV-cache 到注意力层

所以这些变量无法使用，当前只能通过 `KVCacheTracker` 计算理论显存。

#### 实现真正 KV-cache 的改动范围

要实现真正的 KV-cache 推理加速，需要修改：

1. **Transformer.forward()**：添加 `past_key_values` 参数
2. **Decoder.forward()**：添加并传递 `past_key_values`
3. **DecoderLayer.forward()**：传递 KV-cache 到注意力层
4. **生成循环**：每步只输入新 token，复用历史 cache

**当前实现的目的**：**测量和对比 MLA 的显存节省效果**（34.4%），而非推理加速。

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

- **计算公式详解**: `doc/files/inference/KV-cache存储量计算.md`
- **精度说明**: `doc/files/inference/精度统一说明.md`

# KV-cache 实现说明

> **类型**: 技术说明 | **难度**: 高级

## 概述

KV-cache 是推理优化的核心技术。本文档说明当前项目的实现方式。

## 当前实现

### 核心思想

在自回归生成中，每个新 token 的注意力计算只需要当前 token 的 Q，但需要所有历史 token 的 K/V。通过缓存 K/V，可以避免重复计算。

**标准注意力**:
- 每 step 缓存完整的 K/V：`[B, H, seq_len, head_dim]`
- 存储大小：`2 × H × head_dim × seq_len × num_layers`

**MLA 优化**:
- 缓存压缩的 K/V：`[B, H, seq_len, compressed_dim]`  
- 存储大小：`(kv_lora_rank + qk_rope_head_dim + v_head_dim) × seq_len × num_layers`
- 压缩比：34.0%（64 token 场景）

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
        key_states = torch.cat([past_k, key_states], dim=2)  # 拼接历史 K
        v_states = torch.cat([past_v, v_states], dim=2)      # 拼接历史 V
    
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
            
            # 记录 KV-cache 大小
            kv_size = self._calculate_cache_size(kv_cache)
            kv_cache_sizes.append(kv_size)
            step_times.append(time.time() - start)
    
    return {
        'kv_cache_sizes': kv_cache_sizes,
        'max_kv_cache_size': max(kv_cache_sizes),
        'step_times': step_times
    }
```

## 显存效率

**64 token 生成对比** (bf16 精度):

| 模型 | KV-cache | 推理时间 | 每步时间 |
|------|----------|---------|---------|
| 标准注意力 | 1.00 MB | 2.344s | 36.57ms |
| MLA | 0.66 MB | 2.667s | 41.61ms |
| **节省** | **34.0%** | **-13.8%** | **-13.8%** |

**长序列优势**: 序列越长，MLA 的显存节省相对于速度损失的收益越明显。

## 关键实现细节

1. **past_key_value 格式**: 使用元组 `(past_k, past_v)` 存储缓存
2. **序列维度拼接**: `dim=2` 对应序列长度维度
3. **显存计算**: 使用 bf16（2 bytes/参数）统一计算
4. **梯度处理**: 推理时使用 `torch.no_grad()` 避免计算梯度

## 测试脚本

运行 KV-cache 对比测试:
```bash
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths 64
```

输出示例:
```
💾 KV-cache: 0.66 MB (MLA) vs 1.00 MB (Standard) | 节省 34.0%
⏱️  推理时间: 2.667s (MLA) vs 2.344s (Standard) | 慢 13.8%
```

## 相关文档

- **详细对比报告**: `doc/files/inference/KV-cache存储量计算.md`
- **精度说明**: `doc/files/inference/精度统一说明.md`
- **代码位置**: `train_tmp.py`(759-916行) | `inference/compare_kv_cache_mla.py`

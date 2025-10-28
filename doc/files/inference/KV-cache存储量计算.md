# 1. KV-Cache 存储量计算

参考文献：https://zhuanlan.zhihu.com/p/16730036197

## 术语定义

| 符号 | 含义 | 说明 |
|-----|------|------|
| `L` | Transformer层数 | 指decoder的层数 |
| `H` | 注意力头数 (num_heads) | 单层内的并行注意力头 |
| `d_h` | 头维度 (head_dim) | 每个注意力头的向量维度 |
| `d` | 隐藏维度 (d_model) | 总维度，满足 `d = H × d_h` |
| `S` | 序列长度 | token数量 |
| `B` | Batch大小 | 并行处理的样本数 |

## 1.1 模型配置对比

### 1.1.1 Qwen-72B 配置

- **Transformer层数 (L)**: 80
- **注意力头数 (H)**: 64
- **头维度 (d_h)**: 128
- **隐藏维度 (d)**: 8192

### 1.1.2 你的模型配置

- **Transformer层数 (L)**: 8
- **注意力头数 (H)**: 8
- **头维度 (d_h)**: 64
- **隐藏维度 (d)**: 512

## 1.2 单token的KV-Cache计算

### 1.2.1 公式定义

**MHA (Multi-Head Attention)**:
$$
KV\text{-}Cache = L \times H \times d_h \times 2
$$

其中 2 代表 K 和 V 两个矩阵。

**MQA (Multi-Query Attention)**:
$$
KV\text{-}Cache = L \times d_h \times 2
$$

所有头共享单个 K/V。

**MLA (Multi-head Latent Attention)**:
$$
KV\text{-}Cache = L \times (d_{kv\_compressed} + d_{rope})
$$

其中：
- \(d_{kv\_compressed}\) 是压缩的 KV 维度
- \(d_{rope}\) 是 RoPE 位置编码维度

### 1.2.2 单token参数个数

**Qwen-72B (MHA)**:

```
L × H × d_h × 2 = 80 × 64 × 128 × 2 = 1,310,720 个参数
```

**你的模型 (MHA)**:

```
L × H × d_h × 2 = 8 × 8 × 64 × 2 = 8,192 个参数
```

**你的模型 (MQA)**:
```
L × d_h × 2 = 8 × 64 × 2 = 1,024 个参数
```

**你的模型 (MLA)**:
```
L × (d_kv_compressed + d_rope + H × v_head_dim)
= 8 × (128 + 32 + 8 × 64)
= 8 × (128 + 32 + 512)
= 8 × 672
= 5,376 个参数
```

其中：
- d_kv_compressed = 128 (= d_model / 4)
- d_rope = 32 (= d_h / 2)
- H × v_head_dim = 8 × 64 = 512

#### 1.2.2.1 MLA 参数设置说明（低秩分解设计）

MLA 通过**低秩分解（LoRA）**压缩 KV-Cache，关键设置来自 DeepSeek-V3 最佳实践：

**1) `kv_lora_rank = d_model // 4`（KV 低秩维度）**

| 项目 | 说明 |
|------|------|
| **设置** | `128 = 512 // 4` |
| **含义** | K 通过低秩投影，维度从 512 压缩到 128 |
| **优势** | 节省 **75% 的 K-cache** 显存（512 → 128） |
| **原理** | 低秩分解能有效捕捉 K 的主要信息，无需全维度存储 |

**2) `qk_nope_head_dim = head_dim // 2`（内容维度）**

| 项目 | 说明 |
|------|------|
| **设置** | `32 = 64 // 2` |
| **含义** | Q 和 K 分成两部分：内容(nope)和位置(rope)各占一半 |
| **分解** | Q/K 头维度 = nope(32) + rope(32) = 64 |
| **作用** | nope 处理语义特征，rope 处理位置信息，解耦两个维度 |

**综合效果**：

```
KV-Cache 结构 = (kv_lora_rank + qk_rope_head_dim + H×v_head_dim) × L × S
              = (128 + 32 + 512) × 8 × S
              ≈ 0.66 MB (S=64)  // 相比标准注意力的 1.00 MB 节省 34.4%
```

### 1.2.3 单token的KV-Cache数据元素个数和显存占用

**KV-Cache数据元素个数** (公式):

$$
num_{kv} = 2 \times L \times n_h
$$

其中：
- \(2\) 表示 K 和 V 两个矩阵
- \(L\) 是 Transformer 层数
- \(n_h\) 是每层内的维度 (= H × d_h)

**单token显存占用** (bf16精度，2 bytes/数据元素):

$$
1token\_mem_{kv} = num_{kv} \times 2 \text{ bytes}
$$

**示例 - Qwen-72B (MHA)**:
```
数据元素个数: 2 × L × n_h = 2 × 80 × 64 = 10,240 个数据元素
显存占用:    10,240 × 2 bytes = 2.62 MB
```

**示例 - 你的模型 (MHA)**:
```
数据元素个数: 2 × L × n_h = 2 × 8 × 8 × 64 = 8,192 个数据元素
显存占用:    8,192 × 2 bytes = 16 KB
```

**示例 - 你的模型 (MQA)**:
```
数据元素个数: 2 × L × d_h = 2 × 8 × 64 = 1,024 个数据元素
显存占用:    1,024 × 2 bytes = 2 KB
```

**示例 - 你的模型 (MLA)**:
```
数据元素个数: L × (128 + 32 + 512) = 8 × 672 = 5,376 个数据元素
显存占用:    5,376 × 2 bytes = 10.8 KB
```

## 1.3 对比表格

| 模型 | 注意力类型 | L | H | d_h | 单token数据元素 | 单token显存 |
|------|-----------|---|---|-----|---------------|-----------|
| Qwen-72B | MHA | 80 | 64 | 128 | 1,310,720 | 2.62 MB |
| 你的模型 | MHA | 8 | 8 | 64 | 8,192 | 16 KB |
| 你的模型 | MQA | 8 | 1 | 64 | 1,024 | 2 KB |
| 你的模型 | MLA | 8 | - | 64 | 5,376 | 10.8 KB |

## 1.4 序列长度为 S 时的总KV-Cache

对于 Batch 大小为 1：

$$
Total\text{-}KV\text{-}Cache = S \times (\text{单token数据元素} \times 2\text{ bytes})
$$

**你的模型 (S=64)**:
- MHA: 64 × 16 KB = 1.00 MB
- MQA: 64 × 2 KB = 128 KB
- MLA: 64 × 10.8 KB = 691.2 KB

## 1.5 MLA vs 标准注意力 (64 tokens)

### 1.5.1 理论对比

**标准注意力 (MHA)**:
$$
KV\text{-}Cache = 8 \times 8 \times 64 \times 64 \times 2 = 1.00 \text{ MB}
$$

**MLA**:
$$
KV\text{-}Cache = (128 + 32 + 512) \times 8 \times 64 \times 2 = 0.66 \text{ MB}
$$

**压缩比**: 
$$
(1.00 - 0.66) / 1.00 = 34.0\%
$$

### 1.5.2 实验验证

运行 KV-cache 对比测试:

```bash
CUDA_VISIBLE_DEVICES=1 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/mid_e1_s222.pt \
    --no_mla_checkpoint checkpoints_no_mla/mid_e1_s222.pt \
    --test_lengths 64
     # --test_lengths 128
```

#### 1.5.2.1 log日志（128tokens）

```shell
2025-10-26 21:04:55.603 | INFO     | __main__:main:448 - 使用设备: cuda
2025-10-26 21:04:55.603 | INFO     | __main__:main:459 - 
🚀 MLA vs Standard KV-cache 对比
2025-10-26 21:04:55.604 | INFO     | __main__:main:460 - ============================================================
2025-10-26 21:04:55.604 | INFO     | __main__:main:465 - 
============================================================
2025-10-26 21:04:55.604 | INFO     | __main__:main:466 - 测试: 128 tokens
2025-10-26 21:04:55.604 | INFO     | __main__:main:467 - ============================================================
2025-10-26 21:04:55.604 | INFO     | __main__:benchmark_model:362 - 
============================================================
2025-10-26 21:04:55.604 | INFO     | __main__:benchmark_model:363 - 测试 MLA 模型 (生成 128 tokens)
2025-10-26 21:04:55.604 | INFO     | __main__:benchmark_model:364 - ============================================================
2025-10-26 21:04:55.828 | INFO     | __main__:load_model_and_tokenizers:257 - 加载模型: MLA
2025-10-26 21:05:11.497 | INFO     | __main__:load_model_and_tokenizers:340 - ✅ Checkpoint loaded from checkpoints/mid_e1_s222.pt
2025-10-26 21:05:14.139 | INFO     | __main__:benchmark_model:376 - 模型加载完成，GPU内存: 1649.0 MB
2025-10-26 21:05:21.116 | INFO     | __main__:benchmark_model:389 - 
生成: 128 tokens | 时间: 6.867s | KV-cache: 1.31 MB
2025-10-26 21:05:23.381 | INFO     | __main__:benchmark_model:362 - 
============================================================
2025-10-26 21:05:23.381 | INFO     | __main__:benchmark_model:363 - 测试 Standard 模型 (生成 128 tokens)
2025-10-26 21:05:23.381 | INFO     | __main__:benchmark_model:364 - ============================================================
2025-10-26 21:05:23.573 | INFO     | __main__:load_model_and_tokenizers:257 - 加载模型: Standard
2025-10-26 21:05:38.940 | INFO     | __main__:load_model_and_tokenizers:340 - ✅ Checkpoint loaded from checkpoints_no_mla/mid_e1_s222.pt
2025-10-26 21:05:40.643 | INFO     | __main__:benchmark_model:376 - 模型加载完成，GPU内存: 1688.7 MB
2025-10-26 21:05:45.620 | INFO     | __main__:benchmark_model:389 - 
生成: 128 tokens | 时间: 4.856s | KV-cache: 2.00 MB
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:410 - 
============================================================
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:411 - 📊 MLA vs Standard 对比结果
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:412 - ============================================================
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:422 - 
💾 KV-cache: 1.31 MB vs 2.00 MB | 节省 34.4%
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:423 - ⏱️  推理时间: 6.867s vs 4.856s | 慢 41.4%
2025-10-26 21:05:45.739 | INFO     | __main__:compare_results:424 - 
✅ MLA节省 34.4% KV-cache显存，但推理慢 41.4%
2025-10-26 21:05:47.935 | INFO     | __main__:main:504 - 
============================================================
2025-10-26 21:05:47.935 | INFO     | __main__:main:505 - ✅ 测试完成
2025-10-26 21:05:47.935 | INFO     | __main__:main:506 - ============================================================
(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers# 
```

#### 1.5.2.2 log日志（64 tokens）

```shell
2025-10-26 21:00:31.584 | INFO     | __main__:main:448 - 使用设备: cuda
2025-10-26 21:00:31.584 | INFO     | __main__:main:459 - 
🚀 MLA vs Standard KV-cache 对比
2025-10-26 21:00:31.585 | INFO     | __main__:main:460 - ============================================================
2025-10-26 21:00:31.585 | INFO     | __main__:main:465 - 
============================================================
2025-10-26 21:00:31.585 | INFO     | __main__:main:466 - 测试: 64 tokens
2025-10-26 21:00:31.585 | INFO     | __main__:main:467 - ============================================================
2025-10-26 21:00:31.585 | INFO     | __main__:benchmark_model:362 - 
============================================================
2025-10-26 21:00:31.585 | INFO     | __main__:benchmark_model:363 - 测试 MLA 模型 (生成 64 tokens)
2025-10-26 21:00:31.585 | INFO     | __main__:benchmark_model:364 - ============================================================
2025-10-26 21:00:31.789 | INFO     | __main__:load_model_and_tokenizers:257 - 加载模型: MLA
2025-10-26 21:00:49.349 | INFO     | __main__:load_model_and_tokenizers:340 - ✅ Checkpoint loaded from checkpoints/mid_e1_s222.pt
2025-10-26 21:00:50.779 | INFO     | __main__:benchmark_model:376 - 模型加载完成，GPU内存: 1649.0 MB
2025-10-26 21:00:54.672 | INFO     | __main__:benchmark_model:389 - 
生成: 64 tokens | 时间: 3.836s | KV-cache: 0.66 MB
2025-10-26 21:00:56.261 | INFO     | __main__:benchmark_model:362 - 
============================================================
2025-10-26 21:00:56.261 | INFO     | __main__:benchmark_model:363 - 测试 Standard 模型 (生成 64 tokens)
2025-10-26 21:00:56.261 | INFO     | __main__:benchmark_model:364 - ============================================================
2025-10-26 21:00:56.458 | INFO     | __main__:load_model_and_tokenizers:257 - 加载模型: Standard
2025-10-26 21:01:15.920 | INFO     | __main__:load_model_and_tokenizers:340 - ✅ Checkpoint loaded from checkpoints_no_mla/mid_e1_s222.pt
2025-10-26 21:01:17.682 | INFO     | __main__:benchmark_model:376 - 模型加载完成，GPU内存: 1688.7 MB
2025-10-26 21:01:20.096 | INFO     | __main__:benchmark_model:389 - 
生成: 64 tokens | 时间: 2.350s | KV-cache: 1.00 MB
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:410 - 
============================================================
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:411 - 📊 MLA vs Standard 对比结果
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:412 - ============================================================
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:422 - 
💾 KV-cache: 0.66 MB vs 1.00 MB | 节省 34.4%
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:423 - ⏱️  推理时间: 3.836s vs 2.350s | 慢 63.2%
2025-10-26 21:01:20.218 | INFO     | __main__:compare_results:424 - 
✅ MLA节省 34.4% KV-cache显存，但推理慢 63.2%
2025-10-26 21:01:22.405 | INFO     | __main__:main:504 - 
============================================================
2025-10-26 21:01:22.405 | INFO     | __main__:main:505 - ✅ 测试完成
2025-10-26 21:01:22.405 | INFO     | __main__:main:506 - ============================================================
(train_transformers) root@iv-ydg6wcq3ggay8n6dmn75:/data2/workspace/yszhang/train_transformers# 
```

#### 1.5.2.2 对比分析

**综合对比表** - 展示token个数对MLA效果的影响：

| 指标 | 64 tokens (MLA) | 64 tokens (标准) | 128 tokens (MLA) | 128 tokens (标准) |
|------|---|---|---|---|
| KV-cache 显存 | 0.66 MB | 1.00 MB | 1.31 MB | 2.00 MB |
| 推理时间 | 3.836s | 2.350s | 6.867s | 4.856s |
| 显存节省比例 | 34.4% | - | 34.4% | - |
| 速度损失 | 63.2% | - | 41.4% | - |

**关键发现**：

1. **显存节省稳定**
   - 无论64还是128 tokens，MLA 都稳定节省 **34.4%** KV-cache显存
   - 显存节省比例与序列长度无关，由注意力机制本身决定

2. **速度损失随token增加而减少**
   - 64 tokens: MLA 慢 **63.2%**
   - 128 tokens: MLA 慢 **41.4%**
   - **趋势**: 序列越长，MLA 的速度损失百分比越小
   - **原因**: 固定成本（模型加载）占比降低，KV-cache拼接操作占比增加

3. **长序列优势明显**
   - 在128 tokens上，MLA的速度劣势已从63%降至41%
   - 继续增加序列长度，MLA相对速度优势会进一步改善
   - 最终在长序列/大batch场景，显存节省的收益会大于速度损失

**结论**: MLA 有效压缩 KV-cache（34.4%），在短序列有速度成本，但**随着序列变长，这个成本比例逐渐摊薄**。

## 相关文件

- **实现代码**: `train_tmp.py` (759-916行)
- **测试脚本**: `inference/compare_kv_cache_mla.py`
- **精度说明**: `doc/files/inference/精度统一说明.md`


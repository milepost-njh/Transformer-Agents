# 1. KV-Cache 存储量计算

参考文献：https://zhuanlan.zhihu.com/p/16730036197

## 术语定义

| 符号 | 含义 | 说明 |
|-----|------|------|
| `L` | Transformer层数 | 指decoder堆栈中的层数 |
| `H` | 注意力头数 (num_heads) | 单层内的并行注意力头 |
| `d_h` | 头维度 (head_dim) | 每个注意力头的向量维度 |
| `d` | 隐藏维度 (d_model) | 总维度，满足 `d = H × d_h` |
| `S` | 序列长度 | token数量 |
| `B` | Batch大小 | 并行处理的样本数 |

## 1.1 模型配置对比

### 1.1.1 你的模型配置
- **Transformer层数 (L)**: 8
- **注意力头数 (H)**: 8
- **头维度 (d_h)**: 64
- **隐藏维度 (d)**: 512

### 1.1.2 Qwen-72B 配置
- **Transformer层数 (L)**: 80
- **注意力头数 (H)**: 64
- **头维度 (d_h)**: 128
- **隐藏维度 (d)**: 8192

## 1.2 单token的KV-Cache计算

### 1.2.1 公式定义

**MHA (Multi-Head Attention)**:
$$KV\text{-}Cache = L \times H \times d_h \times 2$$

其中 2 代表 K 和 V 两个矩阵。

**MQA (Multi-Query Attention)**:
$$KV\text{-}Cache = L \times d_h \times 2$$

所有头共享单个 K/V。

**MLA (Multi-head Latent Attention)**:
$$KV\text{-}Cache = L \times (d_{kv\_compressed} + d_{rope})$$

其中：
- $d_{kv\_compressed}$ 是压缩的 KV 维度
- $d_{rope}$ 是 RoPE 位置编码维度

### 1.2.2 具体数值

**你的模型 (MHA)**:
```
L × H × d_h × 2 = 8 × 8 × 64 × 2 = 8,192 参数
```

**你的模型 (MQA)**:
```
L × d_h × 2 = 8 × 64 × 2 = 1,024 参数
```

**你的模型 (MLA)**:
```
L × (128 + 32 + 512) = 8 × 672 = 5,376 参数
```

**Qwen-72B (MHA)**:
```
L × H × d_h × 2 = 80 × 64 × 128 × 2 = 1,310,720 参数
```

### 1.2.3 单token显存占用 (bf16精度，2 bytes/参数)

**你的模型 (MHA)**:
```
8,192 × 2 bytes = 16 KB
```

**你的模型 (MQA)**:
```
1,024 × 2 bytes = 2 KB
```

**你的模型 (MLA)**:
```
5,376 × 2 bytes = 10.8 KB
```

**Qwen-72B (MHA)**:
```
1,310,720 × 2 bytes = 2.62 MB
```

## 1.3 对比表格

| 模型 | 注意力类型 | L | H | d_h | 单token参数 | 单token显存 |
|------|-----------|---|---|-----|-----------|-----------|
| Qwen-72B | MHA | 80 | 64 | 128 | 1,310,720 | 2.62 MB |
| 你的模型 | MHA | 8 | 8 | 64 | 8,192 | 16 KB |
| 你的模型 | MQA | 8 | 1 | 64 | 1,024 | 2 KB |
| 你的模型 | MLA | 8 | - | 64 | 5,376 | 10.8 KB |

## 1.4 序列长度为 S 时的总KV-Cache

对于 Batch 大小为 1：

$$Total\text{-}KV\text{-}Cache = S \times (\text{单token参数} \times 2\text{ bytes})$$

**你的模型 (S=64)**:
- MHA: 64 × 16 KB = 1.00 MB
- MQA: 64 × 2 KB = 128 KB
- MLA: 64 × 10.8 KB = 691.2 KB

## 1.5 MLA vs 标准注意力 (64 tokens)

### 1.5.1 理论对比

**标准注意力 (MHA)**:
$$KV\text{-}Cache = 8 \times 8 \times 64 \times 64 \times 2 = 1.00 \text{ MB}$$

**MLA**:
$$KV\text{-}Cache = (128 + 32 + 512) \times 8 \times 64 \times 2 = 0.66 \text{ MB}$$

**压缩比**: $(1.00 - 0.66) / 1.00 = 34.0\%$

### 1.5.2 实验验证 (64 tokens)

| 指标 | MLA | 标准 | 差异 |
|------|-----|------|------|
| KV-cache显存 | 0.66 MB | 1.00 MB | 节省 34.0% |
| 推理时间 | 2.667s | 2.344s | 慢 13.8% |

**结论**: MLA 有效压缩 KV-cache（34.0%），代价是推理速度略慢。

## 相关文件

- **实现代码**: `train_tmp.py` (759-916行)
- **测试脚本**: `inference/compare_kv_cache_mla.py`
- **精度说明**: `doc/files/inference/精度统一说明.md`


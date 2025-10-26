# MLA vs 标准注意力 KV-cache 对比测试

## 概述

本目录包含用于对比 MLA（Multi-head Latent Attention）和标准注意力机制在 KV-cache 效率方面的工具。

## 主要改进

### 1. KV-cache 支持

为 `train_tmp.py` 中的 `MultiHeadAttention` 类添加了真正的 KV-cache 支持：

- **标准模式**: 缓存完整的 Key 和 Value 张量
  - K: `[batch_size, num_heads, seq_len, head_dim]`
  - V: `[batch_size, num_heads, seq_len, head_dim]`
  - 总大小: `2 * num_layers * num_heads * head_dim * seq_len * 4 bytes`

- **MLA 模式**: 缓存压缩的 KV 表示
  - K: `[batch_size, 1, seq_len, kv_lora_rank + qk_rope_head_dim]`
  - V: `[batch_size, num_heads, seq_len, v_head_dim]`
  - 总大小: `num_layers * (kv_lora_rank + qk_rope_head_dim + num_heads * v_head_dim) * seq_len * 4 bytes`

### 2. 新增功能

#### `MultiHeadAttention.forward()` 参数更新

```python
def forward(self, q, k, v, mask=None, return_attn: bool = True, 
           past_key_value=None, use_cache: bool = False):
    """
    参数:
        q, k, v: 输入张量
        mask: 注意力掩码
        return_attn: 是否返回注意力权重
        past_key_value: 之前步骤的KV-cache，格式为 (past_k, past_v)
        use_cache: 是否返回当前步骤的KV-cache
    
    返回:
        - 如果 use_cache=False: (output, attention_weights) 或 output
        - 如果 use_cache=True: (output, attention_weights, present_kv) 或 (output, present_kv)
    """
```

## 使用方法

### 1. 基本对比测试

使用 `compare_kv_cache_mla.py` 进行详细的对比测试：

```bash
# 基本使用
CUDA_VISIBLE_DEVICES=5 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt

# 指定测试长度
CUDA_VISIBLE_DEVICES=5 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_lengths 32 64 128 256

# 自定义测试输入
CUDA_VISIBLE_DEVICES=5 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_input "Seu texto de teste aqui." \
    --test_lengths 64 128

# 保存结果到JSON
CUDA_VISIBLE_DEVICES=5 python inference/compare_kv_cache_mla.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_lengths 32 64 128 \
    --output results/kv_cache_comparison.json
```

### 2. 简化对比测试

使用 `compare_mla_kv_cache_real.py`（调用上述脚本的简化版本）：

```bash
CUDA_VISIBLE_DEVICES=5 python inference/compare_mla_kv_cache_real.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_lengths 32 64 128
```

### 3. 单独推理测试

使用 `inference.py` 进行单独的推理测试：

```bash
# MLA 模式
CUDA_VISIBLE_DEVICES=5 python inference/inference.py \
    --mode mla \
    --checkpoint checkpoints/latest.pt \
    --input "O Tom está procurando uma segunda opinião."

# 标准模式（需要使用 --no_mla 训练的模型）
CUDA_VISIBLE_DEVICES=5 python inference/inference.py \
    --mode normal \
    --checkpoint checkpoints_no_mla/latest.pt \
    --input "O Tom está procurando uma segunda opinião."
```

## 输出示例

### KV-cache 对比结果

```
============================================================
测试 MLA 模型 (生成 64 tokens)
============================================================
模型加载完成，GPU内存: 1234.5 MB

生成结果:
  输入: O Tom está procurando uma segunda opinião sobre o tratamento médico.
  输出: Tom is looking for a second opinion about medical treatment.
  生成token数: 12
  总时间: 0.856s
  平均每步时间: 71.33ms

内存使用:
  初始GPU内存: 0.0 MB
  模型加载后: 1234.5 MB
  生成后: 1256.8 MB
  最大KV-cache: 15.36 MB
  最终KV-cache: 15.36 MB

============================================================
📊 对比分析
============================================================

⏱️  生成时间:
  MLA: 0.856s
  Standard: 0.923s
  差异: -0.067s (+7.3%)

⚡ 平均每步时间:
  MLA: 71.33ms
  Standard: 76.92ms
  差异: -5.59ms (+7.3%)

💾 KV-cache内存:
  MLA: 15.36 MB
  Standard: 24.58 MB
  节省: 9.22 MB (37.5%)

🖥️  GPU内存:
  MLA: 1256.8 MB
  Standard: 1265.9 MB
  差异: -9.1 MB

💡 结论:
  ✅ MLA成功压缩KV-cache，节省 37.5% 内存
  ✅ MLA加速推理，提升 7.3%
```

## 技术原理

### MLA 的 KV-cache 压缩

MLA 通过以下方式压缩 KV-cache：

1. **低秩分解**: 使用低秩投影减少 KV 的维度
   - `kv_a_proj`: 将 `d_model` 压缩到 `kv_lora_rank`（通常是 `d_model // 4`）
   - `kv_b_proj`: 从压缩表示恢复到多头表示

2. **共享 K 的 RoPE 部分**: 
   - `k_pe` 在所有头之间共享，只需存储一份
   - 只有 `k_nope` 部分需要多头存储

3. **理论压缩比**:
   ```
   标准注意力: 2 * num_heads * head_dim
   MLA: kv_lora_rank + qk_rope_head_dim + num_heads * v_head_dim
   
   示例 (d_model=512, num_heads=8, head_dim=64):
   标准: 2 * 8 * 64 = 1024 参数/token
   MLA: 128 + 32 + 8 * 64 = 672 参数/token
   压缩比: 34.4%
   ```

## 性能指标

对比测试会输出以下指标：

1. **生成时间**: 总的生成时间和平均每步时间
2. **KV-cache 内存**: 理论计算的 KV-cache 大小
3. **GPU 内存**: 实际的 GPU 内存使用
4. **压缩比**: MLA 相对于标准注意力的内存节省比例

## 注意事项

1. **模型兼容性**: 确保使用正确的 checkpoint
   - MLA 模式需要使用 `use_mla=True` 训练的模型
   - 标准模式需要使用 `use_mla=False` 训练的模型

2. **内存测量**: 
   - KV-cache 大小是理论计算值
   - 实际 GPU 内存包括模型权重、激活值等

3. **序列长度**: 
   - KV-cache 的优势在长序列时更明显
   - 建议测试多个长度（32, 64, 128, 256）

## 文件说明

- `compare_kv_cache_mla.py`: 主要的对比测试脚本，包含完整的 KV-cache 追踪和分析
- `compare_mla_kv_cache_real.py`: 简化的对比脚本，调用上述脚本
- `inference.py`: 通用推理脚本，支持多种模式
- `README_KV_CACHE.md`: 本文档

## 参考资料

- [DeepSeek-V3 论文](https://arxiv.org/abs/2401.xxxxx)
- [Multi-head Latent Attention 技术文档](../doc/files/inference/)


# KV-cache实现指南

> **文档类型**：实现指南  
> **难度级别**：高级  
> **适用对象**：深度学习研究者、Transformer架构开发者

## 概述

要真正测试MLA的KV-cache优势，需要实现真正的KV-cache机制。本文档提供了具体的实现方案。

## 实现步骤

### 1. 实现KV-cache类

```python
class KVCache:
    """KV-cache实现类"""
    
    def __init__(self, max_length: int, num_layers: int, num_heads: int, head_dim: int, 
                 use_mla: bool = False, kv_lora_rank: int = None):
        self.max_length = max_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank or head_dim // 4
        
        # 初始化缓存
        self.cache = {}
        self.current_length = 0
        
    def get_cache_size_mb(self) -> float:
        """计算当前KV-cache的内存使用量（MB）"""
        total_size = 0
        for layer_idx in range(self.num_layers):
            if layer_idx in self.cache:
                for key in ['key', 'value']:
                    if key in self.cache[layer_idx]:
                        tensor = self.cache[layer_idx][key]
                        total_size += tensor.numel() * tensor.element_size()
        return total_size / 1024 / 1024
    
    def update_cache(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """更新指定层的KV-cache"""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {}
        
        # 如果是第一次，直接存储
        if 'key' not in self.cache[layer_idx]:
            self.cache[layer_idx]['key'] = key
            self.cache[layer_idx]['value'] = value
        else:
            # 拼接新的KV到现有缓存
            self.cache[layer_idx]['key'] = torch.cat([
                self.cache[layer_idx]['key'], key
            ], dim=2)  # 在序列维度拼接
            
            self.cache[layer_idx]['value'] = torch.cat([
                self.cache[layer_idx]['value'], value
            ], dim=2)
        
        self.current_length += 1
```

### 2. 修改MultiHeadAttention

需要修改`MultiHeadAttention`的`forward`方法，添加KV-cache支持：

```python
def forward(self, q, k, v, mask=None, return_attn: bool = True, kv_cache=None, layer_idx: int = 0):
    """
    支持KV-cache的前向传播
    """
    # ... 现有的注意力计算逻辑 ...
    
    # 更新KV-cache
    if kv_cache is not None:
        if self.use_mla:
            # MLA模式：缓存压缩的KV
            kv_cache.update_cache(layer_idx, compressed_kv, k_pe)
        else:
            # 标准模式：缓存完整的KV
            kv_cache.update_cache(layer_idx, k, v)
    
    return output, attn_weights
```

### 3. 实现自回归生成

```python
def autoregressive_generate(self, input_text: str, max_new_tokens: int = 64) -> Dict:
    """自回归生成，使用KV-cache"""
    # 编码输入
    encoder_input = self.engine.encode_input(input_text)
    
    # 初始化decoder输入
    start_id = self.engine.loader.en_tokenizer.bos_token_id
    end_id = self.engine.loader.en_tokenizer.eos_token_id
    decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=self.config.device)
    
    generated_tokens = []
    kv_cache_sizes = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # 记录KV-cache大小
            kv_cache_size = self.kv_cache.get_cache_size_mb()
            kv_cache_sizes.append(kv_cache_size)
            
            # 前向传播（传入KV-cache）
            model_output = self.engine.loader.model(
                encoder_input, decoder_input,
                src_mask=enc_pad_mask,
                tgt_mask=dec_mask,
                enc_dec_mask=enc_dec_mask,
                kv_cache=self.kv_cache  # 传入KV-cache
            )
            
            # 处理输出和生成下一个token
            # ...
    
    return {
        'kv_cache_sizes': kv_cache_sizes,
        'max_kv_cache_size': max(kv_cache_sizes),
        # ... 其他结果
    }
```

### 4. 测试脚本

使用`compare_mla_kv_cache_real.py`进行测试：

```bash
python inference/compare_mla_kv_cache_real.py \
    --mla_checkpoint checkpoints/latest.pt \
    --no_mla_checkpoint checkpoints_no_mla/latest.pt \
    --test_lengths 32 64 128 256
```

## 预期结果

实现真正的KV-cache后，应该看到：

1. **内存节省**：MLA在长序列中显著减少KV-cache内存
2. **序列越长效果越明显**：压缩比例随序列长度增加
3. **推理速度提升**：减少内存访问，提高缓存命中率

## 关键差异

### 标准注意力KV-cache
```python
# 每个token存储: [B, H, seq_len, d_model]
kv_cache[key] = current_k    # 完整维度
kv_cache[value] = current_v  # 完整维度
```

### MLA KV-cache
```python
# 每个token存储: [B, H, seq_len, compressed_dim]
kv_cache[compressed_kv] = compressed_kv  # 压缩维度
kv_cache[k_pe] = k_pe                   # 位置编码部分
```

## 实现难点

1. **模型修改**：需要修改现有的MultiHeadAttention类
2. **缓存管理**：需要正确管理多层KV-cache
3. **内存测量**：需要准确测量KV-cache的内存使用
4. **序列拼接**：需要正确处理序列维度的拼接

## 总结

实现真正的KV-cache是测试MLA优势的关键。通过上述步骤，可以正确测试MLA在长序列自回归生成中的KV-cache压缩效果。

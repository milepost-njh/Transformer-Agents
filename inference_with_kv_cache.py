"""
推理阶段 KV-cache 演示脚本
展示 MLA 如何在推理时节省 KV-cache 内存
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from loguru import logger


class SimpleKVCache:
    """简单的 KV-cache 实现，用于演示"""
    
    def __init__(self):
        self.key_cache = []    # 每层的 key cache
        self.value_cache = []  # 每层的 value cache
        
    def update(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        更新 KV cache
        Args:
            layer_idx: 层索引
            new_keys: [B, H, 1, head_dim] 新生成 token 的 keys
            new_values: [B, H, 1, v_head_dim] 新生成 token 的 values
        Returns:
            完整的 keys 和 values: [B, H, seq_len, dim]
        """
        # 确保 cache 有足够的层
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        # 拼接新的 KV
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = new_keys
            self.value_cache[layer_idx] = new_values
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], new_keys], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], new_values], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """计算 KV-cache 占用的内存（MB）"""
        total_memory = 0
        for keys, values in zip(self.key_cache, self.value_cache):
            if keys is not None:
                total_memory += keys.element_size() * keys.nelement()
                total_memory += values.element_size() * values.nelement()
        return {
            'total_mb': total_memory / (1024 ** 2),
            'total_bytes': total_memory
        }


def calculate_kv_cache_size(
    batch_size: int,
    num_layers: int, 
    num_heads: int,
    seq_len: int,
    head_dim: int,
    v_head_dim: Optional[int] = None,
    dtype_bytes: int = 2  # float16/bfloat16
):
    """
    计算 KV-cache 需要的内存
    
    Args:
        batch_size: batch 大小
        num_layers: transformer 层数
        num_heads: 注意力头数
        seq_len: 序列长度
        head_dim: key 的每个头维度
        v_head_dim: value 的每个头维度（MLA 中可能不同）
        dtype_bytes: 数据类型字节数（float16=2, float32=4）
    """
    if v_head_dim is None:
        v_head_dim = head_dim
    
    # Key cache: [batch, num_layers, num_heads, seq_len, head_dim]
    key_size = batch_size * num_layers * num_heads * seq_len * head_dim * dtype_bytes
    
    # Value cache: [batch, num_layers, num_heads, seq_len, v_head_dim]  
    value_size = batch_size * num_layers * num_heads * seq_len * v_head_dim * dtype_bytes
    
    total_bytes = key_size + value_size
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    
    return {
        'key_mb': key_size / (1024 ** 2),
        'value_mb': value_size / (1024 ** 2),
        'total_mb': total_mb,
        'total_gb': total_gb,
        'total_bytes': total_bytes
    }


def demonstrate_kv_cache_savings():
    """演示 MLA 相比标准注意力节省了多少 KV-cache"""
    
    logger.info("=" * 80)
    logger.info("🔍 KV-Cache 内存占用对比分析")
    logger.info("=" * 80)
    
    # 配置参数（与你的训练配置一致）
    batch_size = 1  # 推理时通常 batch_size=1
    num_layers = 4
    num_heads = 8
    d_model = 512
    head_dim = d_model // num_heads  # 64
    
    # 测试不同序列长度
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192]
    
    logger.info(f"📊 模型配置:")
    logger.info(f"   Batch Size: {batch_size}")
    logger.info(f"   层数: {num_layers}")
    logger.info(f"   注意力头数: {num_heads}")
    logger.info(f"   d_model: {d_model}")
    logger.info(f"   标准 head_dim: {head_dim}")
    logger.info("")
    
    # MLA 配置
    q_lora_rank = d_model // 2  # 256
    kv_lora_rank = d_model // 4  # 128
    qk_rope_head_dim = head_dim // 2  # 32
    qk_nope_head_dim = head_dim // 2  # 32
    v_head_dim = head_dim  # 64
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 64
    
    logger.info(f"🔧 MLA 配置:")
    logger.info(f"   Q LoRA Rank: {q_lora_rank}")
    logger.info(f"   KV LoRA Rank: {kv_lora_rank}")
    logger.info(f"   qk_rope_head_dim: {qk_rope_head_dim}")
    logger.info(f"   qk_nope_head_dim: {qk_nope_head_dim}")
    logger.info(f"   v_head_dim: {v_head_dim}")
    logger.info(f"   q_head_dim: {q_head_dim}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("📈 不同序列长度下的 KV-Cache 内存占用对比")
    logger.info("=" * 80)
    
    for seq_len in seq_lengths:
        # 标准注意力的 KV-cache
        standard_cache = calculate_kv_cache_size(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            v_head_dim=head_dim,
            dtype_bytes=2  # float16
        )
        
        # MLA 的 KV-cache（K 和 V 的维度相同，都是 q_head_dim）
        # 注意：MLA 中 K 的维度是 q_head_dim（包含 nope 和 rope），V 是 v_head_dim
        mla_cache = calculate_kv_cache_size(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=q_head_dim,  # Key 维度
            v_head_dim=v_head_dim,  # Value 维度
            dtype_bytes=2  # float16
        )
        
        # 计算节省
        saved_mb = standard_cache['total_mb'] - mla_cache['total_mb']
        saved_ratio = (saved_mb / standard_cache['total_mb']) * 100
        
        logger.info(f"\n📍 序列长度: {seq_len}")
        logger.info(f"   标准注意力 KV-Cache: {standard_cache['total_mb']:.2f} MB ({standard_cache['total_gb']:.4f} GB)")
        logger.info(f"   MLA KV-Cache: {mla_cache['total_mb']:.2f} MB ({mla_cache['total_gb']:.4f} GB)")
        logger.info(f"   💡 节省内存: {saved_mb:.2f} MB ({saved_ratio:.2f}%)")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 关键洞察")
    logger.info("=" * 80)
    logger.info("1. 🔴 在当前配置下，MLA 的 KV-cache 节省为 0%")
    logger.info("   原因: q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 32 + 32 = 64")
    logger.info("         v_head_dim = 64")
    logger.info("         与标准注意力的 head_dim = 64 相同")
    logger.info("")
    logger.info("2. 💡 要真正体现 MLA 的 KV-cache 优势，需要:")
    logger.info("   - 减小 v_head_dim (例如 v_head_dim = d_model // num_heads // 2 = 32)")
    logger.info("   - 或者使用 Multi-Query Attention (MQA): num_kv_heads < num_heads")
    logger.info("")
    logger.info("3. 🚀 DeepSeekV3 的实际配置:")
    logger.info("   - 使用了 Grouped-Query Attention (GQA)")
    logger.info("   - num_key_value_heads 远小于 num_attention_heads")
    logger.info("   - 例如: num_heads=128, num_kv_heads=8 (16:1 比例)")
    logger.info("")
    logger.info("4. 📈 序列越长，KV-cache 节省越明显")
    logger.info("   - 对于 8192 tokens 的长序列尤其重要")
    logger.info("   - 这是 MLA 在推理时的主要优势")
    logger.info("=" * 80)


def demonstrate_improved_mla_config():
    """演示改进的 MLA 配置，展示真正的内存节省"""
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 改进的 MLA 配置 - 真正的内存节省")
    logger.info("=" * 80)
    
    batch_size = 1
    num_layers = 4
    num_heads = 8
    d_model = 512
    head_dim = d_model // num_heads  # 64
    
    # 改进的 MLA 配置：减小 v_head_dim
    v_head_dim_reduced = head_dim // 2  # 32（减半）
    q_head_dim = head_dim  # 保持 64
    
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192]
    
    logger.info(f"📊 改进配置:")
    logger.info(f"   标准 head_dim: {head_dim}")
    logger.info(f"   MLA q_head_dim: {q_head_dim} (保持)")
    logger.info(f"   MLA v_head_dim: {v_head_dim_reduced} (减半)")
    logger.info("")
    
    for seq_len in seq_lengths:
        # 标准注意力
        standard_cache = calculate_kv_cache_size(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            v_head_dim=head_dim,
            dtype_bytes=2
        )
        
        # 改进的 MLA
        improved_mla_cache = calculate_kv_cache_size(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=q_head_dim,
            v_head_dim=v_head_dim_reduced,  # 减半
            dtype_bytes=2
        )
        
        saved_mb = standard_cache['total_mb'] - improved_mla_cache['total_mb']
        saved_ratio = (saved_mb / standard_cache['total_mb']) * 100
        
        logger.info(f"📍 序列长度: {seq_len}")
        logger.info(f"   标准: {standard_cache['total_mb']:.2f} MB")
        logger.info(f"   改进 MLA: {improved_mla_cache['total_mb']:.2f} MB")
        logger.info(f"   💡 节省: {saved_mb:.2f} MB ({saved_ratio:.2f}%)")
    
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="{message}",
        level="INFO"
    )
    
    # 运行演示
    demonstrate_kv_cache_savings()
    demonstrate_improved_mla_config()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ 演示完成！")
    logger.info("=" * 80)
    logger.info("")
    logger.info("💡 总结:")
    logger.info("1. 训练阶段不需要 KV-cache")
    logger.info("2. 推理阶段才需要，用于避免重复计算")
    logger.info("3. MLA 的内存节省取决于 v_head_dim 的配置")
    logger.info("4. 配合 GQA/MQA 可以进一步减少内存")
    logger.info("5. 序列越长，节省越明显")
    logger.info("=" * 80)


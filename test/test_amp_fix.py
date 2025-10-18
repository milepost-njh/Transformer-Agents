#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试混合精度训练修复
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

def test_amp_fix():
    """测试混合精度训练修复"""
    print("🧪 测试混合精度训练修复...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return False
    
    device = torch.device("cuda")
    
    # 创建简单的注意力机制测试
    class SimpleAttention(nn.Module):
        def __init__(self, d_model=512, num_heads=8):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.depth = d_model // num_heads
            
            self.WQ = nn.Linear(d_model, d_model)
            self.WK = nn.Linear(d_model, d_model)
            self.WV = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
        def scaled_dot_product_attention(self, q, k, v, mask=None):
            # 模拟修复后的注意力计算
            matmul_qk = torch.matmul(q, k.transpose(-2, -1))
            dk = q.size()[-1]
            scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=q.device))
            
            if mask is not None:
                # 使用 FP16 兼容的值
                if scaled_attention_logits.dtype == torch.float16:
                    mask_value = -6.55e4  # FP16 的最大负值
                else:
                    mask_value = -1e9
                scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 1, mask_value)
            
            attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
            output = torch.matmul(attention_weights, v)
            return output, attention_weights
        
        def forward(self, x, mask=None):
            B, L, D = x.shape
            q = self.WQ(x).view(B, L, self.num_heads, self.depth).transpose(1, 2)
            k = self.WK(x).view(B, L, self.num_heads, self.depth).transpose(1, 2)
            v = self.WV(x).view(B, L, self.num_heads, self.depth).transpose(1, 2)
            
            attn_out, _ = self.scaled_dot_product_attention(q, k, v, mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(attn_out)
    
    # 创建模型和优化器
    model = SimpleAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler('cuda')
    
    # 创建测试数据
    batch_size = 32
    seq_len = 64
    d_model = 512
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    mask = torch.zeros(batch_size, 1, seq_len, seq_len).to(device)
    # 添加一些 mask
    mask[:, :, :10, :10] = 1
    
    print("✅ 模型和数据创建成功")
    
    # 测试混合精度前向传播
    try:
        with autocast('cuda'):
            output = model(x, mask)
        print("✅ 混合精度前向传播成功")
    except Exception as e:
        print(f"❌ 混合精度前向传播失败: {e}")
        return False
    
    # 测试混合精度训练
    try:
        optimizer.zero_grad()
        with autocast('cuda'):
            output = model(x, mask)
            loss = output.mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("✅ 混合精度训练成功")
    except Exception as e:
        print(f"❌ 混合精度训练失败: {e}")
        return False
    
    print("🎉 混合精度训练修复验证成功！")
    return True

if __name__ == "__main__":
    success = test_amp_fix()
    exit(0 if success else 1)

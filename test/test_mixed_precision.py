#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合精度训练功能测试脚本
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def test_mixed_precision_basic():
    """测试基本的混合精度训练功能"""
    print("🧪 测试混合精度训练基本功能...")
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过混合精度测试")
        return False
    
    device = torch.device("cuda")
    
    # 创建简单的测试模型
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    # 创建优化器和缩放器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # 创建测试数据
    x = torch.randn(32, 100).to(device)
    y = torch.randint(0, 10, (32,)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    print("✅ 模型和优化器创建成功")
    
    # 测试混合精度前向传播
    try:
        with autocast():
            output = model(x)
            loss = criterion(output, y)
        print("✅ 混合精度前向传播成功")
    except Exception as e:
        print(f"❌ 混合精度前向传播失败: {e}")
        return False
    
    # 测试混合精度反向传播
    try:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("✅ 混合精度反向传播成功")
    except Exception as e:
        print(f"❌ 混合精度反向传播失败: {e}")
        return False
    
    # 测试缩放器状态
    print(f"📊 GradScaler 状态:")
    print(f"   启用状态: {scaler.is_enabled()}")
    print(f"   当前缩放因子: {scaler.get_scale():.2e}")
    print(f"   增长因子: {scaler.get_growth_factor()}")
    print(f"   回退因子: {scaler.get_backoff_factor()}")
    
    return True

def test_mixed_precision_memory():
    """测试混合精度训练的内存使用"""
    print("\n🧪 测试混合精度训练内存使用...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过内存测试")
        return False
    
    device = torch.device("cuda")
    
    # 创建较大的模型
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # 创建测试数据
    x = torch.randn(128, 1000).to(device)
    y = torch.randint(0, 100, (128,)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 清理内存
    torch.cuda.empty_cache()
    
    # 测试标准精度训练内存使用
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    standard_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"📊 标准精度训练内存使用: {standard_memory:.2f} GB")
    
    # 清理内存
    torch.cuda.empty_cache()
    
    # 测试混合精度训练内存使用
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    with autocast():
        output = model(x)
        loss = criterion(output, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    mixed_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"📊 混合精度训练内存使用: {mixed_memory:.2f} GB")
    
    # 计算内存节省
    memory_saving = (standard_memory - mixed_memory) / standard_memory * 100
    print(f"💾 内存节省: {memory_saving:.1f}%")
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始混合精度训练功能测试\n")
    
    # 基本功能测试
    basic_test = test_mixed_precision_basic()
    
    # 内存使用测试
    memory_test = test_mixed_precision_memory()
    
    # 总结
    print("\n📋 测试结果总结:")
    print(f"   基本功能测试: {'✅ 通过' if basic_test else '❌ 失败'}")
    print(f"   内存使用测试: {'✅ 通过' if memory_test else '❌ 失败'}")
    
    if basic_test and memory_test:
        print("\n🎉 所有测试通过！混合精度训练功能正常工作。")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

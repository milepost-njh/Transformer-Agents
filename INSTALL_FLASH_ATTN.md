# Flash Attention 安装指南

## 🚀 什么是 Flash Attention？

Flash Attention 是一种高效的注意力机制实现，可以带来：
- ✅ **2-4倍** 注意力计算加速
- ✅ **5-20倍** 内存减少
- ✅ 支持更长序列或更大 batch size

## 📦 安装方法

### 方法 1: 使用 pip（推荐）

```bash
pip install flash-attn --no-build-isolation
```

**注意事项：**
- 需要 CUDA 11.8+ 或 12.x
- 需要 PyTorch 2.0+
- 编译时间约 5-10 分钟（需要从源码编译）

### 方法 2: 预编译版本（更快）

如果上述方法编译时间太长，可以使用预编译版本：

```bash
pip install flash-attn --no-build-isolation --no-deps
```

## ✅ 验证安装

```python
python -c "from flash_attn import flash_attn_func; print('Flash Attention 安装成功！')"
```

## 🎯 代码中的使用

代码已经集成了 Flash Attention！
- ✅ 如果已安装：自动启用，打印 "✅ Flash Attention 已启用！"
- ⚠️ 如果未安装：自动回退到标准实现，打印警告信息

**无需修改代码**，安装后重新运行训练即可自动启用。

## 📊 预期效果

安装前后对比：
- **速度**: 当前 56 samples/sec → 预期 **112-224 samples/sec**（2-4倍）
- **内存**: 当前 7.4GB → 预期 **4-6GB**（节省 20-30%）
- **支持更大 batch**: 64 → 可能支持 128-256

## ⚠️ 故障排查

如果安装失败：

```bash
# 1. 检查 CUDA 版本
nvcc --version

# 2. 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 3. 确保版本匹配后重新安装
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

## 📝 参考链接

- Flash Attention 官方: https://github.com/Dao-AILab/flash-attention
- 论文: https://arxiv.org/abs/2205.14135


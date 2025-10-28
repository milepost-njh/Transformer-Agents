# 推理相关文档索引

本目录包含所有与模型推理相关的技术文档。

## 📚 KV-cache相关文档

### 核心文档（按推荐阅读顺序）

1. **[KV-cache使用教程](KV-cache使用教程.md)** ⭐ **推荐从这里开始**
   - ✅ 无需重新训练说明
   - 🚀 3步快速上手
   - 📚 详细的使用方法和代码示例
   - 💡 性能优化建议
   - ❓ 常见问题解答

2. **[KV-cache实现指南](KV-cache实现指南.md)**
   - 🔧 技术实现细节
   - 📝 阶段一：模拟KV-cache（显存测量）
   - ⚡ 阶段二：真正的KV-cache（推理加速）
   - 🎯 实现原理说明

3. **[KV-cache存储量计算](KV-cache存储量计算.md)**
   - 🧮 显存占用计算公式
   - 📊 标准注意力 vs MLA 对比
   - 📈 实际案例分析

4. **[精度统一说明](精度统一说明.md)**
   - 🎯 bf16/fp16精度说明
   - 🔄 混合精度训练

## 📖 文档阅读顺序建议

### 新用户快速上手：
1. [KV-cache使用教程](KV-cache使用教程.md) - **从这里开始**（包含3步快速上手）
2. [KV-cache实现指南](KV-cache实现指南.md) - 深入理解原理

### 性能优化：
1. [KV-cache存储量计算](KV-cache存储量计算.md) - 了解显存占用
2. [KV-cache使用教程](KV-cache使用教程.md) - 学习优化技巧
3. [精度统一说明](精度统一说明.md) - 选择合适的精度

### 深入开发：
1. [KV-cache实现指南](KV-cache实现指南.md) - 了解实现细节
2. 阅读源码：`core/models/kv_cache.py` 和 `train_tmp.py`
3. 参考脚本：`inference/compare_kv_cache_mla.py`

## 🔗 相关代码文件

- `core/models/kv_cache.py` - KV-cache核心实现
- `inference/compare_kv_cache_mla.py` - KV-cache推理和性能对比脚本

## 💡 快速链接

- **测试现有MLA模型**：直接运行 `python inference/compare_kv_cache_mla.py --mla_checkpoint YOUR_MODEL.pt --test_lengths 64`
- **对比MLA vs 标准注意力**：运行 `python inference/compare_kv_cache_mla.py --mla_checkpoint MLA.pt --no_mla_checkpoint STD.pt --test_lengths 64 128`

## ❓ 常见问题

**Q: 已有模型需要重新训练吗？**  
A: 不需要！KV-cache是推理优化，现有checkpoint可以直接使用。

**Q: 选择哪种Cache类型？**  
A: 
- 常规推理：`DynamicCache`（推荐）
- torch.compile优化：`StaticCache`
- MLA模型：`MLACache`（自动）

**Q: 性能提升有多少？**  
A: 长序列生成可提速2-5倍，序列越长提升越明显。

---

**文档维护日期**: 2025-10-28  
**文档版本**: 1.0


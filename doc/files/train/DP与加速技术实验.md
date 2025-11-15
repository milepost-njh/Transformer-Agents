# DP 与加速技术实验记录

## 一、技术栈配置

### 硬件环境
- **GPU**: 5×NVIDIA L20 (46GB显存/卡)
- **卡号**: 1,2,5,6,7
- **CUDA**: 支持 bfloat16

### 软件环境
- **PyTorch**: 2.0+
- **flash-attn**: 2.7.4
- **并行模式**: DataParallel (DP)

---

## 二、训练配置

### 模型参数
```python
num_layers = 8
d_model = 512
num_heads = 8
dff = 2048
max_length = 64
vocab_size = 8192
```

### 启用的技术
- ✅ **MLA** (Multi-head Latent Attention)
- ✅ **MoE** (8 experts, top-2)
- ✅ **MTP** (Multi-Token Prediction)
- ✅ **RoPE** 位置编码
- ✅ **RMSNorm**
- ✅ **bfloat16** 混合精度

---

## 三、加速技术对比实验

### 3.1 基础配置 (Baseline)
```python
batch_size = 32
num_workers = 0
torch.compile = False
flash_attn = False
```

**性能指标：**
- 吞吐量: **27.8 samples/sec**
- 每100 batch: **115秒**
- GPU利用率: **8-9%**
- 显存使用: **7.4GB/卡**

**瓶颈分析：**
- ❌ batch_size 太小，显存利用率低 (16%)
- ❌ num_workers=0，GPU 等待数据加载
- ❌ 没有算子优化

---

### 3.2 优化 #1: 增大 Batch + 并行加载

```python
batch_size = 64          # 32 → 64 (2倍)
num_workers = 4          # 0 → 4
prefetch_factor = 2      # 预取2个batch
persistent_workers = True
```

**性能指标：**
- 吞吐量: **56.1 samples/sec** ⬆️ **+101%**
- 每100 batch: **114秒** (但处理样本翻倍)
- GPU利用率: 保持低位 (数据加载优化后会提升)
- 显存使用: 7.4GB/卡

**结论：**
- ✅ 吞吐量提升 **2倍**
- ✅ 显存仍有余量 (16%)，可继续增大 batch

---

### 3.3 优化 #2: torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

**特点：**
- 首次编译开销: **~30秒**
- 编译后稳定: 每100 batch **110-112秒**
- 总训练时间: ~20小时，编译占比 **0.04%** (可忽略)

**性能变化：**
```
Batch 100→200: 114秒
Batch 200→300: 112秒
Batch 300→400: 110秒  ← 逐渐加快
Batch 400→500: 110秒  ← 稳定
```

**结论：**
- ✅ 稳定后有 **~4%** 加速 (114→110秒)
- ✅ 长时间训练建议启用
- ⚠️ 首次编译有警告 (recompile_limit)，但不影响后续训练

---

### 3.4 优化 #3: Flash Attention (即将测试)

**预期效果：** (基于论文和社区实验)
- **注意力加速**: 2-4倍
- **显存减少**: 5-20倍
- **支持更长序列**: 内存复杂度 O(N) vs O(N²)

**预测指标：**
- 吞吐量: 56 → **112-224 samples/sec** (2-4倍)
- 每100 batch: 110秒 → **30-60秒**
- 显存: 7.4GB → **5-6GB** (可增大 batch 到 128+)

**待验证...**

---

## 四、已知问题与解决

### 4.1 bfloat16 溢出
**问题：**
```
RuntimeError: value cannot be converted to type at::Half without overflow
```

**原因：** mask 填充值 `-1e9` 超出 bf16 范围

**解决：**
```python
mask_value = -1e4 if q.dtype in [torch.float16, torch.bfloat16] else -1e9
```

### 4.2 MTP 日志刷屏
**问题：** DataParallel 模式下每个 GPU 副本都打印日志

**解决：** 使用类级别标志，只打印一次
```python
if not hasattr(DeepSeekMTPWrapper, '_first_forward_logged'):
    logger.info(...)
    DeepSeekMTPWrapper._first_forward_logged = True
```

---

## 五、运行指令

### 启动训练
```bash
CUDA_VISIBLE_DEVICES=1,2,5,6,7 python train_dp_latest.py
```

### 后台运行
```bash
./run_dp.sh
tail -f logs/train_dp_*.log
```

---

## 六、性能优化路线图

### ✅ 已完成 (累计加速: ~2倍)
1. Batch Size 优化 → **2倍吞吐量**
2. 并行数据加载 → **消除数据瓶颈**
3. torch.compile → **4% 额外加速**
4. Fused AdamW → **优化器加速**

### 🔄 测试中
5. Flash Attention → **预期 2-4倍加速**

### 📋 待实施 (可选)
6. 梯度累积 → 模拟更大 batch
7. 激活检查点 → 节省显存 30-50%
8. 增大 batch_size 到 128/256 → 充分利用显存

---

## 七、关键配置总结

```python
# 训练配置
epochs = 20
batch_size = 64
learning_rate = 1e-4
num_workers = 4
prefetch_factor = 2

# 模型配置
use_mla = True    # Multi-head Latent Attention
use_mtp = True    # Multi-Token Prediction  
use_moe = True    # Mixture of Experts
use_rope = True   # Rotary Position Embedding

# 加速技术
torch.compile(model, mode="reduce-overhead")  # 启用
flash_attention = True  # 自动启用（如果可用）
mixed_precision = "bfloat16"  # 自动启用
```

---

## 八、实验结论

### DataParallel vs DDP
- **DP 优势**: 启动简单 (`python` 直接运行，无需 `torchrun`)
- **DP 劣势**: 单进程，GPU 0 负载略高，扩展性不如 DDP
- **适用场景**: ≤8 GPU，快速实验

### torch.compile
- ✅ **推荐启用**：编译开销可忽略 (0.04%)
- ✅ 稳定后有 **~4%** 加速
- ⚠️ 首次编译约 30 秒

### Flash Attention
- ✅ **强烈推荐**：理论 2-4倍加速
- ✅ 已在代码中集成，自动启用
- ⚠️ 需要 fp16/bf16 + CUDA

### 性能提升路径
```
Baseline (27.8 samples/s)
  ↓ +Batch+DataLoader → 56.1 samples/s (+101%)
  ↓ +torch.compile    → 58.3 samples/s (+4%)
  ↓ +Flash Attention  → 116-233 samples/s (+100-300%) [待测]
```

**总预期加速**: **4-8倍** 🚀


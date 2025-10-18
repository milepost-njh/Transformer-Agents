# Transformer多卡训练实现指南

## 概述

本文档基于 `train_rope_moe_parall.py` 和 `train_rope_moe.py` 的对比分析，详细介绍了如何实现Transformer模型的多卡训练。多卡训练可以显著提升训练速度，特别是在处理大规模模型和数据集时。

## 核心差异对比

### 1. 环境设置

#### 单卡版本 (`train_rope_moe.py`)
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用一张GPU
```

#### 多卡版本 (`train_rope_moe_parall.py`)
```python
# 多卡训练设置
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

# 设置可见的GPU，可以根据需要修改
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5,6,7"  # 使用6张GPU

# 修复警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizers并行以避免fork警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用TensorFlow oneDNN优化信息
```

### 2. 多卡训练核心函数

#### 2.1 多卡环境检测和设置
```python
def setup_multi_gpu():
    """
    设置多卡训练环境
    返回: (device, use_multi_gpu, gpu_count)
    """
    if not torch.cuda.is_available():
        logger.info("CUDA不可用，使用CPU训练")
        return torch.device("cpu"), False, 0
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_count} 张GPU")
    
    if gpu_count > 1:
        logger.info(f"启用多卡训练，使用 {gpu_count} 张GPU")
        device = torch.device("cuda:0")  # 主GPU
        return device, True, gpu_count
    else:
        logger.info("只有1张GPU，使用单卡训练")
        device = torch.device("cuda:0")
        return device, False, 1
```

#### 2.2 模型多卡包装
```python
def wrap_model_for_multi_gpu(model, use_multi_gpu, gpu_count):
    """
    为多卡训练包装模型
    """
    if use_multi_gpu and gpu_count > 1:
        logger.info(f"使用DataParallel包装模型，GPU数量: {gpu_count}")
        # 设置DataParallel的device_ids参数
        device_ids = list(range(gpu_count))
        model = DataParallel(model, device_ids=device_ids)
        return model
    else:
        logger.info("使用单卡训练")
        return model
```

#### 2.3 GPU内存监控
```python
def log_gpu_memory_usage(step_name="", use_multi_gpu=False, gpu_count=1):
    """
    记录GPU内存使用情况
    """
    if not torch.cuda.is_available():
        return
    
    if use_multi_gpu and gpu_count > 1:
        for i in range(gpu_count):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            logger.info(f"GPU {i} {step_name} - 内存使用: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
    else:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        logger.info(f"GPU {step_name} - 内存使用: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
```

### 3. DataLoader多卡优化

#### 3.1 多卡DataLoader构建
```python
def build_dataloaders(
        train_dataset,
        val_dataset,
        pt_tokenizer,
        en_tokenizer,
        batch_size: int = 64,
        max_length: int = 48,
        num_workers: int = 0,
        shuffle_train: bool = True,
        use_multi_gpu: bool = False,  # 新增参数
        gpu_count: int = 1,           # 新增参数
):
    # 多卡训练时调整batch size和num_workers
    if use_multi_gpu and gpu_count > 1:
        # 增加batch size以充分利用多GPU
        effective_batch_size = batch_size * gpu_count
        # 增加数据加载worker数量
        effective_num_workers = min(num_workers * 2, 8)  # 最多8个worker
        logger.info(f"多卡训练优化: batch_size {batch_size} -> {effective_batch_size}, num_workers {num_workers} -> {effective_num_workers}")
    else:
        effective_batch_size = batch_size
        effective_num_workers = num_workers

    # DataLoader配置
    train_loader = DataLoader(
        PairsDataset(train_pairs),
        batch_size=effective_batch_size,
        shuffle=shuffle_train,
        collate_fn=lambda b: collate_padded(b, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id),
        num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # 多卡训练时启用pin_memory
    )
    # ... 验证集DataLoader类似配置
```

### 4. 训练步骤多卡适配

#### 4.1 改进的训练步骤
```python
def train_step(batch, transformer, optimizer, scheduler=None, device=None, moe_config=None, use_multi_gpu=False):
    # ... 前向传播代码 ...
    
    # 改进的梯度裁剪策略
    # 1. 先计算梯度范数
    # 对于DataParallel包装的模型，需要访问module属性
    model_for_grad_clip = transformer.module if use_multi_gpu else transformer
    grad_norm = torch.nn.utils.clip_grad_norm_(model_for_grad_clip.parameters(), max_norm=0.5)
    
    # 2. 更严格的梯度监控
    if grad_norm > 5.0:
        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")
        # 如果梯度范数过大，进一步裁剪
        torch.nn.utils.clip_grad_norm_(model_for_grad_clip.parameters(), max_norm=0.1)
        logger.warning(f"Applied additional gradient clipping to 0.1")
    
    # 3. 检查是否有NaN梯度
    has_nan_grad = False
    for name, param in model_for_grad_clip.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.error(f"NaN gradient detected in {name}")
            has_nan_grad = True
            break
    
    if has_nan_grad:
        logger.error("Skipping this batch due to NaN gradients")
        return 0.0, 0.0

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    acc = token_accuracy(tar_real, logits, pad_id=TGT_PAD_ID)
    return loss.item(), acc
```

### 5. Checkpoint多卡支持

#### 5.1 多卡模型保存
```python
def save_ckpt(model, optimizer, scheduler, epoch, step, ckpt_dir="checkpoints", tag="latest", use_multi_gpu=False):
    """
    保存 checkpoint
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 对于DataParallel包装的模型，保存原始模型的状态
    model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()
    
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model_state,
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
        "use_multi_gpu": use_multi_gpu,
    }
    # ... 保存逻辑 ...
```

#### 5.2 多卡模型加载
```python
def load_ckpt(model, optimizer=None, scheduler=None, ckpt_dir="checkpoints", device="cpu", use_multi_gpu=False):
    """
    加载最新 checkpoint
    """
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        logger.info("⚠️ No checkpoint found, training from scratch.")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)
    
    # 对于DataParallel包装的模型，加载到原始模型
    target_model = model.module if use_multi_gpu else model
    target_model.load_state_dict(ckpt["model"])
    
    if optimizer: optimizer.load_state_dict(ckpt["optim"])
    if scheduler and ckpt["sched"]: scheduler.load_state_dict(ckpt["sched"])
    
    saved_multi_gpu = ckpt.get("use_multi_gpu", False)
    logger.info(f"✅ checkpoint loaded (epoch={ckpt['epoch']}, step={ckpt['step']}, multi_gpu={saved_multi_gpu})")
    return ckpt["epoch"], ckpt["step"]
```

### 6. 主训练流程多卡集成

#### 6.1 多卡训练主流程
```python
if __name__ == "__main__":
    # ... 配置参数 ...
    
    # 1.1 设置多卡训练
    device, use_multi_gpu, gpu_count = setup_multi_gpu()
    logger.info(f"多卡训练设置: use_multi_gpu={use_multi_gpu}, gpu_count={gpu_count}")

    # ... 数据加载和模型构建 ...
    
    # 1.2 为多卡训练包装模型
    model = wrap_model_for_multi_gpu(model, use_multi_gpu, gpu_count)

    # ... 优化器和调度器设置 ...
    
    # 3.3 构建 batch data loader（多卡优化）
    train_loader2, val_loader2 = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=4,  # 增加数据加载worker数量
        shuffle_train=True,
        use_multi_gpu=use_multi_gpu,  # 传递多卡标志
        gpu_count=gpu_count          # 传递GPU数量
    )

    # 8. 训练模型（多卡支持）
    train_model(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader2,
        val_loader=val_loader2,
        scheduler=scheduler,
        device=device,
        log_every=100,
        ckpt_dir="checkpoints",
        ckpt_prefix="transformer",
        tensorboard_dir="runs",
        moe_config=moe_config,
        use_multi_gpu=use_multi_gpu,  # 传递多卡标志
    )
```

## 多卡训练优势

### 1. 性能提升
- **并行计算**：多张GPU同时处理不同的batch
- **内存扩展**：总GPU内存 = 单卡内存 × GPU数量
- **训练加速**：理论上可获得接近线性的加速比

### 2. 内存优化
- **Batch Size扩展**：多卡训练时有效batch size = 单卡batch size × GPU数量
- **数据并行**：每个GPU处理不同的数据子集
- **梯度聚合**：所有GPU的梯度自动聚合后更新

### 3. 稳定性改进
- **梯度裁剪**：针对多卡环境优化的梯度裁剪策略
- **NaN检测**：增强的异常检测和处理机制
- **内存监控**：实时监控所有GPU的内存使用情况

## 使用建议

### 1. 硬件要求
- 至少2张NVIDIA GPU（推荐4张或更多）
- 足够的GPU内存（每张GPU至少8GB）
- 高速GPU间通信（NVLink或PCIe）

### 2. 配置优化
- 根据GPU数量调整batch size
- 适当增加num_workers以充分利用CPU
- 启用pin_memory提高数据传输效率

### 3. 监控要点
- 监控所有GPU的内存使用
- 观察梯度范数变化
- 检查训练速度和收敛性

### 4. 故障排除
- 如果出现OOM错误，减少batch size
- 如果梯度爆炸，调整学习率或梯度裁剪阈值
- 如果训练不稳定，检查数据加载和模型初始化

## 总结

多卡训练实现通过以下关键技术实现了显著的性能提升：

1. **DataParallel包装**：自动将模型分布到多张GPU
2. **Batch Size扩展**：充分利用多GPU的计算能力
3. **内存优化**：pin_memory和worker数量优化
4. **梯度处理**：改进的梯度裁剪和异常检测
5. **Checkpoint支持**：完整的多卡模型保存和加载

通过对比单卡和多卡版本，可以看出多卡训练在保持代码结构清晰的同时，显著提升了训练效率和稳定性。

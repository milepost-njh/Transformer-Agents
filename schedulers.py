# coding=utf-8
import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau as TorchReduceLROnPlateau

# 导入transformers调度器作为备用
try:
    from transformers import get_cosine_schedule_with_warmup as transformers_cosine_warmup
except ImportError:
    transformers_cosine_warmup = None


class CosineWarmupLR(_LRScheduler):
    """
    线性预热 + 余弦衰减学习率调度器

    lr(t) = base_lr * scale(t)
      其中:
        if t < warmup_steps: scale = t / warmup_steps
        else:                 scale = 0.5 * (1 + cos(pi * progress))

    progress 在 [warmup_steps, total_steps] 范围内从 0 到 1
    支持可选的 num_cycles 参数来设置多个余弦半周期
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_lr: float = 0.0,
    ) -> None:
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.num_training_steps = max(1, int(num_training_steps))
        self.num_cycles = float(num_cycles)
        self.min_lr = float(min_lr)
        # 记录基础学习率 (PyTorch已经在_LRScheduler中填充了base_lrs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = min(self.last_epoch + 1, self.num_training_steps)
        if self.num_warmup_steps > 0 and step < self.num_warmup_steps:
            scale = float(step) / float(max(1, self.num_warmup_steps))
        else:
            if self.num_training_steps == self.num_warmup_steps:
                progress = 1.0
            else:
                progress = float(step - self.num_warmup_steps) / float(
                    max(1, self.num_training_steps - self.num_warmup_steps)
                )
            scale = 0.5 * (1.0 + math.cos(math.pi * (2.0 * self.num_cycles * progress)))

        lrs = []
        for base_lr in self.base_lrs:
            lr = max(self.min_lr, base_lr * scale)
            lrs.append(lr)
        return lrs


class OneCycleLRCustom(_LRScheduler):
    """
    轻量级单周期学习率调度器 (仅学习率)

    - 从 max_lr / div_factor 开始
    - 在 pct_start * total_steps 步内预热到 max_lr
    - 其余时间余弦退火到 max_lr / final_div_factor

    注意: 这是简化版本，专注于学习率调度 (无动量调度)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.1,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ) -> None:
        assert total_steps > 0, "total_steps must be > 0"
        assert 0.0 < pct_start < 1.0, "pct_start must be in (0,1)"
        self.total_steps = int(total_steps)
        self.warmup_steps = max(1, int(self.total_steps * float(pct_start)))
        self.max_lr = float(max_lr)
        self.start_lr = float(max_lr) / float(div_factor)
        self.end_lr = float(max_lr) / float(final_div_factor)

        # 初始化基础学习率为start_lr，但记住每个参数组特定的max_lr
        self._per_group_max = []
        for group in optimizer.param_groups:
            group_max_lr = group.get("max_lr", self.max_lr)
            self._per_group_max.append(group_max_lr)
            group.setdefault("initial_lr", self.start_lr)
            group["lr"] = self.start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = min(self.last_epoch + 1, self.total_steps)
        lrs: List[float] = []
        for base_lr, group_max_lr in zip(self.base_lrs, self._per_group_max):
            max_lr = group_max_lr
            start_lr = max_lr * (self.start_lr / self.max_lr)
            end_lr = max_lr * (self.end_lr / self.max_lr)

            if step <= self.warmup_steps:
                # 从start_lr到max_lr的线性预热
                scale = float(step) / float(self.warmup_steps)
                lr = start_lr + scale * (max_lr - start_lr)
            else:
                # 从max_lr到end_lr的余弦退火
                progress = float(step - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = end_lr + (max_lr - end_lr) * cosine

            lrs.append(lr)
        return lrs


class ReduceLROnPlateauWrapper:
    """
    围绕torch.optim.lr_scheduler.ReduceLROnPlateau的轻量级包装器，
    保持一致的API接口，支持step(metrics)并方便获取当前学习率
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 0.0,
        verbose: bool = False,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
    ) -> None:
        self._scheduler = TorchReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def step(self, metrics: float) -> None:
        self._scheduler.step(metrics)

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class MoEWarmupCosineLR(_LRScheduler):
    """
    MoE感知的预热 + 余弦衰减调度器

    - 在warmup_ratio * total_steps步内线性预热
    - 之后进行余弦衰减
    - 保留每个参数的初始学习率(如果提供)

    这与CosineWarmupLR类似，但单独提供以允许未来的
    MoE特定扩展(例如，路由参数的不同缩放)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.2,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_lr: float = 0.0,
    ) -> None:
        assert total_steps > 0
        self.total_steps = int(total_steps)
        self.warmup_steps = max(1, int(self.total_steps * float(warmup_ratio)))
        self.num_cycles = float(num_cycles)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = min(self.last_epoch + 1, self.total_steps)
        if step <= self.warmup_steps:
            scale = float(step) / float(self.warmup_steps)
        else:
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * (2.0 * self.num_cycles * progress)))

        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


class TransformersCosineWarmup:
    """
    围绕transformers.get_cosine_schedule_with_warmup的包装器，保持一致的API接口
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        if transformers_cosine_warmup is None:
            raise ImportError("transformers库不可用")
        
        self._scheduler = transformers_cosine_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
        self.optimizer = optimizer
    
    def step(self) -> None:
        self._scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        return self._scheduler.get_last_lr()


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_training_steps: int,
    learning_rate: float = 1e-3,
    warmup_ratio: float = 0.15,
    **kwargs
):
    """
    创建不同类型调度器的工厂函数
    
    参数:
        scheduler_type: 要创建的调度器类型
            - "transformers_cosine": transformers.get_cosine_schedule_with_warmup
            - "cosine_warmup": 自定义CosineWarmupLR
            - "onecycle": OneCycleLRCustom
            - "reduce_on_plateau": ReduceLROnPlateauWrapper
            - "moe_cosine": MoEWarmupCosineLR
        optimizer: PyTorch优化器
        num_training_steps: 总训练步数
        learning_rate: 最大学习率
        warmup_ratio: 用于预热的步数比例
        **kwargs: 特定调度器的额外参数
    
    返回:
        调度器对象
    """
    warmup_steps = int(warmup_ratio * num_training_steps)
    
    if scheduler_type == "transformers_cosine":
        return TransformersCosineWarmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    
    elif scheduler_type == "cosine_warmup":
        return CosineWarmupLR(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=kwargs.get("num_cycles", 0.5),
            min_lr=kwargs.get("min_lr", 0.0),
        )
    
    elif scheduler_type == "onecycle":
        return OneCycleLRCustom(
            optimizer=optimizer,
            max_lr=learning_rate,
            total_steps=num_training_steps,
            pct_start=warmup_ratio,
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
        )
    
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateauWrapper(
            optimizer=optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 3),
            min_lr=kwargs.get("min_lr", 0.0),
            verbose=kwargs.get("verbose", True),
        )
    
    elif scheduler_type == "moe_cosine":
        return MoEWarmupCosineLR(
            optimizer=optimizer,
            total_steps=num_training_steps,
            warmup_ratio=warmup_ratio,
            num_cycles=kwargs.get("num_cycles", 0.5),
            min_lr=kwargs.get("min_lr", 0.0),
        )
    
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")


__all__ = [
    "CosineWarmupLR",
    "OneCycleLRCustom", 
    "ReduceLROnPlateauWrapper",
    "MoEWarmupCosineLR",
    "TransformersCosineWarmup",
    "create_scheduler",
]



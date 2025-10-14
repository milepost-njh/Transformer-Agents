#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 16:35
@Author  : nijiahui
@FileName: optimizer.py.py
@Software: PyCharm
 
'''

import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

class CustomizedSchedule(_LRScheduler):
    """
    Noam / Transformer LR:
      lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # 确保从 1 开始
        scale = self.d_model ** -0.5
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]


def plot_customized_lr_curve(optimizer, scheduler, total_steps: int, label: str = None):
    """
    绘制学习率曲线（支持传入已有 optimizer 和 scheduler）

    Args:
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        total_steps (int): 总训练步数
        label (str): 图例标签，默认使用 scheduler 配置
    """
    lrs = []
    for step in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

    # 绘制曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, total_steps + 1), lrs, label=label or "LR Curve")
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()
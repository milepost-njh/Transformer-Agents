#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 17:27
@Author  : nijiahui
@FileName: utils.py
@Software: PyCharm
 
'''
import os
import torch
from datetime import datetime

def save_ckpt(model, optimizer, scheduler, epoch, step, ckpt_dir="checkpoints", tag="latest"):
    """
    保存 checkpoint
    Args:
        model: nn.Module
        optimizer: torch.optim
        scheduler: torch.optim.lr_scheduler (可选)
        epoch: 当前 epoch
        step: 全局 step
        ckpt_dir: 保存目录
        tag: 保存标识 ("latest", "error", "custom" 等)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
    }

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(ckpt, latest_path)
    # print(f"✅ checkpoint updated: {latest_path}")

    # 1. 默认保存 latest
    if tag == "latest":
        path = os.path.join(ckpt_dir, f"mid_e{epoch}_s{step}.pt")

    elif tag == "error":
        # 避免覆盖，用时间戳
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ckpt_dir, f"error_e{epoch}_s{step}_{ts}.pt")
    else:
        path = os.path.join(ckpt_dir, f"{tag}_e{epoch}_s{step}.pt")

    torch.save(ckpt, path)
    # print(f"✅ checkpoint saved: {path}")
    return path


def load_ckpt(model, optimizer=None, scheduler=None, ckpt_dir="checkpoints", device="cpu"):
    """
    加载最新 checkpoint
    """
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        print("⚠️ No checkpoint found, training from scratch.")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer: optimizer.load_state_dict(ckpt["optim"])
    if scheduler and ckpt["sched"]: scheduler.load_state_dict(ckpt["sched"])
    print(f"✅ checkpoint loaded (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]
#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/14 11:28
@Author  : nijiahui
@FileName: utils.py
@Software: PyCharm
 
'''
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl


def check_env():
    """
    检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本。
    返回推荐的 device ('cuda' 或 'cpu')。
    """
    print("===== PyTorch & 系统信息 =====")
    print("torch.__version__:", torch.__version__)
    print("python version:", sys.version_info)

    print("\n===== 常用库版本 =====")
    for module in (mpl, np, pd, torch):
        print(module.__name__, module.__version__)

    print("\n===== GPU 检查 =====")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    try:
        print("cudnn version:", torch.backends.cudnn.version())
    except Exception as e:
        print("cudnn version: N/A", e)

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Current device id:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("bfloat16 supported:", torch.cuda.is_bf16_supported())

        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        device = "cuda"
    else:
        print("⚠️ 没检测到 CUDA，可强制 device='cpu' 运行，但速度会慢")
        device = "cpu"

    print("\n推荐使用 device: Cuda;")
    return device

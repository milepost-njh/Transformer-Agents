# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
from loguru import logger

# 指定使用GPU卡2
device = torch.device('cuda:2')


def larger_model_amp_benchmark(enabled=True, n=100):
    # 使用更大的模型
    model = nn.Sequential(
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192),
        nn.ReLU(),
        nn.Linear(8192, 8192),
    ).to(device)  # 使用指定的device

    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(128, 8192, device=device)  # 使用指定的device

    torch.cuda.synchronize(device)  # 同步指定的GPU
    start = time.time()

    for _ in range(n):
        with torch.amp.autocast('cuda', enabled=enabled):
            y = model(x)
            loss = y.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device)  # 同步指定的GPU
    return (time.time() - start) / n


# 多次测试取平均值
for enabled in [False, True]:
    times = []
    for _ in range(10):  # 运行10次取平均
        time_taken = larger_model_amp_benchmark(enabled)
        times.append(time_taken)

    avg_time = sum(times) / len(times)
    logger.info(f"enabled={enabled}, Avg Time: {avg_time:.6f}s (±{max(times) - min(times):.6f}s)")

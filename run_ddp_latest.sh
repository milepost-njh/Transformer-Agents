#!/bin/bash

set -euo pipefail

mkdir -p logs

# 写死使用1,2,5,6,7号卡训练
export CUDA_VISIBLE_DEVICES=1,2,5,6,7
NPROC=5

# 分布式必备环境，提升稳定性
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # torch 2.8+ 替代 NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# 后台运行，日志写入文件
LOG_FILE=logs/train_ddp_latest_$(date +%Y%m%d_%H%M%S).log
(
  set -x
  torchrun --standalone --nproc_per_node="${NPROC}" \
    train_ddp_latest.py "$@"
) >"${LOG_FILE}" 2>&1 &

PID=$!
echo
echo "训练进程已启动 (DDP, nproc=${NPROC}), PID: ${PID}"
echo "日志: ${LOG_FILE}"
echo "实时查看: tail -f ${LOG_FILE}"
echo "停止训练: kill ${PID}"

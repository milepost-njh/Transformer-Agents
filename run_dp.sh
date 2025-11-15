#!/bin/bash
# 真正的 DataParallel 模式（单进程多GPU，不需要 torchrun）

set -euo pipefail

mkdir -p logs

# 设置要使用的 GPU (使用 1、2、5、6、7 编号的卡)
export CUDA_VISIBLE_DEVICES=1,2,5,6,7

# 后台运行，日志写入文件
LOG_FILE=logs/train_dp_$(date +%Y%m%d_%H%M%S).log
(
  set -x
  python train_dp_latest.py "$@"
) >"${LOG_FILE}" 2>&1 &

PID=$!
echo
echo "训练进程已启动 (DataParallel 模式, 单进程多GPU), PID: ${PID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "日志: ${LOG_FILE}"
echo "实时查看: tail -f ${LOG_FILE}"
echo "停止训练: kill ${PID}"


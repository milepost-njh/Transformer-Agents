#!/bin/bash

# Kimi 模型训练启动脚本
# 使用 nohup 后台运行，日志保存到 logs/ 目录

# 创建日志目录
mkdir -p logs

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_kimi_${TIMESTAMP}.log"

echo "🚀 Starting Kimi training..."
echo "📝 Log file: ${LOG_FILE}"

# 使用 nohup 后台运行训练脚本
nohup python train_kimi.py > "${LOG_FILE}" 2>&1 &

# 获取进程 PID
PID=$!
echo "✅ Training started with PID: ${PID}"
echo "${PID}" > logs/train_kimi.pid

echo ""
echo "📊 查看训练日志: tail -f ${LOG_FILE}"
echo "🛑 停止训练: kill ${PID}"
echo "   或: kill \$(cat logs/train_kimi.pid)"


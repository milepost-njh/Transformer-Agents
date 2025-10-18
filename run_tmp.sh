#!/bin/bash

rm -rf checkpoints  

# 使用nohup在后台运行训练，即使SSH断开也不会停止
# 同时输出到日志文件和控制台
nohup python train_tmp.py > logs/train_rsmnorm.log 2>&1 &

# 获取进程ID
PID=$!
echo ""
echo "训练进程已启动，PID: $PID"
echo "使用 'tail -f logs/train_rsmnorm.log' 查看实时日志"
echo "使用 'kill ${PID}' 停止训练"

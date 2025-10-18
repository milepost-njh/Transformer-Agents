#!/bin/bash
# 使用方法: 
#   1. 取消注释想要使用的SCHEDULER行（只保留一个）
#   2. 运行 ./run_tmp.sh
# 
# 每个scheduler会使用独立的checkpoint目录和log文件

# ============================================
# 选择调度器类型（取消注释想要使用的一个）
# ============================================
# SCHEDULER="moe_cosine"          # DeepSeek MoE 余弦调度器
# SCHEDULER="onecycle"            # OneCycle 调度器
SCHEDULER="transformers_cosine" # Transformers 余弦调度器
# SCHEDULER="cosine_warmup"       # 自定义余弦预热调度器
# SCHEDULER="reduce_on_plateau"   # 自适应学习率调度器

# ============================================
# 配置
# ============================================
# 获取当前日期时间（精确到小时）
DATETIME=$(date +"%Y%m%d_%H")

# 为每个调度器创建独立的checkpoint目录和日志文件
CHECKPOINT_DIR="./checkpoints_${SCHEDULER}"
LOG_FILE="logs/log_scheduler_${SCHEDULER}_${DATETIME}.log"

# ============================================
# 执行训练
# ============================================
echo "=========================================="
echo "开始训练"
echo "调度器类型: ${SCHEDULER}"
echo "Checkpoint目录: ${CHECKPOINT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "时间: $(date)"
echo "=========================================="

# 清理之前的checkpoint目录
if [ -d "${CHECKPOINT_DIR}" ]; then
    echo "清理旧的checkpoint目录: ${CHECKPOINT_DIR}"
    rm -rf ${CHECKPOINT_DIR}
fi

# 使用nohup在后台运行训练，即使SSH断开也不会停止
# 同时输出到日志文件和控制台
nohup python train_tmp.py --scheduler ${SCHEDULER} --checkpoint_dir ${CHECKPOINT_DIR} > ${LOG_FILE} 2>&1 &

# 获取进程ID
PID=$!
echo ""
echo "训练进程已启动，PID: $PID"
echo "使用 'tail -f ${LOG_FILE}' 查看实时日志"
echo "使用 'kill ${PID}' 停止训练"

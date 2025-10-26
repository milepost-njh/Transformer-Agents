#!/bin/bash
# KV-cache 对比测试快速启动脚本

# 设置默认值
GPU_ID=${GPU_ID:-5}
MLA_CHECKPOINT=${MLA_CHECKPOINT:-"checkpoints/latest.pt"}
NO_MLA_CHECKPOINT=${NO_MLA_CHECKPOINT:-"checkpoints_no_mla/latest.pt"}
TEST_LENGTHS=${TEST_LENGTHS:-"32 64 128"}
OUTPUT_DIR=${OUTPUT_DIR:-"results"}

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/kv_cache_comparison_$TIMESTAMP.json"

echo "========================================"
echo "MLA vs 标准注意力 KV-cache 对比测试"
echo "========================================"
echo "GPU ID: $GPU_ID"
echo "MLA Checkpoint: $MLA_CHECKPOINT"
echo "Standard Checkpoint: $NO_MLA_CHECKPOINT"
echo "Test Lengths: $TEST_LENGTHS"
echo "Output File: $OUTPUT_FILE"
echo "========================================"
echo ""

# 检查checkpoint文件
if [ ! -f "$MLA_CHECKPOINT" ]; then
    echo "❌ Error: MLA checkpoint not found: $MLA_CHECKPOINT"
    exit 1
fi

if [ ! -f "$NO_MLA_CHECKPOINT" ]; then
    echo "❌ Error: Standard checkpoint not found: $NO_MLA_CHECKPOINT"
    exit 1
fi

# 运行对比测试
CUDA_VISIBLE_DEVICES=$GPU_ID python inference/compare_kv_cache_mla.py \
    --mla_checkpoint "$MLA_CHECKPOINT" \
    --no_mla_checkpoint "$NO_MLA_CHECKPOINT" \
    --test_lengths $TEST_LENGTHS \
    --output "$OUTPUT_FILE"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ 测试完成！"
    echo "结果已保存到: $OUTPUT_FILE"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "❌ 测试失败！"
    echo "========================================"
    exit 1
fi


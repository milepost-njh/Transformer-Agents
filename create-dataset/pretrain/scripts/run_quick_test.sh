#!/bin/bash
# 快速测试脚本 - 1小时完成预训练+微调流程验证

set -e  # 遇到错误立即退出

echo "=========================================="
echo "预训练+微调 快速测试"
echo "预计时间: 1小时"
echo "=========================================="

# 进入scripts目录
cd "$(dirname "$0")"

# 1. 下载示例数据（10分钟）
echo ""
echo "[1/3] 下载示例数据..."
python3 download_pretrain_data.py \
    --method sample \
    --langs en pt \
    --max-samples 10000 \
    --output-dir ../data/pretrain

# 检查数据是否下载成功
if [ ! -f "../data/pretrain/en_wiki_sample.txt" ]; then
    echo "❌ 数据下载失败"
    exit 1
fi

echo "✅ 数据下载完成"

# 2. 预训练（30分钟）
echo ""
echo "[2/3] 开始预训练..."
python3 pretrain_dae.py \
    --train-data ../data/pretrain/en_wiki_sample.txt \
    --epochs 1 \
    --batch-size 16 \
    --max-length 64 \
    --lr 1e-4 \
    --noise-type mixed \
    --checkpoint-dir ../checkpoints/pretrain

# 检查checkpoint是否生成
if [ ! -f "../checkpoints/pretrain/pretrain_latest.pt" ]; then
    echo "❌ 预训练失败"
    exit 1
fi

echo "✅ 预训练完成"

# 3. 提示微调步骤
echo ""
echo "[3/3] 预训练完成！"
echo "=========================================="
echo "✅ 成功！现在可以进行微调："
echo ""
echo "方法1: 修改 train_ddp_latest.py，加载预训练checkpoint"
echo "方法2: 使用以下命令直接微调（需要先下载翻译数据）："
echo ""
echo "cd ../../.."
echo "python train_ddp_latest.py \\"
echo "    --load-pretrain create-dataset/pretrain/checkpoints/pretrain/pretrain_latest.pt \\"
echo "    --epochs 5 \\"
echo "    --lr 5e-5"
echo ""
echo "=========================================="
echo "📊 查看预训练模型:"
echo "ls -lh ../checkpoints/pretrain/"
echo ""
echo "📈 查看训练日志:"
echo "tail -n 50 ../data/download_pretrain.log"
echo "=========================================="

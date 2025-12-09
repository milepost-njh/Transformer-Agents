#!/bin/bash
# 快速测试脚本 - 下载小数据集验证流程

echo "================================================"
echo "快速测试：下载TED2020数据集（约20万句对）"
echo "================================================"

# 1. 安装依赖
echo "检查依赖..."
pip install -q requests tqdm loguru 2>/dev/null || pip3 install -q requests tqdm loguru

# 2. 下载小数据集
echo ""
echo "下载TED2020数据集..."
python3 download_and_prepare_data.py \
    --datasets TED2020 \
    --output-dir ../test_data \
    --max-pairs 50000

# 3. 检查结果
echo ""
echo "================================================"
echo "数据准备完成！"
echo "================================================"

if [ -f "../test_data/processed/pt_en_pretrain_train.csv" ]; then
    train_lines=$(wc -l < ../test_data/processed/pt_en_pretrain_train.csv)
    val_lines=$(wc -l < ../test_data/processed/pt_en_pretrain_val.csv)
    
    echo "✅ 训练集: $train_lines 句对"
    echo "✅ 验证集: $val_lines 句对"
    echo ""
    echo "查看样本数据:"
    echo "----------------------------------------"
    head -n 3 ../test_data/processed/pt_en_pretrain_train.csv
    echo "----------------------------------------"
    echo ""
    echo "现在可以修改训练脚本使用这些数据："
    echo "train_path = 'create-dataset/finetune/test_data/processed/pt_en_pretrain_train.csv'"
    echo "val_path = 'create-dataset/finetune/test_data/processed/pt_en_pretrain_val.csv'"
else
    echo "❌ 数据准备失败，请检查日志"
fi

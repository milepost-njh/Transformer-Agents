# 🚀 预训练+微调数据准备工具

完整的两阶段训练数据准备方案

---

## 📁 项目结构

```
create-dataset/
├── pretrain/          # 预训练数据和脚本
│   ├── scripts/       # Wikipedia数据下载、预训练脚本
│   └── README.md      # 详细文档
│
└── finetune/          # 微调数据和脚本
    ├── scripts/       # 翻译数据下载
    └── README.md      # 详细文档
```

---

## 🚀 快速开始

### 阶段1：预训练
```bash
cd pretrain/scripts
bash run_quick_test.sh  # 快速测试
# 或
python3 download_pretrain_data.py --method huggingface --max-samples 300000
python3 pretrain_dae.py --train-data ../data/pretrain/en_wiki.txt --epochs 2
```

### 阶段2：微调
```bash
cd finetune/scripts
bash quick_test.sh  # 快速测试
# 或
python3 download_and_prepare_data.py --datasets Europarl TED2020 --output-dir ../data
```

---

## 📚 详细文档

- **预训练详细说明**：[pretrain/README.md](pretrain/README.md)
- **微调详细说明**：[finetune/README.md](finetune/README.md)

---

## 🎯 面试价值

✅ 完整的预训练+微调流程（现代范式）  
✅ BART风格去噪自编码实现  
✅ 20+个可讲的技术点  
✅ 面试价值：⭐⭐⭐⭐⭐

---

**预期效果**：BLEU 25-28，训练时间 2-3天

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
bash run_quick_test.sh  # 快速测试（UL2 Mixture of Denoisers）
# 或
python3 download_pretrain_data.py --method huggingface --max-samples 300000
python3 pretrain_dae.py --train-data ../data/pretrain/en_wiki.txt --epochs 2 --noise-type ul2
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

✅ **UL2 Mixture of Denoisers**（Encoder-Decoder架构最新方法）  
✅ 了解2024-2025趋势（Llama 3/DeepSeek-V3）且能合理选型  
✅ 20+个可讲的技术点 + 技术演进理解  
✅ 面试价值：⭐⭐⭐⭐⭐

> **面试话术**：采用Google 2022的UL2统一预训练范式（Encoder-Decoder最新方法），混合R/S/X三种去噪器。虽然2024-2025主流转向Decoder-only，但对于翻译任务和小参数规模，UL2 + Encoder-Decoder是最优选择，这展示了技术判断力。

### 技术演进认知

- **2019-2022**: BART → T5 → **UL2**（Encoder-Decoder最新）
- **2023-2025**: Llama 3, DeepSeek-V3（Decoder-only主流）
- **选型**: 根据任务(翻译)和资源(50M参数)选UL2，不盲目追新 ✅

---

**预期效果**：BLEU 25-28，训练时间 2-3天

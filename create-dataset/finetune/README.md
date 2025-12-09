# 微调数据准备工具

这个工具用于下载和处理葡萄牙语-英语翻译的微调数据（平行语料）。

## 功能特点

- ✅ 自动从OPUS下载高质量平行语料
- ✅ 智能数据清洗（去除URL、HTML、异常字符等）
- ✅ 数据质量过滤（长度、比例、内容检查）
- ✅ 自动去重
- ✅ 进度条显示
- ✅ 支持多数据集合并

## 安装依赖

```bash
pip install requests tqdm loguru
```

## 快速开始

### 1. 下载推荐数据集（约230万句对）

```bash
cd scripts
python3 download_and_prepare_data.py \
    --datasets Europarl TED2020 News-Commentary \
    --output-dir ../data
```

### 2. 下载更多数据集（约250万句对）

```bash
cd scripts
python3 download_and_prepare_data.py \
    --datasets Europarl TED2020 News-Commentary GlobalVoices \
    --output-dir ../data
```

### 3. 限制每个数据集的句对数

```bash
cd scripts
python3 download_and_prepare_data.py \
    --datasets Europarl OpenSubtitles \
    --max-pairs 500000 \
    --output-dir ../data
```

## 数据集说明

| 数据集 | 预期句对数 | 质量 | 推荐度 |
|--------|-----------|------|--------|
| Europarl | 200万 | 高 | ⭐⭐⭐⭐⭐ |
| TED2020 | 20万 | 高 | ⭐⭐⭐⭐⭐ |
| News-Commentary | 5万 | 高 | ⭐⭐⭐⭐ |
| GlobalVoices | 10万 | 中 | ⭐⭐⭐ |
| OpenSubtitles | 1000万 | 中 | ⭐⭐ |

## 输出文件

数据会保存在 `data/processed/` 目录下：

```
data/
├── raw/                          # 原始下载文件
│   ├── Europarl.zip
│   ├── TED2020.zip
│   └── News-Commentary.zip
├── processed/                    # 清洗后的数据
│   ├── Europarl_cleaned.tsv
│   ├── TED2020_cleaned.tsv
│   ├── News-Commentary_cleaned.tsv
│   ├── pt_en_pretrain_train.csv  # 合并的训练集
│   └── pt_en_pretrain_val.csv    # 合并的验证集
└── data_preparation.log          # 处理日志
```

## 数据清洗规则

自动过滤以下内容：

- ❌ 长度 < 5 或 > 100 tokens
- ❌ 长度比例 > 2.0
- ❌ URL、邮箱、HTML标签
- ❌ 过多标点符号（< 50%字母数字）
- ❌ 过多数字（> 50%数字）
- ❌ 重复句对

## 在训练脚本中使用

修改项目根目录的 `train_ddp_latest.py` 中的数据路径：

```python
# 原始代码（第2249-2250行）
train_path = "/workspace/tensorflow_datasets/por_en_train.csv"
val_path = "/workspace/tensorflow_datasets/por_en_test.csv"

# 改为
train_path = "create-dataset/finetune/data/processed/pt_en_pretrain_train.csv"
val_path = "create-dataset/finetune/data/processed/pt_en_pretrain_val.csv"
```

## 高级用法

### 只合并已有数据（跳过下载）

```bash
python download_and_prepare_data.py \
    --datasets Europarl TED2020 \
    --skip-download \
    --output-dir data
```

### 自定义清洗参数

编辑脚本中的 `DataCleaner` 类参数：

```python
cleaner = DataCleaner(
    min_length=5,          # 最小长度
    max_length=100,        # 最大长度
    max_length_ratio=2.0   # 最大长度比例
)
```

## 预期数据量

使用推荐配置：

```
下载前：~230万句对（原始）
清洗后：~180-200万句对（有效）
训练集：~170-190万句对（95%）
验证集：~10万句对（5%）

文件大小：约 500MB-800MB
```

## 故障排除

### 1. 下载超时

网络问题导致，可重新运行脚本，已下载的文件会自动跳过。

### 2. 解压失败

删除损坏的zip文件，重新下载：

```bash
rm data/raw/*.zip
python download_and_prepare_data.py --datasets Europarl
```

### 3. 内存不足

处理大数据集（如OpenSubtitles）时，使用 `--max-pairs` 限制：

```bash
python download_and_prepare_data.py \
    --datasets OpenSubtitles \
    --max-pairs 1000000
```

## 性能优化建议

对于50M参数的模型：

| 目标 | 数据量 | 训练步数 | 预计时间 |
|------|--------|---------|---------|
| 快速验证 | 10万句对 | 5千步 | 2小时 |
| 基础训练 | 100万句对 | 3万步 | 1天 |
| 推荐配置 | 300万句对 | 8万步 | 2-3天 |
| 充分训练 | 500万+句对 | 15万步 | 4-5天 |

## 数据质量检查

运行后检查日志文件：

```bash
tail -f data_preparation.log
```

关键指标：
- 有效句对比例应 > 70%
- 去重后保留比例应 > 90%

## 其他说明

- 所有数据来自OPUS开源语料库
- 遵循CC-BY-SA等开源协议
- 建议定期更新数据集版本

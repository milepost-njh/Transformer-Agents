# Transformer 推理系统

这是一个基于 `train_tmp.py` 训练的 Transformer 模型的推理系统，支持普通推理模式、MLA推理模式、MTP推理模式。

## 功能特性

- **多模式支持**: 普通模式、MLA模式、MTP模式
- **多种解码策略**: 贪心解码、采样解码、束搜索
- **批量推理**: 支持批量处理多个输入
- **模型比较**: 可以比较不同模式的性能
- **评估指标**: 支持BLEU、ROUGE、METEOR、TER等评估指标
- **灵活配置**: 支持预设配置和自定义配置

## 文件结构

```
inference/
├── inference.py      # 主推理脚本
├── evaluate.py       # 模型评估脚本
├── config.py         # 配置文件
└── README.md         # 说明文档
```

## 安装依赖

```bash
# 基础依赖
pip install torch transformers loguru

# 评估指标依赖（可选）
pip install nltk rouge-score
```

## 使用方法

### 1. 基础推理

```bash
# 普通模式推理
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá, como você está?"

# MLA模式推理
python inference.py --mode mla --checkpoint checkpoints/latest.pt --input "Olá, como você está?"

# MTP模式推理
python inference.py --mode mtp --checkpoint checkpoints/latest.pt --input "Olá, como você está?"

# 比较所有模式
python inference.py --mode all --checkpoint checkpoints/latest.pt --input "Olá, como você está?"
```

### 2. 不同解码策略

```bash
# 贪心解码（默认）
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá, como você está?" --decode_method greedy

# 采样解码
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá, como você está?" --decode_method sample --temperature 0.8 --top_k 50 --top_p 0.9

# 束搜索
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá, como você está?" --decode_method beam --num_beams 4
```

### 3. 批量推理

```bash
# 创建输入文件
echo "Olá, como você está?" > input.txt
echo "Bom dia!" >> input.txt
echo "Como vai?" >> input.txt

# 批量推理
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input input.txt --batch --output results.txt
```

### 4. 模型评估

```bash
# 创建测试文件（格式：源语言\t目标语言）
echo "Olá, como você está?\tHello, how are you?" > test.txt
echo "Bom dia!\tGood morning!" >> test.txt

# 评估单个模式
python evaluate.py --checkpoint checkpoints/latest.pt --test_file test.txt --mode normal

# 比较所有模式
python evaluate.py --checkpoint checkpoints/latest.pt --test_file test.txt --mode all --output evaluation_results.json
```

## 参数说明

### 推理参数

- `--mode`: 推理模式 (normal, mla, mtp, all)
- `--checkpoint`: 模型检查点路径
- `--input`: 输入文本或文件路径
- `--output`: 输出文件路径（可选）
- `--device`: 设备 (cuda, cpu, auto)
- `--max_length`: 最大生成长度
- `--temperature`: 采样温度
- `--top_k`: Top-k采样
- `--top_p`: Top-p采样
- `--num_beams`: 束搜索的束数量
- `--decode_method`: 解码方法 (greedy, sample, beam)
- `--batch`: 批量处理模式
- `--compare`: 比较模式
- `--verbose`: 详细输出

### 评估参数

- `--test_file`: 测试文件路径
- `--mode`: 评估模式
- `--output`: 结果输出文件
- `--decode_method`: 解码方法

## 配置系统

### 预设配置

```python
from inference.config import get_preset_config

# 获取小模型配置
config = get_preset_config("small")

# 获取大模型配置
config = get_preset_config("large", max_length=128)
```

### 模式配置

```python
from inference.config import get_mode_config

# 获取MLA模式配置
mla_config = get_mode_config("mla")

# 获取MTP模式配置
mtp_config = get_mode_config("mtp")
```

### 解码配置

```python
from inference.config import get_decode_config

# 获取束搜索配置
beam_config = get_decode_config("beam_search")

# 获取核采样配置
nucleus_config = get_decode_config("nucleus_sampling")
```

### 完整配置

```python
from inference.config import create_inference_config

# 创建高质量推理配置
config = create_inference_config(
    preset="large",
    mode="mla",
    decode_strategy="beam_search",
    checkpoint_path="checkpoints/latest.pt",
    num_beams=8
)
```

## 示例配置

```python
from inference.config import get_example_config

# 快速推理配置
fast_config = get_example_config("fast_inference")

# 高质量推理配置
quality_config = get_example_config("high_quality")

# 创意生成配置
creative_config = get_example_config("creative_generation")

# MTP实验配置
mtp_config = get_example_config("mtp_experimental")
```

## 评估指标

系统支持以下评估指标：

- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **TER**: Translation Error Rate

## 性能优化

### GPU加速

```bash
# 使用GPU
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá" --device cuda

# 自动检测设备
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá" --device auto
```

### 内存优化

```bash
# 使用较小的模型
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá" --max_length 32

# 使用较小的批处理大小
python inference.py --mode normal --checkpoint checkpoints/latest.pt --input "Olá" --batch_size 1
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少最大长度
   python inference.py --max_length 32
   
   # 使用CPU
   python inference.py --device cpu
   ```

2. **检查点加载失败**
   ```bash
   # 检查检查点文件是否存在
   ls -la checkpoints/
   
   # 使用详细输出查看错误
   python inference.py --verbose
   ```

3. **Tokenizer加载失败**
   ```bash
   # 检查tokenizer文件是否存在
   ls -la tok_pt/
   ls -la tok_en/
   ```

### 调试模式

```bash
# 启用详细输出
python inference.py --verbose --mode normal --checkpoint checkpoints/latest.pt --input "Olá"
```

## 扩展功能

### 自定义评估指标

```python
from inference.evaluate import Evaluator

class CustomEvaluator(Evaluator):
    def compute_custom_score(self, predictions, references):
        # 实现自定义评估指标
        pass
```

### 自定义解码策略

```python
from inference.inference import InferenceEngine

class CustomInferenceEngine(InferenceEngine):
    def custom_decode(self, encoder_input):
        # 实现自定义解码策略
        pass
```

## 许可证

本项目遵循与主项目相同的许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个推理系统。

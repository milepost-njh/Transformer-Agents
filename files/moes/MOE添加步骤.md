# Transformer添加MOE的步骤说明

## 核心思路总结
**将Transformer的FFN层替换为包含8个专家的MOE网络（DeepseekV3MoE），每个token通过路由机制选择Top-2专家进行计算。实现上需要修改feed_forward_network函数使其返回`(output, router_logits)`元组而非单一张量，然后自底向上修改EncoderLayer、DecoderLayer、EncoderModel、DecoderModel和Transformer的forward方法来收集并传递所有层的router_logits。最后在损失函数中结合主损失和负载均衡辅助损失（load_balancing_loss），用辅助损失系数0.01来平衡专家使用，避免部分专家负载过重。训练配置上需要降低学习率（1e-3→5e-4）和延长warmup（10%→15%）来稳定MOE训练。**

---

## 1. 导入MOE模块 (第27-28行)
```python
from modeling_deepseek import DeepseekV3MoE
from collections import OrderedDict
```

## 2. 新增MOE配置类 (第33-72行)
创建`MoEConfig`类,包含关键配置:
- `num_experts`: 专家数量(8)
- `num_experts_per_tok`: 每个token激活的专家数(2)
- `router_aux_loss_coef`: 路由辅助损失系数(0.01)
- `topk_method`: Top-K方法("noaux_tc")
- 其他专家配置参数

## 3. 修改feed_forward_network函数 (第773-792行)
在`feed_forward_network`函数中:
- 添加`use_moe`和`moe_config`参数
- 当`use_moe=True`时返回`DeepseekV3MoE(moe_config)`替代原来的Sequential网络

## 4. 修改EncoderLayer (第803-838行)
- 构造函数添加`use_moe`和`moe_config`参数
- forward方法处理MoE返回的`(ffn_out, router_logits)`元组
- 返回`(out2, router_logits)`如果使用MOE

## 5. 修改DecoderLayer (第852-898行)
- 构造函数添加`use_moe`和`moe_config`参数
- forward方法处理MoE返回的router_logits
- 返回`(out3, attn_weights1, attn_weights2, router_logits)`如果使用MOE

## 6. 修改EncoderModel (第903-966行)
- 构造函数添加`use_moe`和`moe_config`参数
- forward方法收集所有层的router_logits到列表
- 返回`(x, router_logits_list)`如果使用MOE

## 7. 修改DecoderModel (第977-1037行)
- 构造函数添加`use_moe`和`moe_config`参数
- forward方法收集所有层的router_logits到列表
- 返回`(x, attention_weights, router_logits_list)`如果使用MOE

## 8. 修改Transformer (第1043-1115行)
- 构造函数添加`use_moe`和`moe_config`参数,传递给encoder和decoder
- forward方法合并encoder和decoder的router_logits
- 返回`(logits, attention_weights, router_logits)`如果使用MOE

## 9. 新增负载均衡损失函数 (第1197-1234行)
新增`load_balancing_loss_func`函数计算MOE的负载均衡损失

## 10. 修改损失函数 (第1164-1195行)
在`loss_function`中:
- 添加`router_logits`和`moe_config`参数
- 计算MOE辅助损失: `aux_loss = load_balancing_loss_func(...)`
- 最终损失: `total_loss = main_loss + moe_config.router_aux_loss_coef * aux_loss`

## 11. 修改训练步骤 (第1292-1352行)
在`train_step`中:
- 添加`moe_config`参数
- 处理transformer返回的router_logits
- 将router_logits传给loss_function
- 添加GPU内存清理和梯度监控

## 12. 修改训练函数 (第1354-1483行)
在`train_model`中:
- 添加`moe_config`参数
- 传递moe_config到train_step
- 添加GPU内存监控日志

## 13. 主函数配置 (第1754-1909行)

### 13.1 创建MOE配置 (第1754-1773行)
```python
use_moe = True
moe_config = MoEConfig(
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=d_model,  # 512
    intermediate_size=dff,  # 2048
    router_aux_loss_coef=0.01,
    topk_method="noaux_tc",
)
```

### 13.2 创建MOE模型 (第1830-1844行)
```python
model = Transformer(
    ...,
    use_moe=use_moe,
    moe_config=moe_config,
)
```

### 13.3 训练配置调整
- **学习率**: 从`1e-3`降至`5e-4` (第1742行)
- **Warmup步数**: 从10%增加到15% (第1936行)
- **训练轮数**: 从50减至10 (第1738行)

### 13.4 训练模型 (第1976-1989行)
```python
train_model(
    ...,
    moe_config=moe_config,
)
```

## 核心改动总结
1. **数据流变化**: feed_forward_network函数输出从单个张量变为`(output, router_logits)`元组
2. **损失计算**: 主损失 + MOE辅助损失
3. **参数传递**: 自底向上传递`use_moe`和`moe_config`参数
4. **训练稳定性**: 降低学习率、延长warmup、增加梯度监控


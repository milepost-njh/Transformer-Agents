# MTP 调试

## 方案1：本地模拟（随机数据）

直接运行 `debug_mtp.py`，用 PyCharm 打断点调试。

## 方案2：真实数据调试

### 1. 服务器上保存数据

```bash
python test/debugs/save_real_data.py
```

生成 `test/debugs/mtp_real_data.pt`

### 2. 下载到本地

```bash
scp user@server:/path/to/test/debugs/mtp_real_data.pt \
    /Users/zys/app/Transformer-Agents/test/debugs/
```

### 3. 本地调试

运行 `debug_mtp.py`，会自动加载真实数据。

## PyCharm 调试

在标记 `🔴` 的行打断点，查看：
- `previous_hidden_states` - MTP 输入（主模型输出）
- `inputs_embeds` - MTP 输入（embedding）
- `hidden_states` - MTP 输出
- `mtp_logits` - 预测 logits


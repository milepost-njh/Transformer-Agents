## 🔥 KV-Cache核心机制分析

### 一、核心原理：K和V到底cache的是什么？

从`train_tmp.py`的`MultiHeadAttention.forward`方法（第821-829行）可以看到：

```760:829:train_tmp.py
    def forward(self, q, k, v, mask=None, return_attn: bool = True, past_key_value=None, use_cache: bool = False):
        """
        q, k, v: [B, Lq/Lk/Lv, d_model]
        mask: 期望形状为 [B, 1, Lq, Lk] 或 [B, Lq, Lk]；值为1表示屏蔽，0表示保留
        past_key_value: 可选的KV-cache，格式为(past_k, past_v)
        use_cache: 是否返回KV-cache
        return:
          output: [B, Lq, d_model]
          attention_weights (可选): [B, num_heads, Lq, Lk]
          present_key_value (可选): 当前的KV-cache
        """
        B = q.size(0)

        if self.use_mla:
            # MLA 模式：借鉴 DeepseekV3Attention
            bsz, q_len, _ = q.size()
            k_len = k.size(1)  # 获取 k 的实际序列长度

            # Q 投影：通过低秩分解
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(q)))
            q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)  # [B, H, Lq, q_head_dim]
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            # KV 投影：压缩的 KV
            compressed_kv = self.kv_a_proj_with_mqa(k)  # 注意这里使用 k 作为输入
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe.view(bsz, k_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # [B, 1, Lk, rope_dim]

            kv = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
                .view(bsz, k_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)  # [B, H, Lk, nope_dim + v_dim]
            )

            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # 应用 RoPE 到 q_pe 和 k_pe
            if self.use_rope:
                Lq = q_pe.size(2)
                Lk = k_pe.size(2)
                cos_q, sin_q = self._rope_get_cos_sin(Lq, self.qk_rope_head_dim, q.device)
                cos_k, sin_k = self._rope_get_cos_sin(Lk, self.qk_rope_head_dim, k.device)
                q_pe = self._rope_apply(q_pe, cos_q, sin_q)
                k_pe = self._rope_apply(k_pe, cos_k, sin_k)

            # 组合 query_states 和 key_states
            query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

            # k_pe 需要广播到所有头
            k_pe_expanded = k_pe.expand(bsz, self.num_heads, k_len, self.qk_rope_head_dim)
            key_states = k_pe.new_empty(bsz, self.num_heads, k_len, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe_expanded

            # 使用 value_states 作为 v
            v_states = value_states
            
            # KV-cache 处理
            if past_key_value is not None:
                # 使用缓存的K和V
                past_k, past_v = past_key_value
                key_states = torch.cat([past_k, key_states], dim=2)  # [B, H, Lk_past+Lk, head_dim]
                v_states = torch.cat([past_v, v_states], dim=2)  # [B, H, Lk_past+Lk, v_head_dim]
            
            # 保存当前的KV用于下一步
            present_key_value = (key_states, v_states) if use_cache else None
```

### 二、核心机制解析

#### 1. **Cache的是谁？**

```python
# train_tmp.py第822-829行揭示了核心机制
if past_key_value is not None:
    # 从上一步中取出已缓存的K和V
    past_k, past_v = past_key_value
    
    # 【核心】将历史K和新的K拼接起来
    key_states = torch.cat([past_k, key_states], dim=2)  # 序列维度拼接
    
    # 【核心】将历史V和新的V拼接起来  
    v_states = torch.cat([past_v, v_states], dim=2)

# 【核心】保存当前完整的K和V，供下一步使用
present_key_value = (key_states, v_states) if use_cache else None
```

**Cache的内容：**
- **K (Key)**: 形状 `[Batch, num_Heads, Seq_Len_累积, head_dim]` - 所有历史token的Key向量
- **V (Value)**: 形状 `[Batch, num_Heads, Seq_Len_累积, v_head_dim]` - 所有历史token的Value向量

#### 2. **为什么只cache K和V，不cache Q？**

看第266-273行的推理逻辑：

```209:273:inference/compare_kv_cache_mla.py
           if use_real_cache:
                # ====== 真正的KV-cache实现 ======
                logger.info(f"  [真正Cache] 开始生成（Prefill + Decode两阶段）")
                
                # Prefill阶段：运行encoder一次，生成encoder cache
                enc_pad_mask, _, _ = create_masks(
                    encoder_input, decoder_input,
                    src_pad_id=self.pt_tokenizer.pad_token_id,
                    tgt_pad_id=self.en_tokenizer.pad_token_id,
                )
                
                # Encoder只运行一次（prefill）
                logger.info(f"  [Prefill阶段] Encoder输入长度: {encoder_input.size(1)} tokens")
                enc_output = self.model.encoder_model(
                    encoder_input, 
                    src_mask=enc_pad_mask,
                    use_cache=True
                )
                logger.info(f"  [Prefill阶段] Encoder运行完成，后续将复用encoder输出")
                
                # 处理encoder输出
                if isinstance(enc_output, tuple):
                    if len(enc_output) == 3:
                        encoder_outputs, _, encoder_cache = enc_output
                    else:
                        encoder_outputs, encoder_cache = enc_output
                else:
                    encoder_outputs = enc_output
                    encoder_cache = None
                
                # KV-cache: 只需要decoder的self-attention cache
                # encoder cache在encoder内部使用，cross-attention不需要cache（因为encoder输出固定）
                past_key_values = {"encoder": None, "decoder": None}
                
                # Decode阶段：逐token生成
                logger.info(f"  [Decode阶段] 开始逐token生成，目标生成 {max_new_tokens} tokens")
                for step in range(max_new_tokens):
                    step_start = time.time()
                    
                    # 记录KV-cache大小
                    current_seq_len = decoder_input.size(1)
                    cache_size = self.kv_tracker.record_step(current_seq_len)
                    
                    # 创建masks（只需要decoder的mask）
                    _, dec_mask, enc_dec_pad_mask = create_masks(
                        encoder_input, decoder_input,
                        src_pad_id=self.pt_tokenizer.pad_token_id,
                        tgt_pad_id=self.en_tokenizer.pad_token_id,
                    )
                    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)
                    
                    # 第一步：完整输入；后续步骤：只输入最后一个token
                    if step == 0:
                        # 第一步：输入完整的start token
                        current_decoder_input = decoder_input
                        if step % 20 == 0:
                            logger.info(f"    Step {step}: Decoder输入长度={current_decoder_input.size(1)} | 累积序列长度={decoder_input.size(1)}")
                    else:
                        # 后续步骤：只输入最后一个token，复用cache
                        current_decoder_input = decoder_input[:, -1:]
                        if step % 20 == 0:
                            logger.info(f"    Step {step}: Decoder输入长度={current_decoder_input.size(1)} ✅仅1个token | 累积序列长度={decoder_input.size(1)} | KV-cache: {cache_size:.2f}MB")
                        # 调整mask维度
                        dec_mask = dec_mask[:, :, -1:, :]
                        enc_dec_mask = enc_dec_mask[:, :, -1:, :]
```

**关键点：**
- **Q (Query)** 是当前步骤新生成的token产生的，每次只需要计算最后1个token的Q
- **K和V** 来自所有历史token（包括已生成的），需要累积保存

#### 3. **完整的逐步工作流程**

**Step 0（第一个token）：**
```python
decoder_input = [<start>]  # 只有开始符
current_decoder_input = decoder_input  # [1, 1]

# 计算第一个token
Q = WQ(current_decoder_input)  # [1, 1, d_model] 
K = WK(current_decoder_input)  # [1, 1, d_model]
V = WV(current_decoder_input)  # [1, 1, d_model]

# 没有past_key_value，直接计算attention
attn_out = Attention(Q, K, V)

# 保存cache供下一步使用
present_key_value = (K, V)  # 形状: ([1, H, 1, Dh], [1, H, 1, Dh])
```

**Step 1（第二个token）：**
```python
decoder_input = [<start>, token1]  # 累积序列
current_decoder_input = decoder_input[:, -1:]  # [1, 1] 只取最后一个token！

# 只计算新token的Q、K、V
Q_new = WQ(current_decoder_input)  # [1, 1, d_model] 
K_new = WK(current_decoder_input)  # [1, 1, d_model]
V_new = WV(current_decoder_input)  # [1, 1, d_model]

# 【核心】使用cache：拼接历史K和V
past_k, past_v = past_key_value  # 上一步保存的
K = torch.cat([past_k, K_new], dim=2)  # [1, H, 2, Dh]  包含2个token的K
V = torch.cat([past_v, V_new], dim=2)  # [1, H, 2, Dh]  包含2个token的V

# 计算attention（Q只有1个，但K/V有2个）
attn_out = Attention(Q_new, K, V)  # Q: [1,H,1,Dh], K/V: [1,H,2,Dh]

# 更新cache
present_key_value = (K, V)  # 保存完整的K和V
```

**Step N（第N个token）：**
```python
# Q: 只计算1个新token  -> [1, H, 1, Dh]
# K: 累积了N个token    -> [1, H, N, Dh]  
# V: 累积了N个token    -> [1, H, N, Dh]

# Attention计算：
# Q @ K^T = [1, H, 1, Dh] @ [1, H, Dh, N] = [1, H, 1, N]
# score @ V = [1, H, 1, N] @ [1, H, N, Dh] = [1, H, 1, Dh]
```

### 三、性能优化的本质

#### **无Cache（每次都重新计算）：**
```
Step 1: 计算 1个token的KV
Step 2: 重新计算 2个token的KV
Step 3: 重新计算 3个token的KV
...
Step N: 重新计算 N个token的KV

总计算量: 1+2+3+...+N = O(N²)
```

#### **有Cache（复用历史KV）：**
```
Step 1: 计算 1个token的KV，保存
Step 2: 只计算 1个新token的KV，与缓存拼接
Step 3: 只计算 1个新token的KV，与缓存拼接
...
Step N: 只计算 1个新token的KV，与缓存拼接

总计算量: N次独立计算 = O(N)
```

**加速比：`O(N²) -> O(N)`，对于长序列效果显著！**

### 四、核心代码流程图

```
推理循环 (逐token生成)
│
├─ Step 0: decoder_input = [<start>]
│   ├─ Q, K, V = 计算1个token
│   ├─ Attention(Q, K, V)
│   └─ 保存: cache = (K, V)
│
├─ Step 1: decoder_input = [<start>, t1]
│   ├─ 只输入最后一个: [:, -1:] = [t1]
│   ├─ Q_new, K_new, V_new = 计算1个token
│   ├─ K = cat([cache_K, K_new])  ← 核心！
│   ├─ V = cat([cache_V, V_new])  ← 核心！
│   ├─ Attention(Q_new, K完整, V完整)
│   └─ 更新: cache = (K, V)
│
└─ Step N: ...重复...
```

### 五、总结

**KV-Cache的核心就是三个字：`torch.cat`！**

1. **Cache什么**：每个attention layer的Key和Value张量
2. **为什么有效**：避免重复计算历史token的K/V
3. **实现方式**：在序列维度上拼接 `torch.cat([past_kv, new_kv], dim=2)`
4. **内存代价**：随生成长度线性增长 `[B, H, L累积, D]`
5. **计算收益**：从`O(N²)`降到`O(N)`

这就是KV-Cache的本质！它不是什么黑魔法，就是简单的**"保存历史、拼接新的"**策略。
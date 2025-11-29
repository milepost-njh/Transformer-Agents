#!/usr/bin/env python3
"""在服务器上运行，保存 MTP 真实数据"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from inference.inference_ddp import load_tokenizer, create_model, load_checkpoint, create_masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
pt_tokenizer = load_tokenizer("tok_pt/tokenizer.json", 64)
en_tokenizer = load_tokenizer("tok_en/tokenizer.json", 64)

model = create_model(
    input_vocab_size=pt_tokenizer.vocab_size,
    target_vocab_size=en_tokenizer.vocab_size,
    num_layers=8, d_model=512, num_heads=8, dff=2048,
    use_mla=True, use_moe=True, use_mtp=True,
    pt_pad_token_id=pt_tokenizer.pad_token_id,
    en_pad_token_id=en_tokenizer.pad_token_id,
)

checkpoint_path = "checkpoints_ddp_mla_mtp_strong/mid_e12_s2664.pt"
model = load_checkpoint(model, checkpoint_path, device)
model.eval()

# 准备输入
input_text = "Eu gosto de aprender idiomas."
pt_ids = pt_tokenizer.encode(input_text, add_special_tokens=False)
inp_ids = torch.tensor([[pt_tokenizer.bos_token_id] + pt_ids + [pt_tokenizer.eos_token_id]], device=device)
tgt_ids = torch.tensor([[en_tokenizer.bos_token_id]], device=device)

encoder_padding_mask, tgt_mask, enc_dec_mask = create_masks(
    inp_ids, tgt_ids, pt_tokenizer.pad_token_id, en_tokenizer.pad_token_id
)

# 获取主模型输出（MTP 的输入）
print("🚀 运行推理...")
with torch.no_grad():
    # 直接调用 base_transformer 获取 hidden_states
    if hasattr(model, 'base_transformer'):
        transformer = model.base_transformer
        enc_output = transformer.encoder_model(inp_ids, src_mask=encoder_padding_mask)
        enc_out = enc_output[0] if isinstance(enc_output, tuple) else enc_output
        
        dec_output = transformer.decoder_model(tgt_ids, enc_out, tgt_mask=tgt_mask, enc_dec_mask=enc_dec_mask)
        previous_hidden_states = dec_output[0] if isinstance(dec_output, tuple) else dec_output
    else:
        raise ValueError("模型未使用 MTP wrapper")

# 准备 inputs_embeds
inputs_embeds = model.embed_tokens(tgt_ids) * (512 ** 0.5)

# 位置信息
positions = torch.arange(tgt_ids.shape[1], device=device).unsqueeze(0)

# 保存
save_data = {
    'previous_hidden_states': previous_hidden_states.cpu(),
    'inputs_embeds': inputs_embeds.cpu(),
    'positions': positions.cpu(),
}

save_path = 'test/debugs/mtp_real_data.pt'
torch.save(save_data, save_path)

print(f"✅ 已保存到: {save_path}")
print(f"   previous_hidden_states: {previous_hidden_states.shape}")
print(f"   inputs_embeds: {inputs_embeds.shape}")
print(f"\n下一步：scp user@server:{os.path.abspath(save_path)} 到本地")


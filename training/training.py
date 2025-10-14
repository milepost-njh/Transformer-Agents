#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2025/10/10 16:31
@Author  : nijiahui
@FileName: training.py
@Software: PyCharm
 
'''

import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup

from .utils import check_env
from core.models.transformers.transformer_model import Transformer, get_position_embedding, plot_position_embedding
from core.datasets.translate_datasets import load_translation_dataset, train_and_load_tokenizers, test_tokenizers, build_dataloaders, test_dataloaders
from core.checkpointing.utils import save_ckpt, load_ckpt
from core.mask_utils.mask_utils import create_masks
from core.optimizer.optimizer import CustomizedSchedule, plot_customized_lr_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "4"



def loss_function(real, pred):
    """
    Args:
        real: (B, L) target ids (shift 后)
        pred: (B, L, V) logits
    Returns:
        loss (float): 平均有效 token 的交叉熵损失
    """
    B, L, V = pred.shape

    # 展平
    pred = pred.reshape(-1, V)  # (B*L, V)
    real = real.reshape(-1)  # (B*L,)

    # token 级别交叉熵 (padding 已被 ignore_index 屏蔽)
    loss_ = loss_object(pred, real)  # (B*L,)

    # # 统计有效 token
    # valid = (real != PAD_ID_TGT).float()

    # # 均值损失（只对有效 token 求平均）
    # loss = (loss_ * valid).sum() / valid.sum()
    return loss_.mean()



@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    mask = (real != pad_id)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)


class AverageMeter:
    def __init__(self, name="meter"): self.name = name; self.reset()

    def reset(self): self.sum = 0.0; self.n = 0

    def update(self, val, count=1): self.sum += float(val) * count; self.n += count

    @property
    def avg(self): return self.sum / max(1, self.n)


def train_step(batch, transformer, optimizer, scheduler=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.train()

    inp = batch["pt_input_ids"].to(device)
    tar = batch["en_input_ids"].to(device)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    SRC_PAD_ID = pt_tokenizer.pad_token_id
    TGT_PAD_ID = en_tokenizer.pad_token_id

    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
        inp, tar_inp, src_pad_id=SRC_PAD_ID, tgt_pad_id=TGT_PAD_ID
    )
    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

    logits, _ = transformer(
        inp, tar_inp,
        src_mask=enc_pad_mask,
        tgt_mask=dec_mask,
        enc_dec_mask=enc_dec_mask
    )

    loss = loss_function(tar_real, logits)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    acc = token_accuracy(tar_real, logits, pad_id=TGT_PAD_ID)
    return loss.item(), acc


def train_model(
        epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        scheduler=None,
        device: str = None,
        log_every: int = 100,
        ckpt_dir: str = "checkpoints",
        ckpt_prefix: str = "ckpt",
):
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_meter = AverageMeter("train_loss")
    train_acc_meter = AverageMeter("train_accuracy")
    global_step = 0

    for epoch in range(epochs):
        try:
            start = time.time()
            train_loss_meter.reset()
            train_acc_meter.reset()
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                loss_val, acc_val = train_step(
                    batch=batch, transformer=model, optimizer=optimizer, scheduler=scheduler, device=device
                )
                train_loss_meter.update(loss_val, 1)
                train_acc_meter.update(acc_val, 1)

                global_step += 1
                if batch_idx % log_every == 0:
                    print(
                        f"Epoch {epoch + 1} Batch {batch_idx} "
                        f"Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}"
                    )
                    save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step=global_step,
                        ckpt_dir=ckpt_dir,
                        tag="latest"
                    )

            print(f"Epoch {epoch + 1} Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}")
            print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

            # 每个epoch结束后进行验证集评测
            validate_loss, validate_acc = evaluate_on_val(model, val_loader, device)
            print(f"Validation - Epoch {epoch + 1} Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}\n")

        except Exception as e:
            print(f"报错啦!!! 报错信息: {e}")
            save_ckpt(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                tag="error"
            )


@torch.no_grad()
def evaluate(
        inp_sentence: str,
        transformer: Transformer,
        pt_tokenizer,
        en_tokenizer,
        max_length: int,
        device: str = None):
    """
    inp_sentence: 输入的源语言字符串 (pt)
    transformers: 已训练的 Transformer
    pt_tokenizer, en_tokenizer: 分别是葡萄牙语和英语 tokenizer
    max_length: 最大生成长度
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.eval()
    transformer.to(device)

    # 1. 编码输入，加 <s> 和 </s>
    inp_ids = encode_with_bos_eos(pt_tokenizer, inp_sentence)
    encoder_input = torch.tensor(inp_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, Ls)

    # 2. decoder 起始符 <s>
    start_id = en_tokenizer.bos_token_id
    end_id = en_tokenizer.eos_token_id
    decoder_input = torch.tensor([[start_id]], dtype=torch.long, device=device)  # (1, 1)

    # 3. 循环预测
    attention_weights = {}
    for _ in range(max_length):
        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            encoder_input, decoder_input,
            src_pad_id=pt_tokenizer.pad_token_id,
            tgt_pad_id=en_tokenizer.pad_token_id,
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, decoder_input.size(1), -1)

        logits, attn = transformer(
            encoder_input, decoder_input,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask,
        )

        # 取最后一步预测
        next_token_logits = logits[:, -1, :]  # (1, V)
        predicted_id = torch.argmax(next_token_logits, dim=-1)  # (1,)

        if predicted_id.item() == end_id:
            break

        # 拼接到 decoder_input
        decoder_input = torch.cat(
            [decoder_input, predicted_id.unsqueeze(0)], dim=-1
        )  # (1, Lt+1)
        attention_weights = attn

    return decoder_input.squeeze(0).tolist(), attention_weights


@torch.no_grad()
def evaluate_on_val(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0

    for batch in val_loader:
        inp = batch["pt_input_ids"].to(device)
        tar = batch["en_input_ids"].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            inp, tar_inp, src_pad_id=pt_tokenizer.pad_token_id, tgt_pad_id=en_tokenizer.pad_token_id
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

        logits, _ = model(
            inp, tar_inp,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask
        )

        loss = loss_function(tar_real, logits)
        acc = token_accuracy(tar_real, logits, pad_id=en_tokenizer.pad_token_id)

        total_loss += loss.item() * inp.size(0)
        total_acc += acc * inp.size(0)
        total_count += inp.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc


def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name):
    """
    attention: 来自 forward 返回的 attention_weights dict
               形状 [B, num_heads, tgt_len, src_len]
    input_sentence: 源语言字符串
    result: 目标句子 token id 列表 (decoder 输出)
    layer_name: 指定可视化的层 key，比如 "decoder_layer1_att2"
    """
    fig = plt.figure(figsize=(16, 8))

    # 源句子编码
    input_id_sentence = pt_tokenizer.encode(input_sentence, add_special_tokens=False)

    # 取 batch 维度 squeeze，并转 numpy
    attn = attention[layer_name].squeeze(0)  # [num_heads, tgt_len, src_len]
    attn = attn.detach().cpu().numpy()

    for head in range(attn.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # 只取 result[:-1] 的注意力 (去掉最后 <eos>)
        ax.matshow(attn[head][:-1, :], cmap="viridis")

        fontdict = {"fontsize": 10}

        # X 轴: 输入 token (<s> + sentence + </s>)
        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_xticklabels(
            ["<s>"] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ["</s>"],
            fontdict=fontdict, rotation=90,
        )

        # Y 轴: decoder 输出 token
        ax.set_yticks(range(len(result)))
        ax.set_yticklabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size],
            fontdict=fontdict,
        )

        ax.set_ylim(len(result) - 1.5, -0.5)
        ax.set_xlabel(f"Head {head + 1}")

    plt.tight_layout()
    plt.show()


def translate(input_sentence, transformer, pt_tokenizer, en_tokenizer,
              max_length=64, device=None, layer_name=""):
    # 调用我们改好的 evaluate (PyTorch 版)
    result, attention_weights = evaluate(
        inp_sentence=input_sentence,
        transformer=transformer,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        max_length=max_length,
        device=device,
    )

    # 把 token id 转回句子
    predicted_sentence = en_tokenizer.decode(
        [i for i in result if i < en_tokenizer.vocab_size],
        skip_special_tokens=True
    )

    print("Input: {}".format(input_sentence))
    print(f"Predicted translation: {predicted_sentence}")

    # 如果传入了 layer_name，就画注意力图
    if layer_name:
        plot_encoder_decoder_attention(
            attention_weights,
            input_sentence,
            result,
            layer_name
        )

    return predicted_sentence


if __name__ == "__main__":
    # 0. 常量定义

    # 数据文件地址
    train_path = "/home/nijiahui/Datas/por_eng_csv/por_en_train.csv"
    val_path = "/home/nijiahui/Datas/por_eng_csv/por_en_test.csv"
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    checkpoint_dir = './checkpoints-tmp22'

    # 构建词表参数
    vocab_size = 2 ** 13  # 词表大小
    min_freq = 2  # 最小词频
    special_tokens = special_tokens  # 特殊符号
    max_length = 128  # 最大序列长度

    # 模型训练超参数
    batch_size = 32  # 批处理数
    # warmup_steps = 4000           # warmup steps数
    epochs = 60  # 训练轮数
    # learning_rate = 1.0           # 学习率
    # betas = (0.9, 0.98)           # Adam 的一阶矩（梯度均值）；二阶矩（梯度平方的均值）
    # eps = 1e-9                    # 防止除零错误的小常数
    learning_rate = 5e-4
    betas = (0.9, 0.98)
    eps = 1e-8
    weight_decay = 1e-6  # L2正则化(（权重衰减）) - 0.01

    # 模型结构
    # num_layers = 8
    # d_model = 512                 # hidden-size
    # dff = 2048
    # num_heads = 8
    # dropout_rate = 0.1

    num_layers = 4
    d_model = 128  # hidden-size
    dff = 512
    num_heads = 8
    dropout_rate = 0.2

    # 1. 检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本；
    device = check_env()
    print("实际使用设备:", device)

    # 2. 加载葡萄牙语-英语翻译数据集
    train_dataset, val_dataset = load_translation_dataset(
        train_path=train_path,
        val_path=val_path
    )
    print("训练集样本数:", len(train_dataset))
    print("验证集样本数:", len(val_dataset))

    # 3. 构建 Tokenizer
    # 3.1 构建 Tokenizer
    print("开始构建 Tokenizer...")
    pt_tokenizer, en_tokenizer = train_and_load_tokenizers(
        train_dataset=train_dataset,  # 数据集
        pt_key="pt",  # 葡语列名
        en_key="en",  # 英语列名
        vocab_size=vocab_size,  # 词表大小
        min_freq=min_freq,  # 最小词频
        special_tokens=special_tokens,  # 特殊符号
        save_dir_pt="tok_pt",  # 保存目录 (pt)
        save_dir_en="tok_en",  # 保存目录 (en)
        max_length=max_length  # 最大序列长度
    )

    # 3.2 【测试】 Tokenizer 代码
    test_tokenizers(en_tokenizer=en_tokenizer, pt_tokenizer=pt_tokenizer)

    # 3.3 构建 batch data loader
    print("开始构建 batch data loader...")
    train_loader2, val_loader2 = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pt_tokenizer=pt_tokenizer,
        en_tokenizer=en_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0,
        shuffle_train=True
    )

    # 3.4 【测试】 batch data loader
    test_dataloaders(train_loader2, val_loader2)

    # 4. 位置编码
    # 4.2 【测试】 - 打印位置编码矩阵图形
    position_embedding = get_position_embedding(max_length, d_model)
    plot_position_embedding(position_embedding)

    # 5. 构建 model 模型 Transformer 结构
    input_vocab_size = pt_tokenizer.vocab_size
    target_vocab_size = en_tokenizer.vocab_size

    model = Transformer(
        num_layers=num_layers,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        rate=dropout_rate,
        src_padding_idx=pt_tokenizer.pad_token_id if hasattr(pt_tokenizer, "pad_token_id") else None,
        tgt_padding_idx=en_tokenizer.pad_token_id if hasattr(en_tokenizer, "pad_token_id") else None,
    )

    ##############################【Test - optimizer | scheduler 】##############################
    # # 6. 自定义学习率和优化器
    # optimizer = optim.Adam(model.parameters(),
    #                    lr=learning_rate,
    #                    betas=betas,
    #                    eps=eps)
    # # 自定义学习率
    # scheduler = CustomizedSchedule(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # 6. 自定义学习率和优化器
    num_training_steps = len(train_loader2) * epochs

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )
    warmup_steps = int(0.1 * num_training_steps)  # 10% 步数用作 warmup
    scheduler = CustomizedSchedule(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # 自定义学习率
    # num_training_steps = len(train_loader2) * epochs
    # # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    # #     optimizer,
    # #     T_max=num_training_steps,
    # #     eta_min=1e-6
    # # )
    # # 设置 warmup steps
    # warmup_steps = int(0.1 * num_training_steps)  # 10% 步数用作 warmup
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=num_training_steps,
    # )

    # 6.2 【测试】 打印自定义学习率曲线
    plot_customized_lr_curve(optimizer, scheduler, total_steps=num_training_steps,
                             label=f"d_model={d_model}, warmup={warmup_steps}")

    ##############################【Test - optimizer | scheduler 】##############################

    # 7. 自定义损失函数
    # PyTorch 的 CrossEntropyLoss 默认就支持 from_logits=True
    PAD_ID_TGT = en_tokenizer.pad_token_id
    loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_ID_TGT)

    # 8. 训练模型 && checkpoints
    print(f"learning_rate:{learning_rate}")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

        train_model(
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader2,
            val_loader=val_loader2,
            scheduler=scheduler,  # Noam 调度
            device=device,  # 自动选 GPU/CPU
            log_every=100,
            ckpt_dir="checkpoints",
            ckpt_prefix="transformers",
        )
    else:
        start_epoch, global_step = load_ckpt(model, optimizer, scheduler, device=device)
        print("Checkpoint loaded successfully!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练脚本 - 去噪自编码（DAE, Denoising Autoencoder）
参考BART/mBART的预训练方式
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
from datetime import datetime

# 导入你现有的模型和工具
from train_ddp_latest import (
    Transformer, check_env, get_position_embedding,
    create_padding_mask, create_look_ahead_mask,
    train_and_load_tokenizers, load_translation_dataset
)


class NoiseStrategy:
    """噪声添加策略（BART风格）"""
    
    @staticmethod
    def token_masking(token_ids, mask_ratio=0.15, mask_token_id=4):
        """Token掩码：随机将15%的token替换为<mask>"""
        noisy_ids = token_ids.copy()
        num_mask = max(1, int(len(token_ids) * mask_ratio))
        mask_positions = random.sample(range(len(token_ids)), num_mask)
        for pos in mask_positions:
            noisy_ids[pos] = mask_token_id
        return noisy_ids
    
    @staticmethod
    def token_deletion(token_ids, delete_ratio=0.1):
        """Token删除：随机删除10%的token"""
        if len(token_ids) <= 2:  # 保留至少BOS和EOS
            return token_ids
        keep_mask = [random.random() > delete_ratio for _ in token_ids]
        # 确保BOS和EOS保留
        keep_mask[0] = True
        keep_mask[-1] = True
        noisy_ids = [tid for tid, keep in zip(token_ids, keep_mask) if keep]
        return noisy_ids
    
    @staticmethod
    def text_infilling(token_ids, span_length=3, span_ratio=0.3, mask_token_id=4):
        """文本填充：随机span被替换为单个mask token"""
        if len(token_ids) <= span_length + 2:
            return token_ids
        
        noisy_ids = []
        i = 0
        while i < len(token_ids):
            if i == 0 or i == len(token_ids) - 1:
                # 保留BOS和EOS
                noisy_ids.append(token_ids[i])
                i += 1
            elif random.random() < span_ratio:
                # 创建一个span并替换为mask
                span_len = random.randint(1, min(span_length, len(token_ids) - i - 1))
                noisy_ids.append(mask_token_id)
                i += span_len
            else:
                noisy_ids.append(token_ids[i])
                i += 1
        
        return noisy_ids
    
    @staticmethod
    def sentence_permutation(token_ids, max_shuffle_distance=3):
        """句子重排：打乱句子顺序（适用于长文本）"""
        # 简化版：对于单句，返回原序列
        return token_ids
    
    @staticmethod
    def apply_noise(token_ids, noise_type="mixed", mask_token_id=4):
        """应用噪声"""
        if noise_type == "mask":
            return NoiseStrategy.token_masking(token_ids, mask_token_id=mask_token_id)
        elif noise_type == "delete":
            return NoiseStrategy.token_deletion(token_ids)
        elif noise_type == "infill":
            return NoiseStrategy.text_infilling(token_ids, mask_token_id=mask_token_id)
        elif noise_type == "mixed":
            # 随机选择一种噪声类型
            noise_type = random.choice(["mask", "delete", "infill"])
            return NoiseStrategy.apply_noise(token_ids, noise_type, mask_token_id)
        else:
            return token_ids


class PretrainDataset(Dataset):
    """预训练数据集"""
    
    def __init__(self, data_file, tokenizer, max_length=64, noise_type="mixed"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_type = noise_type
        self.mask_token_id = tokenizer.mask_token_id
        
        # 加载数据
        logger.info(f"加载数据: {data_file}")
        self.sentences = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.sentences.append(line)
        
        logger.info(f"加载了 {len(self.sentences)} 个句子")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        
        # 编码
        token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        
        # 添加BOS和EOS
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        token_ids = [bos_id] + token_ids[:self.max_length-2] + [eos_id]
        
        # 原始序列（目标）
        clean_ids = token_ids.copy()
        
        # 添加噪声（输入）
        noisy_ids = NoiseStrategy.apply_noise(
            token_ids, 
            noise_type=self.noise_type,
            mask_token_id=self.mask_token_id
        )
        
        return {
            "input_ids": noisy_ids,    # 带噪声的输入
            "target_ids": clean_ids,   # 干净的目标
        }


def collate_fn(batch, pad_token_id=1):
    """动态padding"""
    def pad_sequence(seqs, pad_value):
        max_len = max(len(s) for s in seqs)
        padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            L = len(s)
            padded[i, :L] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, :L] = 1
        return padded, attn_mask
    
    input_ids_list = [ex["input_ids"] for ex in batch]
    target_ids_list = [ex["target_ids"] for ex in batch]
    
    input_ids, input_mask = pad_sequence(input_ids_list, pad_token_id)
    target_ids, target_mask = pad_sequence(target_ids_list, pad_token_id)
    
    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "target_ids": target_ids,
        "target_mask": target_mask,
    }


def pretrain_step(batch, model, optimizer, device, pad_token_id):
    """预训练单步"""
    model.train()
    
    # 将数据移到GPU
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    
    # 创建decoder输入（target去掉最后一个token）
    dec_input = target_ids[:, :-1]
    dec_target = target_ids[:, 1:]
    
    # 创建mask
    from train_ddp_latest import create_masks
    enc_mask, dec_mask, enc_dec_mask = create_masks(
        input_ids, dec_input,
        src_pad_id=pad_token_id,
        tgt_pad_id=pad_token_id
    )
    enc_dec_mask = enc_dec_mask.expand(-1, 1, dec_input.size(1), -1)
    
    # 前向传播
    logits, _, hidden_states = model(
        input_ids, dec_input,
        src_mask=enc_mask,
        tgt_mask=dec_mask,
        enc_dec_mask=enc_dec_mask
    )
    
    # 计算损失
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def pretrain(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs,
    device,
    pad_token_id,
    checkpoint_dir,
    log_every=100,
):
    """预训练主循环"""
    logger.info("开始预训练...")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 训练
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            loss = pretrain_step(batch, model, optimizer, device, pad_token_id)
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            
            # 日志
            if global_step % log_every == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
            
            # 学习率调度
            if scheduler:
                scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")
        
        # 验证
        if val_loader:
            val_loss = validate(model, val_loader, device, pad_token_id)
            logger.info(f"Epoch {epoch+1} 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    checkpoint_dir, tag="best"
                )
                logger.info(f"✅ 保存最佳模型 (val_loss={val_loss:.4f})")
        
        # 每个epoch保存一次
        save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            checkpoint_dir, tag="latest"
        )
    
    logger.info("预训练完成!")


def validate(model, val_loader, device, pad_token_id):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            dec_input = target_ids[:, :-1]
            dec_target = target_ids[:, 1:]
            
            from train_ddp_latest import create_masks
            enc_mask, dec_mask, enc_dec_mask = create_masks(
                input_ids, dec_input,
                src_pad_id=pad_token_id,
                tgt_pad_id=pad_token_id
            )
            enc_dec_mask = enc_dec_mask.expand(-1, 1, dec_input.size(1), -1)
            
            logits, _, hidden_states = model(
                input_ids, dec_input,
                src_mask=enc_mask,
                tgt_mask=dec_mask,
                enc_dec_mask=enc_dec_mask
            )
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, step, checkpoint_dir, tag="latest"):
    """保存checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    
    path = checkpoint_dir / f"pretrain_{tag}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="预训练脚本")
    parser.add_argument("--train-data", required=True, help="训练数据文件")
    parser.add_argument("--val-data", default=None, help="验证数据文件")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--max-length", type=int, default=64, help="最大序列长度")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--noise-type", default="mixed", 
                       choices=["mask", "delete", "infill", "mixed"],
                       help="噪声类型")
    parser.add_argument("--checkpoint-dir", default="../checkpoints/pretrain",
                       help="Checkpoint目录")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("预训练脚本 - 去噪自编码")
    logger.info("=" * 60)
    
    # 设备
    device = check_env()
    
    # 加载tokenizer（使用已训练好的）
    logger.info("加载tokenizer...")
    # 使用你现有的tokenizer加载逻辑
    # 这里简化处理，实际应该从保存的tokenizer加载
    from transformers import PreTrainedTokenizerFast
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tok_en/tokenizer.json")
    en_tokenizer.pad_token = "<pad>"
    en_tokenizer.mask_token = "<mask>"
    en_tokenizer.bos_token = "<s>"
    en_tokenizer.eos_token = "</s>"
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = PretrainDataset(
        args.train_data,
        en_tokenizer,
        max_length=args.max_length,
        noise_type=args.noise_type
    )
    
    val_dataset = None
    if args.val_data:
        val_dataset = PretrainDataset(
            args.val_data,
            en_tokenizer,
            max_length=args.max_length,
            noise_type=args.noise_type
        )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, en_tokenizer.pad_token_id),
        num_workers=4
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, en_tokenizer.pad_token_id),
            num_workers=4
        )
    
    # 创建模型（使用你现有的Transformer）
    logger.info("创建模型...")
    model = Transformer(
        num_layers=6,
        input_vocab_size=len(en_tokenizer),
        target_vocab_size=len(en_tokenizer),
        max_length=args.max_length,
        d_model=512,
        num_heads=8,
        dff=2048,
        rate=0.1,
        use_rope=True,
        use_moe=False,
    )
    model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 学习率调度
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 开始预训练
    pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        pad_token_id=en_tokenizer.pad_token_id,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

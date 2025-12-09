#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据下载和处理脚本
支持从OPUS下载pt-en平行语料并进行清洗
"""

import os
import re
import gzip
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import unicodedata
from loguru import logger

# 配置日志
logger.add("data_preparation.log", rotation="100 MB")


class DatasetDownloader:
    """OPUS数据集下载器"""
    
    # OPUS数据集URL模板
    OPUS_BASE = "https://object.pouta.csc.fi/OPUS-"
    
    DATASETS = {
        "Europarl": {
            "url": "https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-pt.txt.zip",
            "expected_pairs": 2_000_000,
            "quality": "high",
        },
        "OpenSubtitles": {
            "url": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-pt.txt.zip",
            "expected_pairs": 10_000_000,
            "quality": "medium",
        },
        "TED2020": {
            "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-pt.txt.zip",
            "expected_pairs": 200_000,
            "quality": "high",
        },
        "News-Commentary": {
            "url": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-pt.txt.zip",
            "expected_pairs": 50_000,
            "quality": "high",
        },
        "GlobalVoices": {
            "url": "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-pt.txt.zip",
            "expected_pairs": 100_000,
            "quality": "medium",
        },
    }
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, filename):
        """下载文件并显示进度条"""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"文件已存在: {filepath}")
            return filepath
        
        logger.info(f"开始下载: {url}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"下载完成: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return None
    
    def download_dataset(self, dataset_name):
        """下载指定数据集"""
        if dataset_name not in self.DATASETS:
            logger.error(f"未知数据集: {dataset_name}")
            return None
        
        info = self.DATASETS[dataset_name]
        filename = f"{dataset_name}.zip"
        
        return self.download_file(info["url"], filename)


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, min_length=5, max_length=100, max_length_ratio=2.0):
        self.min_length = min_length
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio
        
        # 编译正则表达式（提高性能）
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.multiple_spaces = re.compile(r'\s+')
        self.special_chars = re.compile(r'[^\w\s\.\,\!\?\'\"\-\:\;\(\)]', re.UNICODE)
    
    def clean_text(self, text):
        """清洗单个文本"""
        if not text or not isinstance(text, str):
            return None
        
        # 移除URL
        text = self.url_pattern.sub('', text)
        
        # 移除邮箱
        text = self.email_pattern.sub('', text)
        
        # 移除HTML标签
        text = self.html_pattern.sub('', text)
        
        # 标准化Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # 移除多余空格
        text = self.multiple_spaces.sub(' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text if text else None
    
    def is_valid_pair(self, src_text, tgt_text):
        """检查句对是否有效"""
        if not src_text or not tgt_text:
            return False
        
        # 清洗文本
        src_clean = self.clean_text(src_text)
        tgt_clean = self.clean_text(tgt_text)
        
        if not src_clean or not tgt_clean:
            return False
        
        # 分词（简单按空格）
        src_tokens = src_clean.split()
        tgt_tokens = tgt_clean.split()
        
        src_len = len(src_tokens)
        tgt_len = len(tgt_tokens)
        
        # 长度检查
        if src_len < self.min_length or tgt_len < self.min_length:
            return False
        
        if src_len > self.max_length or tgt_len > self.max_length:
            return False
        
        # 长度比例检查
        length_ratio = max(src_len, tgt_len) / max(min(src_len, tgt_len), 1)
        if length_ratio > self.max_length_ratio:
            return False
        
        # 检查是否大部分是标点符号
        src_alpha_ratio = sum(c.isalnum() for c in src_clean) / max(len(src_clean), 1)
        tgt_alpha_ratio = sum(c.isalnum() for c in tgt_clean) / max(len(tgt_clean), 1)
        
        if src_alpha_ratio < 0.5 or tgt_alpha_ratio < 0.5:
            return False
        
        # 检查是否包含过多数字
        src_digit_ratio = sum(c.isdigit() for c in src_clean) / max(len(src_clean), 1)
        tgt_digit_ratio = sum(c.isdigit() for c in tgt_clean) / max(len(tgt_clean), 1)
        
        if src_digit_ratio > 0.5 or tgt_digit_ratio > 0.5:
            return False
        
        return True
    
    def clean_pair(self, src_text, tgt_text):
        """清洗句对"""
        if not self.is_valid_pair(src_text, tgt_text):
            return None, None
        
        src_clean = self.clean_text(src_text)
        tgt_clean = self.clean_text(tgt_text)
        
        return src_clean, tgt_clean


class ParallelCorpusProcessor:
    """平行语料处理器"""
    
    def __init__(self, output_dir="data/processed", max_pairs_per_dataset=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pairs_per_dataset = max_pairs_per_dataset
        self.cleaner = DataCleaner()
    
    def process_moses_format(self, src_file, tgt_file, dataset_name):
        """处理Moses格式的平行语料"""
        logger.info(f"处理数据集: {dataset_name}")
        logger.info(f"源文件: {src_file}")
        logger.info(f"目标文件: {tgt_file}")
        
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            logger.error(f"文件不存在")
            return None
        
        output_file = self.output_dir / f"{dataset_name}_cleaned.tsv"
        
        total_pairs = 0
        valid_pairs = 0
        seen_pairs = set()  # 用于去重
        
        # 读取文件
        with open(src_file, 'r', encoding='utf-8', errors='ignore') as f_src, \
             open(tgt_file, 'r', encoding='utf-8', errors='ignore') as f_tgt, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for src_line, tgt_line in tqdm(zip(f_src, f_tgt), desc=f"处理 {dataset_name}"):
                total_pairs += 1
                
                # 应用最大句对限制
                if self.max_pairs_per_dataset and valid_pairs >= self.max_pairs_per_dataset:
                    break
                
                # 清洗
                src_clean, tgt_clean = self.cleaner.clean_pair(
                    src_line.strip(), 
                    tgt_line.strip()
                )
                
                if src_clean and tgt_clean:
                    # 去重
                    pair_hash = hash((src_clean, tgt_clean))
                    if pair_hash not in seen_pairs:
                        seen_pairs.add(pair_hash)
                        f_out.write(f"{src_clean}\t{tgt_clean}\n")
                        valid_pairs += 1
        
        logger.info(f"数据集 {dataset_name}: 总句对 {total_pairs}, 有效句对 {valid_pairs} ({valid_pairs/max(total_pairs, 1)*100:.2f}%)")
        
        return output_file, valid_pairs
    
    def merge_datasets(self, dataset_files, output_name="merged_corpus"):
        """合并多个数据集"""
        logger.info(f"合并数据集到: {output_name}")
        
        output_train = self.output_dir / f"{output_name}_train.csv"
        output_val = self.output_dir / f"{output_name}_val.csv"
        
        all_pairs = []
        
        # 读取所有数据
        for dataset_file, pair_count in dataset_files:
            if dataset_file and os.path.exists(dataset_file):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '\t' in line:
                            all_pairs.append(line.strip())
        
        logger.info(f"总句对数: {len(all_pairs)}")
        
        # 打乱
        import random
        random.shuffle(all_pairs)
        
        # 划分训练集和验证集 (95% / 5%)
        split_idx = int(len(all_pairs) * 0.95)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        # 保存训练集
        with open(output_train, 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                f.write(f"{pair}\n")
        
        # 保存验证集
        with open(output_val, 'w', encoding='utf-8') as f:
            for pair in val_pairs:
                f.write(f"{pair}\n")
        
        logger.info(f"训练集: {len(train_pairs)} 句对 -> {output_train}")
        logger.info(f"验证集: {len(val_pairs)} 句对 -> {output_val}")
        
        return output_train, output_val


def unzip_dataset(zip_file, extract_dir):
    """解压数据集"""
    import zipfile
    
    logger.info(f"解压: {zip_file}")
    extract_path = Path(extract_dir) / zip_file.stem
    extract_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"解压完成: {extract_path}")
        return extract_path
    except Exception as e:
        logger.error(f"解压失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载和处理预训练数据")
    parser.add_argument("--datasets", nargs="+", 
                       default=["Europarl", "TED2020", "News-Commentary"],
                       help="要下载的数据集列表")
    parser.add_argument("--max-pairs", type=int, default=None,
                       help="每个数据集的最大句对数")
    parser.add_argument("--output-dir", default="data",
                       help="输出目录")
    parser.add_argument("--skip-download", action="store_true",
                       help="跳过下载步骤")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("预训练数据准备脚本")
    logger.info("=" * 60)
    
    # 1. 下载数据
    raw_dir = Path(args.output_dir) / "raw"
    processed_dir = Path(args.output_dir) / "processed"
    
    downloader = DatasetDownloader(output_dir=raw_dir)
    processor = ParallelCorpusProcessor(
        output_dir=processed_dir,
        max_pairs_per_dataset=args.max_pairs
    )
    
    dataset_files = []
    
    if not args.skip_download:
        for dataset_name in args.datasets:
            logger.info(f"\n处理数据集: {dataset_name}")
            
            # 下载
            zip_file = downloader.download_dataset(dataset_name)
            if not zip_file:
                logger.warning(f"跳过数据集: {dataset_name}")
                continue
            
            # 解压
            extract_dir = unzip_dataset(zip_file, raw_dir)
            if not extract_dir:
                continue
            
            # 查找.en和.pt文件
            en_files = list(extract_dir.glob("*.en"))
            pt_files = list(extract_dir.glob("*.pt"))
            
            if not en_files or not pt_files:
                logger.warning(f"未找到语料文件: {extract_dir}")
                continue
            
            # 处理
            result = processor.process_moses_format(
                pt_files[0],  # 葡萄牙语作为源语言
                en_files[0],  # 英语作为目标语言
                dataset_name
            )
            
            if result:
                dataset_files.append(result)
    else:
        # 使用已有文件
        for dataset_name in args.datasets:
            cleaned_file = processed_dir / f"{dataset_name}_cleaned.tsv"
            if cleaned_file.exists():
                # 统计行数
                with open(cleaned_file, 'r') as f:
                    pair_count = sum(1 for _ in f)
                dataset_files.append((cleaned_file, pair_count))
    
    # 2. 合并数据集
    if dataset_files:
        logger.info("\n" + "=" * 60)
        logger.info("合并所有数据集")
        logger.info("=" * 60)
        
        train_file, val_file = processor.merge_datasets(
            dataset_files,
            output_name="pt_en_pretrain"
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("数据准备完成!")
        logger.info(f"训练集: {train_file}")
        logger.info(f"验证集: {val_file}")
        logger.info("=" * 60)
    else:
        logger.error("没有可用的数据集")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据下载脚本
支持下载Wikipedia等单语数据用于预训练
"""

import os
import gzip
import requests
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import subprocess

logger.add("../data/download_pretrain.log", rotation="100 MB")


class WikipediaDownloader:
    """Wikipedia数据下载器"""
    
    # Wikipedia dumps镜像地址
    WIKI_DUMPS = {
        "en": {
            "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
            "size": "20GB (compressed)",
            "extracted": "80GB",
        },
        "pt": {
            "url": "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2",
            "size": "2GB (compressed)",
            "extracted": "8GB",
        },
    }
    
    # 使用已处理的纯文本版本（更小，更快）
    WIKI_TEXT_DUMPS = {
        "en": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2",
        "pt": "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles-multistream.xml.bz2",
    }
    
    def __init__(self, output_dir="data/pretrain"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, filename):
        """下载文件并显示进度条"""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            logger.info(f"文件已存在: {filepath}")
            return filepath
        
        logger.info(f"开始下载: {url}")
        logger.warning(f"Wikipedia完整数据很大，建议使用sample数据进行测试")
        
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


class SimpleTextExtractor:
    """简单的文本提取器（用于快速测试）"""
    
    @staticmethod
    def create_sample_data(output_dir, lang="en", num_sentences=100000):
        """创建示例数据用于快速测试"""
        output_file = Path(output_dir) / f"{lang}_wiki_sample.txt"
        
        if output_file.exists():
            logger.info(f"示例数据已存在: {output_file}")
            return output_file
        
        logger.info(f"创建{lang}示例数据: {num_sentences}句")
        
        # 生成示例句子（实际应该从Wikipedia下载）
        sample_sentences = {
            "en": [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand human language.",
                "Deep learning models have revolutionized computer vision.",
                "Transformers are the dominant architecture in modern NLP.",
            ],
            "pt": [
                "O rato roeu a roupa do rei de Roma.",
                "A inteligência artificial está transformando o mundo.",
                "O processamento de linguagem natural permite que computadores entendam humanos.",
                "Modelos de aprendizado profundo revolucionaram a visão computacional.",
                "Transformers são a arquitetura dominante em PLN moderna.",
            ],
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_sentences):
                sentence = sample_sentences[lang][i % len(sample_sentences[lang])]
                f.write(sentence + "\n")
        
        logger.info(f"示例数据已创建: {output_file}")
        return output_file


def download_from_huggingface(lang="en", output_dir="data/pretrain", max_samples=1000000):
    """
    从HuggingFace下载Wikipedia数据（推荐方式）
    使用datasets库，更快更方便
    """
    try:
        from datasets import load_dataset
        
        logger.info(f"从HuggingFace下载{lang} Wikipedia数据...")
        
        # 加载Wikipedia数据集
        if lang == "en":
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        elif lang == "pt":
            dataset = load_dataset("wikipedia", "20220301.pt", split="train", streaming=True)
        else:
            logger.error(f"不支持的语言: {lang}")
            return None
        
        output_file = Path(output_dir) / f"{lang}_wiki.txt"
        
        logger.info(f"提取文本到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(tqdm(dataset, total=max_samples, desc=f"处理{lang}数据")):
                if i >= max_samples:
                    break
                
                # 提取文本内容
                text = example.get('text', '')
                if text.strip():
                    # 简单清洗：移除多余空行
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    f.write('\n'.join(lines) + '\n\n')
        
        logger.info(f"✅ {lang} Wikipedia数据下载完成: {output_file}")
        return output_file
    
    except ImportError:
        logger.error("需要安装datasets库: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载预训练数据")
    parser.add_argument("--langs", nargs="+", default=["en", "pt"],
                       help="要下载的语言列表")
    parser.add_argument("--output-dir", default="../data/pretrain",
                       help="输出目录")
    parser.add_argument("--max-samples", type=int, default=500000,
                       help="每个语言的最大样本数")
    parser.add_argument("--method", choices=["huggingface", "sample"], 
                       default="huggingface",
                       help="下载方式: huggingface(推荐) 或 sample(快速测试)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("预训练数据下载脚本")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.method == "huggingface":
        # 推荐方式：从HuggingFace下载
        for lang in args.langs:
            logger.info(f"\n下载 {lang} Wikipedia数据...")
            result = download_from_huggingface(
                lang=lang, 
                output_dir=output_dir,
                max_samples=args.max_samples
            )
            
            if result:
                logger.info(f"✅ {lang} 数据已保存: {result}")
    
    elif args.method == "sample":
        # 快速测试：生成示例数据
        extractor = SimpleTextExtractor()
        for lang in args.langs:
            logger.info(f"\n创建 {lang} 示例数据...")
            result = extractor.create_sample_data(
                output_dir=output_dir,
                lang=lang,
                num_sentences=args.max_samples // 10  # 示例数据更少
            )
            logger.info(f"✅ {lang} 示例数据已创建: {result}")
    
    logger.info("\n" + "=" * 60)
    logger.info("数据下载完成!")
    logger.info(f"数据目录: {output_dir}")
    logger.info("=" * 60)
    
    # 显示数据统计
    logger.info("\n数据统计:")
    for lang in args.langs:
        file_path = output_dir / f"{lang}_wiki.txt"
        if not file_path.exists():
            file_path = output_dir / f"{lang}_wiki_sample.txt"
        
        if file_path.exists():
            # 统计行数
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            
            # 文件大小
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"  {lang}: {lines:,} 行, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
模型评估脚本
支持BLEU、ROUGE、METEOR等评估指标
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
import json
from datetime import datetime
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import InferenceEngine, InferenceConfig, create_moe_config, create_mtp_config


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = InferenceEngine(config)
        
    def initialize(self):
        """初始化评估器"""
        self.engine.initialize()
    
    def load_test_data(self, test_file: str) -> List[Tuple[str, str]]:
        """加载测试数据"""
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    src, tgt = line.split('\t', 1)
                    test_data.append((src.strip(), tgt.strip()))
                elif '|||' in line:
                    src, tgt = line.split('|||', 1)
                    test_data.append((src.strip(), tgt.strip()))
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        
        logger.info(f"✅ Loaded {len(test_data)} test samples")
        return test_data
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """计算BLEU分数"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            
            smoothie = SmoothingFunction().method4
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                
                # 计算BLEU-4分数
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
            
            return np.mean(bleu_scores)
        
        except ImportError:
            logger.warning("NLTK not available, using simple BLEU approximation")
            return self._simple_bleu(predictions, references)
    
    def _simple_bleu(self, predictions: List[str], references: List[str]) -> float:
        """简单的BLEU近似计算"""
        total_score = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # 计算1-gram精确度
            if len(pred_words) == 0:
                score = 0.0
            else:
                matches = sum(1 for word in pred_words if word in ref_words)
                score = matches / len(pred_words)
            
            total_score += score
        
        return total_score / len(predictions)
    
    def compute_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)
            return {
                'rouge-1': scores['rouge-1']['f'],
                'rouge-2': scores['rouge-2']['f'],
                'rouge-l': scores['rouge-l']['f']
            }
        except ImportError:
            logger.warning("ROUGE not available, using simple ROUGE approximation")
            return self._simple_rouge(predictions, references)
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """简单的ROUGE近似计算"""
        rouge_1_scores = []
        rouge_2_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # ROUGE-1 (1-gram overlap)
            if len(pred_words) == 0 or len(ref_words) == 0:
                rouge_1 = 0.0
            else:
                pred_set = set(pred_words)
                ref_set = set(ref_words)
                overlap = len(pred_set & ref_set)
                rouge_1 = overlap / len(ref_set)
            rouge_1_scores.append(rouge_1)
            
            # ROUGE-2 (2-gram overlap)
            if len(pred_words) < 2 or len(ref_words) < 2:
                rouge_2 = 0.0
            else:
                pred_bigrams = set(zip(pred_words[:-1], pred_words[1:]))
                ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
                overlap = len(pred_bigrams & ref_bigrams)
                rouge_2 = overlap / len(ref_bigrams)
            rouge_2_scores.append(rouge_2)
        
        return {
            'rouge-1': np.mean(rouge_1_scores),
            'rouge-2': np.mean(rouge_2_scores),
            'rouge-l': np.mean(rouge_1_scores)  # 简化版本
        }
    
    def compute_meteor_score(self, predictions: List[str], references: List[str]) -> float:
        """计算METEOR分数"""
        try:
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            
            meteor_scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                meteor = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(meteor)
            
            return np.mean(meteor_scores)
        
        except ImportError:
            logger.warning("METEOR not available, using simple METEOR approximation")
            return self._simple_meteor(predictions, references)
    
    def _simple_meteor(self, predictions: List[str], references: List[str]) -> float:
        """简单的METEOR近似计算"""
        total_score = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if len(pred_words) == 0 or len(ref_words) == 0:
                score = 0.0
            else:
                # 计算精确度和召回率的调和平均
                pred_set = set(pred_words)
                ref_set = set(ref_words)
                overlap = len(pred_set & ref_set)
                
                precision = overlap / len(pred_words)
                recall = overlap / len(ref_words)
                
                if precision + recall == 0:
                    score = 0.0
                else:
                    score = 2 * precision * recall / (precision + recall)
            
            total_score += score
        
        return total_score / len(predictions)
    
    def compute_ter_score(self, predictions: List[str], references: List[str]) -> float:
        """计算TER (Translation Error Rate) 分数"""
        try:
            from ter import ter
            ter_scores = []
            for pred, ref in zip(predictions, references):
                ter_score = ter(pred, ref)
                ter_scores.append(ter_score)
            return np.mean(ter_scores)
        except ImportError:
            logger.warning("TER not available, using simple TER approximation")
            return self._simple_ter(predictions, references)
    
    def _simple_ter(self, predictions: List[str], references: List[str]) -> float:
        """简单的TER近似计算"""
        total_score = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # 计算编辑距离
            m, n = len(pred_words), len(ref_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred_words[i-1] == ref_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            # TER = 编辑距离 / 参考长度
            ter_score = dp[m][n] / max(n, 1)
            total_score += ter_score
        
        return total_score / len(predictions)
    
    def evaluate(self, test_data: List[Tuple[str, str]], decode_method: str = "greedy") -> Dict:
        """评估模型"""
        logger.info(f"🚀 Starting evaluation with {len(test_data)} samples...")
        
        # 生成预测
        predictions = []
        references = []
        inference_times = []
        
        for i, (src, tgt) in enumerate(test_data):
            start_time = time.time()
            result = self.engine.infer_single(src, decode_method)
            inference_time = time.time() - start_time
            
            predictions.append(result['output'])
            references.append(tgt)
            inference_times.append(inference_time)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(test_data)} samples")
        
        # 计算评估指标
        logger.info("📊 Computing evaluation metrics...")
        
        metrics = {
            'bleu': self.compute_bleu_score(predictions, references),
            'rouge': self.compute_rouge_score(predictions, references),
            'meteor': self.compute_meteor_score(predictions, references),
            'ter': self.compute_ter_score(predictions, references),
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times),
            'num_samples': len(test_data)
        }
        
        return metrics
    
    def compare_modes(self, test_data: List[Tuple[str, str]], modes: List[str]) -> Dict:
        """比较不同模式的性能"""
        comparison_results = {}
        
        for mode in modes:
            logger.info(f"🔄 Evaluating mode: {mode}")
            
            # 更新配置
            original_mode = self.config.mode
            original_use_mla = self.config.use_mla
            original_use_mtp = self.config.use_mtp
            
            if mode == "normal":
                self.config.use_mla = False
                self.config.use_mtp = False
            elif mode == "mla":
                self.config.use_mla = True
                self.config.use_mtp = False
            elif mode == "mtp":
                self.config.use_mla = False
                self.config.use_mtp = True
            
            # 重新初始化模型（如果需要）
            if mode != original_mode:
                self.engine.loader.create_model()
                self.engine.loader.load_checkpoint()
                self.engine.loader.setup_model()
            
            # 评估
            metrics = self.evaluate(test_data)
            comparison_results[mode] = metrics
            
            # 恢复原始配置
            self.config.mode = original_mode
            self.config.use_mla = original_use_mla
            self.config.use_mtp = original_use_mtp
        
        return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, required=True,
                       help="Path to test file (src\\ttgt format)")
    parser.add_argument("--mode", type=str, default="normal",
                       choices=["normal", "mla", "mtp", "all"],
                       help="Evaluation mode")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    parser.add_argument("--max_length", type=int, default=64,
                       help="Maximum generation length")
    parser.add_argument("--decode_method", type=str, default="greedy",
                       choices=["greedy", "sample", "beam"],
                       help="Decoding method")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # 创建配置
    config = InferenceConfig(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
    )
    
    # 根据模式设置配置
    if args.mode in ["mla", "all"]:
        config.use_mla = True
    if args.mode in ["mtp", "all"]:
        config.use_mtp = True
        config.moe_config = create_moe_config(config)
        config.mtp_config = create_mtp_config(config)
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    try:
        # 初始化
        evaluator.initialize()
        
        # 加载测试数据
        test_data = evaluator.load_test_data(args.test_file)
        
        # 评估
        if args.mode == "all":
            # 比较所有模式
            modes = ["normal", "mla", "mtp"]
            results = evaluator.compare_modes(test_data, modes)
            
            # 输出比较结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                print("\n" + "="*60)
                print("EVALUATION RESULTS COMPARISON")
                print("="*60)
                for mode, metrics in results.items():
                    print(f"\nMode: {mode.upper()}")
                    print(f"BLEU: {metrics['bleu']:.4f}")
                    print(f"ROUGE-1: {metrics['rouge']['rouge-1']:.4f}")
                    print(f"ROUGE-2: {metrics['rouge']['rouge-2']:.4f}")
                    print(f"ROUGE-L: {metrics['rouge']['rouge-l']:.4f}")
                    print(f"METEOR: {metrics['meteor']:.4f}")
                    print(f"TER: {metrics['ter']:.4f}")
                    print(f"Avg Inference Time: {metrics['avg_inference_time']:.3f}s")
        else:
            # 单模式评估
            results = evaluator.evaluate(test_data, args.decode_method)
            
            # 输出结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                print("\n" + "="*60)
                print("EVALUATION RESULTS")
                print("="*60)
                print(f"Mode: {args.mode.upper()}")
                print(f"BLEU: {results['bleu']:.4f}")
                print(f"ROUGE-1: {results['rouge']['rouge-1']:.4f}")
                print(f"ROUGE-2: {results['rouge']['rouge-2']:.4f}")
                print(f"ROUGE-L: {results['rouge']['rouge-l']:.4f}")
                print(f"METEOR: {results['meteor']:.4f}")
                print(f"TER: {results['ter']:.4f}")
                print(f"Avg Inference Time: {results['avg_inference_time']:.3f}s")
                print(f"Total Inference Time: {results['total_inference_time']:.3f}s")
                print(f"Number of Samples: {results['num_samples']}")
    
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

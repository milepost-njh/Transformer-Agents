# -*- coding: utf-8 -*-
"""
推理配置文件
包含不同模式的预设配置
"""

from typing import Dict, Any
from inference.inference import InferenceConfig, create_moe_config, create_mtp_config


# 预设配置
PRESET_CONFIGS = {
    "small": {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "dff": 1024,
        "max_length": 32,
        "vocab_size": 8192,
    },
    "medium": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "dff": 2048,
        "max_length": 64,
        "vocab_size": 8192,
    },
    "large": {
        "d_model": 1024,
        "num_heads": 16,
        "num_layers": 12,
        "dff": 4096,
        "max_length": 128,
        "vocab_size": 16384,
    },
    "xlarge": {
        "d_model": 2048,
        "num_heads": 32,
        "num_layers": 24,
        "dff": 8192,
        "max_length": 256,
        "vocab_size": 32768,
    }
}

# 推理模式配置
MODE_CONFIGS = {
    "normal": {
        "use_mla": False,
        "use_mtp": False,
        "use_moe": False,
    },
    "mla": {
        "use_mla": True,
        "use_mtp": False,
        "use_moe": False,
    },
    "mtp": {
        "use_mla": False,
        "use_mtp": True,
        "use_moe": True,  # MTP通常与MoE一起使用
    },
    "mla_mtp": {
        "use_mla": True,
        "use_mtp": True,
        "use_moe": True,
    }
}

# 解码策略配置
DECODE_CONFIGS = {
    "greedy": {
        "decode_method": "greedy",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "do_sample": False,
    },
    "sampling": {
        "decode_method": "sample",
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
        "num_beams": 1,
        "do_sample": True,
    },
    "beam_search": {
        "decode_method": "beam",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "num_beams": 4,
        "do_sample": False,
    },
    "nucleus_sampling": {
        "decode_method": "sample",
        "temperature": 0.8,
        "top_k": 0,
        "top_p": 0.9,
        "num_beams": 1,
        "do_sample": True,
    }
}


def get_preset_config(preset_name: str, **overrides) -> Dict[str, Any]:
    """获取预设配置"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset_name].copy()
    config.update(overrides)
    return config


def get_mode_config(mode_name: str) -> Dict[str, Any]:
    """获取模式配置"""
    if mode_name not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode_name}. Available: {list(MODE_CONFIGS.keys())}")
    
    return MODE_CONFIGS[mode_name].copy()


def get_decode_config(decode_name: str) -> Dict[str, Any]:
    """获取解码配置"""
    if decode_name not in DECODE_CONFIGS:
        raise ValueError(f"Unknown decode strategy: {decode_name}. Available: {list(DECODE_CONFIGS.keys())}")
    
    return DECODE_CONFIGS[decode_name].copy()


def create_inference_config(
    preset: str = "medium",
    mode: str = "normal",
    decode_strategy: str = "greedy",
    checkpoint_path: str = None,
    **kwargs
) -> InferenceConfig:
    """创建推理配置"""
    
    # 获取基础配置
    base_config = get_preset_config(preset)
    mode_config = get_mode_config(mode)
    decode_config = get_decode_config(decode_strategy)
    
    # 合并配置
    config_dict = {
        **base_config,
        **mode_config,
        **decode_config,
        **kwargs
    }
    
    # 创建配置对象
    config = InferenceConfig(**config_dict)
    
    # 设置检查点路径
    if checkpoint_path:
        config.checkpoint_path = checkpoint_path
    
    # 根据模式创建特殊配置
    if config.use_moe:
        config.moe_config = create_moe_config(config)
    
    if config.use_mtp:
        config.mtp_config = create_mtp_config(config)
    
    return config


# 示例配置
EXAMPLE_CONFIGS = {
    "fast_inference": create_inference_config(
        preset="small",
        mode="normal",
        decode_strategy="greedy",
        max_length=32
    ),
    "high_quality": create_inference_config(
        preset="large",
        mode="mla",
        decode_strategy="beam_search",
        num_beams=8,
        max_length=128
    ),
    "creative_generation": create_inference_config(
        preset="medium",
        mode="normal",
        decode_strategy="nucleus_sampling",
        temperature=1.2,
        top_p=0.8
    ),
    "mtp_experimental": create_inference_config(
        preset="medium",
        mode="mtp",
        decode_strategy="sampling",
        temperature=0.9
    )
}


def get_example_config(config_name: str) -> InferenceConfig:
    """获取示例配置"""
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown example config: {config_name}. Available: {list(EXAMPLE_CONFIGS.keys())}")
    
    return EXAMPLE_CONFIGS[config_name]

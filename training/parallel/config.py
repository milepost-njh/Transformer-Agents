from dataclasses import dataclass
from enum import Enum


class ParallelMode(str, Enum):
    single = "single"
    dp = "dp"
    ddp = "ddp"
    tensor = "tensor"
    pipeline = "pipeline"
    hybrid = "hybrid"


@dataclass
class ParallelConfig:
    mode: ParallelMode = ParallelMode.single

    # DDP
    backend: str = "nccl"
    init_method: str = "env://"
    find_unused_parameters: bool = True  # 对MoE模型必须为True，避免未使用专家导致DDP报错

    # Tensor parallel (placeholder for future)
    tp_size: int = 1
    tp_impl: str = "megatron"

    # Pipeline parallel (placeholder for future)
    pp_size: int = 1
    micro_batch_size: int = 1
    schedule: str = "1f1b"

    # Expert parallel (placeholder for future)
    ep_size: int = 1



from .config import ParallelMode
from .ddp_backend import DDPBackend
from .dp_backend import DPBackend
from .base import ParallelBackend


def create_backend(cfg) -> ParallelBackend:
    if cfg.mode == ParallelMode.ddp:
        return DDPBackend(cfg)
    elif cfg.mode == ParallelMode.dp:
        return DPBackend(cfg)
    # 预留：single/tensor/pipeline/hybrid 可在后续扩展
    return DDPBackend(cfg)  # 默认使用 DDP



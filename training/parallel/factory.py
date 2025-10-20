from .config import ParallelMode
from .ddp_backend import DDPBackend
from .base import ParallelBackend


def create_backend(cfg) -> ParallelBackend:
    if cfg.mode == ParallelMode.ddp:
        return DDPBackend(cfg)
    # 预留：single/dp/tensor/pipeline/hybrid 可在后续扩展
    return DDPBackend(cfg)



import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .base import ParallelBackend


class DDPBackend(ParallelBackend):
    def init_dist(self) -> None:
        if dist.is_initialized():
            return
        backend = self.cfg.backend
        init_method = self.cfg.init_method
        dist.init_process_group(backend=backend, init_method=init_method)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    def wrap_model(self, model):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=self.cfg.find_unused_parameters)
        return model, device

    def get_samplers(self, train_dataset, val_dataset):
        if not dist.is_initialized():
            return None, None
        train_sampler = DistributedSampler(train_dataset, drop_last=True, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, drop_last=False, shuffle=False)
        return train_sampler, val_sampler

    def optimizer_step(self, optimizer, scaler=None):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def should_log(self, step: int) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def save_ckpt(self, path: str, **payload) -> None:
        # 仅在 rank0 保存
        if self.should_log(0):
            torch.save(payload, path)



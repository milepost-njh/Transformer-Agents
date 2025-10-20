from abc import ABC, abstractmethod


class ParallelBackend(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def init_dist(self) -> None:
        ...

    @abstractmethod
    def wrap_model(self, model):
        """Return (wrapped_model, device)."""
        ...

    @abstractmethod
    def get_samplers(self, train_dataset, val_dataset):
        """Return (train_sampler, val_sampler)."""
        ...

    @abstractmethod
    def optimizer_step(self, optimizer, scaler=None):
        ...

    @abstractmethod
    def should_log(self, step: int) -> bool:
        ...

    @abstractmethod
    def save_ckpt(self, path: str, **payload) -> None:
        ...



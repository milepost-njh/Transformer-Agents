"""
DataParallel Backend - 单进程多GPU训练
不需要使用 torchrun，直接 python 运行即可
"""
import torch
from torch.nn import DataParallel
from .base import ParallelBackend


class DPBackend(ParallelBackend):
    """DataParallel 后端（单进程多线程）"""
    
    def init_dist(self) -> None:
        """DP 不需要初始化分布式进程组"""
        pass
    
    def wrap_model(self, model):
        """用 DataParallel 包装模型"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            print(f"使用 DataParallel，GPU 数量: {gpu_count}")
            model = DataParallel(model)
        
        return model, device
    
    def get_samplers(self, train_dataset, val_dataset):
        """DP 不需要分布式采样器，返回 None"""
        return None, None
    
    def optimizer_step(self, optimizer, scaler=None):
        """标准优化器步骤"""
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    def should_log(self, step: int) -> bool:
        """总是记录日志（单进程）"""
        return True
    
    def save_ckpt(self, path: str, **payload) -> None:
        """保存 checkpoint"""
        torch.save(payload, path)


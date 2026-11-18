"""Transformer 论文中的线性预热与平方根倒数衰减。"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TransformerLRScheduler(LRScheduler):
    """按优化器更新次数为每个参数组设置论文公式计算的学习率。"""

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
    ) -> None:

        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model 必须是正整数")
        if type(warmup_steps) is not int or warmup_steps <= 0:
            raise ValueError("warmup_steps 必须是正整数")

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """计算下一次 optimizer.step() 使用的学习率。"""
        step = self.last_epoch + 1
        warmup_rate = step * self.warmup_steps ** (-1.5)
        decay_rate = step ** (-0.5)
        lr = self.d_model ** (-0.5) * min(warmup_rate, decay_rate)

        return [lr for _ in self.optimizer.param_groups]

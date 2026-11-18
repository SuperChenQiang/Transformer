"""组织 Transformer 的训练、验证与 checkpoint 保存和恢复。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import Tensor
from torch.optim import Optimizer

from transformer.model import Transformer
from transformer.training.loss import LabelSmoothedCrossEntropyLoss
from transformer.training.scheduler import TransformerLRScheduler


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: Optimizer,
        criterion: LabelSmoothedCrossEntropyLoss,
        scheduler: TransformerLRScheduler | None = None,
    ) -> None:
        if criterion.pad_id != model.config.pad_id:
            raise ValueError("损失函数与模型的 pad_id 必须一致")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("调度器必须绑定到 Trainer 使用的优化器")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        self.epoch = 0
        self.global_step = 0

    def train_epoch(self, batches: Iterable[dict[str, Tensor]]) -> float:
        loss = self._run_epoch(batches, training=True)
        self.epoch += 1
        return loss

    def evaluate(self, batches: Iterable[dict[str, Tensor]]) -> float:
        return self._run_epoch(batches, training=False)

    def _run_epoch(
        self,
        batches: Iterable[dict[str, Tensor]],
        *,
        training: bool,
    ) -> float:
        self.model.train(training)
        total_loss = 0.0
        total_tokens = 0

        with torch.set_grad_enabled(training):
            for batch in batches:
                targets = batch["tgt_labels"].to(self.device)
                num_tokens = targets.ne(self.criterion.pad_id).sum().item()
                if num_tokens == 0:
                    continue

                src_ids = batch["src_ids"].to(self.device)
                tgt_input_ids = batch["tgt_input_ids"].to(self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(src_ids, tgt_input_ids)
                loss = self.criterion(logits, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()
                    self.global_step += 1

                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            raise ValueError("本轮没有有效的目标词元")

        return total_loss / total_tokens

    def save_checkpoint(self, path: str | Path) -> None:
        """在一轮训练结束后，保存继续训练所需的状态。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_config": self.model.config.to_dict(),
            "label_smoothing": self.criterion.label_smoothing,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

        with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as file:
            temp_path = Path(file.name)

        try:
            torch.save(checkpoint, temp_path)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        """将 checkpoint 恢复到已构造好的模型、优化器和调度器。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if checkpoint["model_config"] != self.model.config.to_dict():
            raise ValueError("checkpoint 的模型配置与当前配置不同")
        if checkpoint["label_smoothing"] != self.criterion.label_smoothing:
            raise ValueError("checkpoint 的标签平滑系数与当前配置不同")

        scheduler_state = checkpoint["scheduler"]
        if (self.scheduler is None) != (scheduler_state is None):
            raise ValueError("checkpoint 与当前 Trainer 的调度器启用状态不同")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

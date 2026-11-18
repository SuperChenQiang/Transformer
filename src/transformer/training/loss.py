"""忽略 PAD，并按有效目标词元平均的标签平滑交叉熵。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """输入 logits [B, T, V] 和 targets [B, T]，返回标量损失。"""

    def __init__(
        self,
        pad_id: int = 0,
        label_smoothing: float = 0.1,
    ) -> None:

        super().__init__()

        if type(pad_id) is not int or pad_id < 0:
            raise ValueError("pad_id 必须是非负整数")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing 必须在 [0, 1) 范围内")

        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """忽略 PAD 标签；整个 batch 都是 PAD 时返回可反向传播的零。"""
        if logits.ndim != 3 or targets.ndim != 2:
            raise ValueError("logits 必须是 [B, T, V]，targets 必须是 [B, T]")
        if logits.shape[:2] != targets.shape:
            raise ValueError("logits 与 targets 的 batch、序列长度必须一致")
        if targets.dtype != torch.long:
            raise TypeError("targets 必须使用 torch.long")
        if self.pad_id >= logits.size(-1):
            raise ValueError("pad_id 必须小于目标词表大小")

        vocab_size = logits.size(-1)

        # logits：[B, T, V]   targets：[B, T]   loss：标量
        total_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_id,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )

        num_tokens = targets.ne(self.pad_id).sum()

        return total_loss / num_tokens.clamp(min=1)

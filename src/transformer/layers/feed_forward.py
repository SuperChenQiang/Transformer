"""逐位置前馈网络。"""

from __future__ import annotations

from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """对每个位置独立执行 Linear → ReLU → Linear。"""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model 和 d_ff 必须为正整数")

        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """输入和输出的形状均为 [B, L, d_model]。"""
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError("x 的形状必须为 [B, L, d_model]")

        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)

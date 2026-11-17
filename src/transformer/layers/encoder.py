"""采用 Post-LN 结构的 Transformer 编码器。"""

from __future__ import annotations

from torch import Tensor, nn

from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """单层编码器：自注意力 → Add & Norm → 前馈网络 → Add & Norm。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """输入输出均为 [B, S, D]；mask 中 True 表示禁止关注。"""
        attention_output, _ = self.self_attention(x, x, x, mask=mask)
        # x_1=LayerNorm(x+Dropout(SelfAttention(x)))
        x = self.norm1(x + self.dropout1(attention_output))

        feed_forward_output = self.feed_forward(x)
        # x_2=LayerNorm(x_1+Dropout(FeedForward(x_1)))
        x = self.norm2(x + self.dropout2(feed_forward_output))

        return x


class Encoder(nn.Module):
    """将多个参数独立的 EncoderLayer 按顺序堆叠。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers 必须大于 0")

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for layer in range(num_layers)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """依次经过所有编码器层，返回 [B, S, D]。"""
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x

"""采用 Post-LN 结构的 Transformer 解码器。"""

from __future__ import annotations

from torch import Tensor, nn

from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """单层解码器：因果自注意力 → 交叉注意力 → 前馈网络。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """输入 x [B, T, D]、memory [B, S, D]，输出 [B, T, D]。
        tgt_mask：目标输入的因果与填充组合掩码，通常为 [B, 1, T, T]。
        memory_mask：源序列填充掩码，通常为 [B, 1, 1, S]。
        """

        # tgt_mask限制每个位置只能关注自己及前面允许的位置，同时屏蔽 PAD。
        attention_output, _ = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attention_output))

        cross_output, _ = self.cross_attention(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(feed_forward_output))

        return x


class Decoder(nn.Module):
    """将多个参数独立的 DecoderLayer 按顺序堆叠。"""

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
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for layer in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return x

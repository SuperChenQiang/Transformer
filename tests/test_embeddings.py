"""词嵌入与正弦位置编码。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """将词元编号 [B, L] 转换为经过缩放的词向量 [B, L, D]。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size 和 d_model 必须为正整数")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id 必须位于词表范围内")

        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """输入为 torch.long 的词元编号，形状为 [B, L]。"""
        if token_ids.dim() != 2:
            raise ValueError("token_ids 的形状必须为 [B, L]")

        return self.embedding(token_ids) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """为 [B, L, D] 的词向量添加正弦位置编码，再应用 dropout。"""

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0 or max_seq_len <= 0:
            raise ValueError("d_model 和 max_seq_len 必须为正整数")

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        angles = position * div_term

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """输入输出均为 [B, L, D]，序列长度不得超过 max_seq_len。"""
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError("x 的形状必须为 [B, L, d_model]")

        seq_len = x.size(1)
        if seq_len > self.pe.size(1):  # pyright: ignore[reportCallIssue]
            raise ValueError("输入序列长度超过预设的 max_seq_len")

        position_encoding = self.pe[:, :seq_len].to(dtype=x.dtype)  # type: ignore
        return self.dropout(x + position_encoding)


def main() -> None:
    torch.manual_seed(42)

    token_ids = torch.tensor(
        [
            [2, 4, 5, 0],
            [2, 6, 0, 0],
        ],
        dtype=torch.long,
    )

    embedding = TokenEmbedding(vocab_size=10, d_model=8, pad_id=0)
    position = SinusoidalPositionalEncoding(d_model=8, max_seq_len=16, dropout=0.0)

    vectors = embedding(token_ids)
    output = position(vectors)

    print("词元编号形状：", token_ids.shape)
    print("词嵌入形状：", vectors.shape)
    print("加入位置编码后的形状：", output.shape)
    print("位置 0 的编码：", position.pe[0, 0])  # pyright: ignore[reportIndexIssue]

    assert vectors.shape == (2, 4, 8)
    assert output.shape == (2, 4, 8)
    assert torch.all(vectors[token_ids == 0] == 0)
    assert torch.allclose(
        position.pe[0, 0],  # type: ignore
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
    )
    assert torch.allclose(output, vectors + position.pe[:, :4])  # type: ignore

    output.square().mean().backward()
    assert embedding.embedding.weight.grad is not None
    assert not position.pe.requires_grad

    odd_position = SinusoidalPositionalEncoding(d_model=7, max_seq_len=4, dropout=0.0)
    assert odd_position(torch.zeros(1, 4, 7)).shape == (1, 4, 7)

    print("Embedding 基础运行检查通过！")


if __name__ == "__main__":
    main()

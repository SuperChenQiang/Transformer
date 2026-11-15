from __future__ import annotations

import math

import torch
from torch import Tensor, nn

"""
    计算注意力，返回输出与 dropout 之前的注意力权重。
    query: [..., L_q, D_k]
    key:   [..., L_k, D_k]
    value: [..., L_k, D_v]
    mask: 布尔张量，True 表示禁止关注，须可广播到 [..., L_q, L_k]。
    输入序列非空；输入张量与 mask 应位于同一设备。
"""


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:

    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask 必须是布尔张量，True 表示禁止关注")

        scores = scores.masked_fill(mask, float("-inf"))

        # 整行被屏蔽时，避免对全 -inf 做 softmax 而产生 NaN。
        all_masked = mask.all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)

    weights = torch.softmax(scores, dim=-1)

    if mask is not None:
        # 同时保证整行被屏蔽时，返回的权重全部为 0。
        weights = weights.masked_fill(mask, 0.0)

    used_weights = dropout(weights) if dropout is not None else weights
    output = used_weights @ value

    return output, weights


class MultiHeadAttention(nn.Module):
    """多头注意力：输入 [B, L, D]，内部拆分为 [B, H, L, D/H]。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model 和 n_heads 必须为正整数")
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能够被 n_heads 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, L, D] -> [B, L, H, D_head] -> [B, H, L, D_head]。"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    """
        返回 output [B, L_q, D] 和 weights [B, H, L_q, L_k]。
        query/key/value 均为三维，最后一维均为 d_model。
        三者 batch 大小相同，key 与 value 的序列长度相同。
        mask 可用 [L_q, L_k]、[B, 1, 1, L_k]
        或 [B, 1, L_q, L_k] 等可广播形状。
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        if any(x.dim() != 3 or x.size(-1) != self.d_model for x in (query, key, value)):
            raise ValueError("query、key、value 的形状必须为 [B, L, d_model]")
        if query.size(0) != key.size(0) or key.size(0) != value.size(0):
            raise ValueError("query、key、value 的 batch 大小必须相同")
        if key.size(1) != value.size(1):
            raise ValueError("key 和 value 的序列长度必须相同")

        batch_size, query_len, _ = query.shape

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attention_dropout
        )

        # [B, H, L_q, D_head] -> [B, L_q, H, D_head] -> [B, L_q, D]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, query_len, self.d_model)

        output = self.out_proj(context)
        return output, weights


def main() -> None:
    torch.manual_seed(42)

    x = torch.randn(2, 4, 8)
    attention = MultiHeadAttention(d_model=8, n_heads=2, dropout=0.0)
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    output, weights = attention(x, x, x, mask=causal_mask)

    print("输入形状：", x.shape)
    print("自注意力输出形状：", output.shape)
    print("注意力权重形状：", weights.shape)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    assert torch.all(weights.masked_select(causal_mask) == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 4))

    output.square().mean().backward()
    assert attention.q_proj.weight.grad is not None

    memory = torch.randn(2, 6, 8)
    cross_output, cross_weights = attention(x, memory, memory)
    assert cross_output.shape == (2, 4, 8)
    assert cross_weights.shape == (2, 2, 4, 6)
    print("交叉注意力输出形状：", cross_output.shape)
    print("Attention 基础运行检查通过！")


if __name__ == "__main__":
    main()

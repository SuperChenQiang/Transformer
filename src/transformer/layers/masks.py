"""生成注意力使用的布尔掩码：True 表示禁止关注。"""

from __future__ import annotations

import torch
from torch import Tensor


def create_padding_mask(token_ids: Tensor, pad_id: int = 0) -> Tensor:
    """将 [B, L_k] 的词元编号转换为 [B, 1, 1, L_k] 的填充掩码。"""
    if token_ids.dim() != 2:
        raise ValueError("token_ids 的形状必须为 [B, L]")

    return (token_ids == pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(
    seq_len: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """生成 [1, 1, L, L] 的因果掩码，禁止关注未来位置。"""
    if seq_len <= 0:
        raise ValueError("seq_len 必须大于 0")

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)


def create_decoder_mask(tgt_input_ids: Tensor, pad_id: int = 0) -> Tensor:
    """组合目标填充掩码与因果掩码，返回 [B, 1, T, T]。"""
    padding_mask = create_padding_mask(tgt_input_ids, pad_id)
    causal_mask = create_causal_mask(
        tgt_input_ids.size(1),
        device=tgt_input_ids.device,
    )
    return padding_mask | causal_mask

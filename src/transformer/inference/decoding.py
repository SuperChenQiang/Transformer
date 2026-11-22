"""Transformer 的批量贪心解码。"""

from __future__ import annotations

import torch
from torch import Tensor

from transformer.model import Transformer


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_ids: Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
) -> Tensor:
    """返回 [B, L]，保留 BOS、生成的 EOS，并用 PAD 补齐已结束的行。"""
    if src_ids.ndim != 2 or src_ids.numel() == 0 or src_ids.dtype != torch.long:
        raise ValueError("src_ids 必须是非空的 [B, S] torch.long 张量")
    if not 1 <= max_new_tokens <= model.config.max_seq_len:
        raise ValueError("max_new_tokens 必须在 1 和模型最大长度之间")

    pad_id = model.config.pad_id
    special_ids = {pad_id, bos_id, eos_id}
    if len(special_ids) != 3 or not all(0 <= i < model.config.tgt_vocab_size for i in special_ids):
        raise ValueError("PAD、BOS、EOS 必须是目标词表中三个不同的编号")

    was_training = model.training
    model.eval()
    try:
        memory, src_mask = model.encode(src_ids)
        generated = src_ids.new_full((src_ids.size(0), 1), bos_id)
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)

        for _ in range(max_new_tokens):
            # decode 输出 [B, L, V]
            scores = model.decode(generated, memory, src_mask)[:, -1, :]
            scores[:, [pad_id, bos_id]] = -torch.inf
            next_ids = scores.argmax(dim=-1)
            next_ids = next_ids.masked_fill(finished, pad_id)

            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished = finished | next_ids.eq(eos_id)
            if finished.all():
                break

        return generated
    finally:
        model.train(was_training)

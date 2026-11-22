"""加载已训练的 Transformer，并对一条文本进行贪心生成。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from train import RUN_INFO, SRC_VOCAB, TGT_VOCAB, select_device

from transformer.config import TransformerConfig
from transformer.data.tokenizer import WordTokenizer
from transformer.inference.decoding import greedy_decode
from transformer.model import Transformer


def load_model_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Transformer, WordTokenizer, WordTokenizer, dict]:
    directory = checkpoint_path.parent
    info = json.loads((directory / RUN_INFO).read_text(encoding="utf-8"))
    src_tokenizer = WordTokenizer.load(directory / SRC_VOCAB)
    tgt_tokenizer = WordTokenizer.load(directory / TGT_VOCAB)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["model_config"])
    if (
        config.src_vocab_size != src_tokenizer.vocab_size
        or config.tgt_vocab_size != tgt_tokenizer.vocab_size
        or config.pad_id != src_tokenizer.pad_id
        or config.pad_id != tgt_tokenizer.pad_id
    ):
        raise ValueError("checkpoint 的词表大小或 PAD 编号与分词器文件不一致")

    model = Transformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, src_tokenizer, tgt_tokenizer, info


def generate_texts(
    model: Transformer,
    src_tokenizer: WordTokenizer,
    tgt_tokenizer: WordTokenizer,
    texts: list[str],
    *,
    max_seq_len: int,
    max_new_tokens: int,
) -> list[str]:
    rows = []
    for text in texts:
        content = src_tokenizer.encode(text)[: max_seq_len - 1]
        rows.append(torch.tensor([*content, src_tokenizer.eos_id], dtype=torch.long))

    src_ids = pad_sequence(rows, batch_first=True, padding_value=src_tokenizer.pad_id)
    src_ids = src_ids.to(next(model.parameters()).device)

    token_ids = greedy_decode(
        model,
        src_ids,
        bos_id=tgt_tokenizer.bos_id,
        eos_id=tgt_tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
    )

    outputs = []
    for row in token_ids.tolist():
        end = row.index(tgt_tokenizer.eos_id) if tgt_tokenizer.eos_id in row else len(row)
        outputs.append(tgt_tokenizer.decode(row[1:end], skip_special_tokens=False))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    try:
        model, src_tokenizer, tgt_tokenizer, info = load_model_bundle(
            args.checkpoint, select_device(args.device)
        )
        limit = info["max_seq_len"]
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else limit
        result = generate_texts(
            model,
            src_tokenizer,
            tgt_tokenizer,
            [args.text],
            max_seq_len=limit,
            max_new_tokens=max_new_tokens,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(result[0])


if __name__ == "__main__":
    main()

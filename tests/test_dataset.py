"""文本对数据集与序列 batch 补齐。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformer.data.tokenizer import WordTokenizer


class ParallelTextDataset(Dataset[dict[str, Tensor]]):
    """将文本对编码为源输入、目标输入与目标标签。
    分词器须先完成 fit() 或 load()。
    max_seq_len 限制最终源输入与目标输入的长度，包含各自的特殊词元。
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        src_tokenizer: WordTokenizer,
        tgt_tokenizer: WordTokenizer,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        if max_seq_len < 1:
            raise ValueError("max_seq_len 必须至少为 1")
        if src_tokenizer.pad_id != tgt_tokenizer.pad_id:
            raise ValueError("当前项目要求源、目标分词器使用相同的 pad_id")

        self.pairs = list(pairs)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """返回文本对数量。"""
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """返回一条样本，三个字段都是尚未补齐的一维 torch.long 张量。"""

        src_text, tgt_text = self.pairs[index]

        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)

        # 为源 EOS、目标输入 BOS 以及目标标签 EOS 预留位置。
        src_tokens = src_tokens[: self.max_seq_len - 1]
        tgt_tokens = tgt_tokens[: self.max_seq_len - 1]

        src_ids = src_tokens + [self.src_tokenizer.eos_id]
        tgt_ids = [self.tgt_tokenizer.bos_id] + tgt_tokens + [self.tgt_tokenizer.eos_id]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_ids[:-1], dtype=torch.long),
            "tgt_labels": torch.tensor(tgt_ids[1:], dtype=torch.long),
        }


def collate_batch(
    batch: list[dict[str, Tensor]],
    pad_id: int = 0,
) -> dict[str, Tensor]:
    """将多条样本右侧补齐，返回 batch 维度在前的二维张量。"""
    if not batch:
        raise ValueError("batch 不能为空")

    result: dict[str, Tensor] = {}
    for name in ("src_ids", "tgt_input_ids", "tgt_labels"):
        sequences = [sample[name] for sample in batch]
        result[name] = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=pad_id,
        )

    return result


def main() -> None:
    texts = ["i love learning", "i learn"]

    tokenizer = WordTokenizer()
    tokenizer.fit(texts)

    # 复制任务：目标文本与源文本相同。
    pairs = [(text, text) for text in texts]
    dataset = ParallelTextDataset(
        pairs,
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        max_seq_len=8,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_batch,
    )
    batch = next(iter(loader))

    print("源输入：")
    print(batch["src_ids"])
    print("目标输入：")
    print(batch["tgt_input_ids"])
    print("目标标签：")
    print(batch["tgt_labels"])

    assert len(dataset) == 2
    assert batch["src_ids"].shape == (2, 4)
    assert batch["tgt_input_ids"].shape == (2, 4)
    assert batch["tgt_labels"].shape == (2, 4)
    assert all(value.dtype == torch.long for value in batch.values())

    assert torch.all(batch["tgt_input_ids"][:, 0] == tokenizer.bos_id)
    assert batch["tgt_labels"][1, 2] == tokenizer.eos_id
    assert batch["tgt_input_ids"][1, -1] == tokenizer.pad_id
    assert torch.equal(batch["src_ids"], batch["tgt_labels"])
    assert torch.equal(
        batch["tgt_input_ids"][0, 1:],
        batch["tgt_labels"][0, :-1],
    )

    print("Dataset 基础运行检查通过！")


if __name__ == "__main__":
    main()

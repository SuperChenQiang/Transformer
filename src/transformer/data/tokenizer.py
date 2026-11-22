"""构建和保存词表，支持空白切词、英文切词与中文字符切分。"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

"""固定四个特殊词元"""
SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
)
ENGLISH_TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")


class WordTokenizer:
    """按所选规则切分文本，用训练集建立固定的词元编号。"""

    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __init__(
        self,
        *,
        mode: str = "word",
        lowercase: bool = True,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> None:
        if mode not in {"word", "en", "char"}:
            raise ValueError("mode 必须是 word、en 或 char")
        if min_freq < 1:
            raise ValueError("min_freq 必须至少为 1")
        if max_vocab_size is not None and max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab_size 必须容纳四个特殊词元")

        self.mode = mode
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.id_to_token = list(SPECIAL_TOKENS)
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

        self._fitted = False

    @property
    def vocab_size(self) -> int:
        """词表大小，包含特殊词元。"""
        return len(self.id_to_token)

    def tokenize(self, text: str) -> list[str]:
        """英文拆开标点并忽略句末句号，中文按非空白字符切分。"""
        if self.lowercase:
            text = text.lower()
        if self.mode == "en":
            text = unicodedata.normalize("NFKC", text).replace("’", "'")
            return ENGLISH_TOKEN.findall(text.strip().rstrip("."))
        if self.mode == "char":
            return [char for char in text if not char.isspace()]
        return text.split()

    def fit(self, texts: Iterable[str]) -> None:
        """只使用训练集文本建词表；再次调用会重建词表。"""
        if isinstance(texts, str):
            raise TypeError("texts 应为多条文本组成的可迭代对象，例如 [句子1, 句子2]")

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        tokens = [
            token
            for token, count in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= self.min_freq and token not in SPECIAL_TOKENS
        ]

        if self.max_vocab_size is not None:
            tokens = tokens[: self.max_vocab_size - len(SPECIAL_TOKENS)]

        # 完成筛选后，重新建立两个映射。
        self.id_to_token = list(SPECIAL_TOKENS) + tokens
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        self._fitted = True

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """将文本转为编号；未知词使用 unk_id，不在此处补齐长度。"""
        self._check_fitted()

        ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]

        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """将编号还原为文本；字符模式直接拼接，其他模式以空格分隔。"""
        self._check_fitted()

        tokens = []
        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"词元编号超出词表范围：{token_id}")

            token = self.id_to_token[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return ("" if self.mode == "char" else " ").join(tokens)

    def save(self, path: str | Path) -> None:
        """保存预处理设置和有序词表，编号由词表顺序决定。"""
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": self.mode,
            "lowercase": self.lowercase,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "vocab": self.id_to_token,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        """加载词表，保持保存前的词元编号。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        tokenizer = cls(
            mode=data.get("mode", "word"),
            lowercase=data["lowercase"],
            min_freq=data["min_freq"],
            max_vocab_size=data["max_vocab_size"],
        )

        vocab = data["vocab"]
        if (
            not isinstance(vocab, list)
            or not all(isinstance(token, str) for token in vocab)
            or tuple(vocab[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS
            or len(set(vocab)) != len(vocab)
        ):
            raise ValueError("词表必须包含正确的特殊词元顺序，且不能有重复词元")

        tokenizer.id_to_token = vocab
        tokenizer.token_to_id = {token: index for index, token in enumerate(vocab)}
        tokenizer._fitted = True

        return tokenizer

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 建立词表，或通过 load() 加载词表")

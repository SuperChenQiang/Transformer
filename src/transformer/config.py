"""Transformer 模型配置与 YAML 加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TransformerConfig:
    """保存模型参数；源、目标词表大小由分词器提供。英翻中"""

    src_vocab_size: int  # 源语言（如英文）词表大小
    tgt_vocab_size: int  # 目标语言（如中文）词表大小
    d_model: int = 512  # Token隐藏维度
    n_heads: int = 8  # 多头注意力的头数
    num_encoder_layers: int = 6  # 编码器堆叠层数
    num_decoder_layers: int = 6  # 解码器堆叠层数
    d_ff: int = 2048  # 前馈网络隐藏层维度
    dropout: float = 0.1  # Dropout 概率
    max_seq_len: int = 512  # 最大序列长度
    pad_id: int = 0  # <pad>Token ID

    def __post_init__(self) -> None:
        """创建配置后，检查参数是否满足模型要求。"""
        positive_fields = (
            "src_vocab_size",
            "tgt_vocab_size",
            "d_model",
            "n_heads",
            "num_encoder_layers",
            "num_decoder_layers",
            "d_ff",
            "max_seq_len",
        )

        for name in positive_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model 必须能够被 n_heads 整除")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须满足 0 <= dropout < 1")

        if type(self.pad_id) is not int or not (
            0 <= self.pad_id < min(self.src_vocab_size, self.tgt_vocab_size)
        ):
            raise ValueError("pad_id 必须是源、目标词表范围内的非负整数")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        src_vocab_size: int,
        tgt_vocab_size: int,
    ) -> TransformerConfig:
        """读取读取 YAML 并创建配置对象"""
        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError("YAML 顶层必须是字典")

        model_config = data.get("model")
        tokenizer_config = data.get("tokenizer", {})

        if not isinstance(model_config, dict):
            raise ValueError("YAML 必须包含 model 字典")

        if not isinstance(tokenizer_config, dict):
            raise ValueError("tokenizer 必须是字典")

        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            pad_id=tokenizer_config.get("pad_id", 0),
            **model_config,
        )

    def to_dict(self) -> dict[str, int | float]:
        """转换成普通字典，便于记录配置或保存 checkpoint。"""
        return asdict(self)

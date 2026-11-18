"""组装 Encoder–Decoder Transformer，输出目标词表上的原始 logits。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from transformer.config import TransformerConfig
from transformer.layers.decoder import Decoder
from transformer.layers.embeddings import SinusoidalPositionalEncoding, TokenEmbedding
from transformer.layers.encoder import Encoder
from transformer.layers.masks import create_decoder_mask, create_padding_mask


class Transformer(nn.Module):
    """连接嵌入、位置编码、编码器、解码器与输出投影。"""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.src_embedding = TokenEmbedding(config.src_vocab_size, config.d_model, config.pad_id)
        self.tgt_embedding = TokenEmbedding(config.tgt_vocab_size, config.d_model, config.pad_id)

        self.src_position = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.tgt_position = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        self.encoder = Encoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout,
        )
        self.decoder = Decoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
        )
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """初始化线性层与嵌入表，并保持 PAD 嵌入初值为零。"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.src_embedding.embedding.weight[self.config.pad_id].zero_()
            self.tgt_embedding.embedding.weight[self.config.pad_id].zero_()

    def encode(self, src_ids: Tensor) -> tuple[Tensor, Tensor]:
        """将源词元 [B, S] 编码为 memory [B, S, D]，同时返回源填充掩码。"""
        src_mask = create_padding_mask(src_ids, self.config.pad_id)
        x = self.src_position(self.src_embedding(src_ids))
        memory = self.encoder(x, mask=src_mask)
        return memory, src_mask

    def decode(
        self,
        tgt_input_ids: Tensor,
        memory: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        """根据目标输入 [B, T] 与 memory，返回 logits [B, T, V_tgt]。"""

        # tgt_mask：目标自注意力，屏蔽未来位置与目标 PAD
        # mask：交叉注意力，屏蔽源 PAD
        tgt_mask = create_decoder_mask(tgt_input_ids, self.config.pad_id)
        x = self.tgt_position(self.tgt_embedding(tgt_input_ids))

        hidden = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )
        return self.output_projection(hidden)

    def forward(self, src_ids: Tensor, tgt_input_ids: Tensor) -> Tensor:
        """源、目标输入均为 torch.long 词元编号；返回原始 logits。"""
        memory, src_mask = self.encode(src_ids)
        return self.decode(tgt_input_ids, memory, src_mask)

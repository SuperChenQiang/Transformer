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


def main() -> None:
    torch.manual_seed(42)

    config = TransformerConfig(
        src_vocab_size=20,
        tgt_vocab_size=30,
        d_model=16,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=16,
    )
    model = Transformer(config)

    src_ids = torch.tensor(
        [
            [4, 5, 3, 0, 0],
            [6, 7, 8, 3, 0],
        ],
        dtype=torch.long,
    )
    tgt_input_ids = torch.tensor(
        [
            [2, 9, 10, 0],
            [2, 11, 0, 0],
        ],
        dtype=torch.long,
    )

    logits = model(src_ids, tgt_input_ids)

    print("源输入形状：", src_ids.shape)
    print("目标输入形状：", tgt_input_ids.shape)
    print("模型输出形状：", logits.shape)

    assert logits.shape == (2, 4, 30)
    assert torch.isfinite(logits).all()

    # 分开调用 encode、decode，应与完整 forward 的结果一致。
    memory, src_mask = model.encode(src_ids)
    separate_logits = model.decode(tgt_input_ids, memory, src_mask)
    assert torch.allclose(logits, separate_logits, atol=1e-6)

    # 修改后面位置的词元，不应影响前两个位置。
    changed_ids = tgt_input_ids.clone()
    changed_ids[:, 2] = 12
    changed_logits = model(src_ids, changed_ids)
    assert torch.allclose(logits[:, :2], changed_logits[:, :2], atol=1e-6)

    logits.square().mean().backward()
    assert model.src_embedding.embedding.weight.grad is not None
    assert model.tgt_embedding.embedding.weight.grad is not None
    assert model.output_projection.weight.grad is not None

    print("Transformer 基础运行检查通过！")


if __name__ == "__main__":
    main()

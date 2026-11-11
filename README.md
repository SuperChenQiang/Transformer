# Transformer

使用 PyTorch 从零复现 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
中的 Encoder–Decoder Transformer，包含多头注意力、正弦位置编码和 Post-LN。

**当前处于开发阶段：工程结构和配置已就绪，模型、训练与推理模块待实现。**

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/SuperChenQiang/Transformer.git
cd Transformer
uv sync --locked --no-dev
```

依赖通过 `pyproject.toml` 和 `uv.lock` 管理，无需逐个安装。
参与开发时执行 `uv sync --locked`，同时安装 pytest、Ruff 等开发工具。

## 项目结构

```text
src/transformer/
├── config.py       # 模型配置
├── model.py        # 完整 Transformer
├── layers/         # 嵌入、注意力、前馈网络、编码器和解码器
├── data/           # 分词器与数据集
├── training/       # 训练器、损失与学习率调度
└── inference/      # 自回归解码
configs/            # tiny / base 配置模板
scripts/            # 训练、翻译、评估入口
tests/              # 模块测试
docs/               # 架构约定与实现顺序
```

## 使用

入口目前为占位，可查看参数：

```bash
uv run --no-dev python scripts/train.py --help
uv run --no-dev python scripts/translate.py --help
uv run --no-dev python scripts/evaluate.py --help
```

模型与训练流程实现后，目标用法为：

```bash
uv run --no-dev python scripts/train.py --config configs/tiny.yaml
uv run --no-dev python scripts/translate.py --checkpoint outputs/checkpoints/model.pt --text "hello world"
```

以上训练和翻译功能尚未接通。`tiny.yaml` 用于本地调试，`base.yaml` 提供 Base 模型主要维度。

从 `data/tokenizer.py` 开始实现，随后完成注意力等模块。
详细步骤见 [实现顺序](docs/implementation_order.md)，接口约定见 [架构说明](docs/architecture.md)。

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

借鉴范围见 [参考说明](docs/references.md)。

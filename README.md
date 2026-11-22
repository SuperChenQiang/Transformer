# Transformer 从零实现英译中

用 PyTorch 实现 [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) 的 Encoder–Decoder Transformer。
默认用真实英中句对训练，输入一句英文，模型逐字生成中文。项目保留了从注意力公式到训练和评估的完整代码，适合按文件逐步复现。

```text
Transformer/
├── configs/
│   ├── translation.yaml          # 默认：Tatoeba 英译中
│   ├── tiny.yaml                 # 可选：合成复制任务，用于排查流程
│   └── base.yaml                 # 论文 Base 模型尺寸，供自定义语料使用
├── scripts/
│   ├── prepare_translation.py    # 下载语料、去重、分组划分
│   ├── train.py                  # 训练、保存和继续训练
│   ├── translate.py              # 加载模型，翻译一句英文
│   └── evaluate.py               # 独立测试集的 BLEU、chrF 和译文
└── src/transformer/
    ├── config.py                 # 模型参数与配置校验
    ├── model.py                  # 组装完整 Transformer
    ├── layers/
    │   ├── attention.py          # 缩放点积和多头注意力
    │   ├── masks.py              # 填充掩码和因果掩码
    │   ├── embeddings.py         # 词嵌入和正弦位置编码
    │   ├── feed_forward.py        # 每个位置的前馈网络
    │   ├── encoder.py             # EncoderLayer 和 Encoder
    │   └── decoder.py             # DecoderLayer 和 Decoder
    ├── data/
    │   ├── tokenizer.py           # 英文切词、中文字符切分、词表
    │   └── dataset.py             # 目标输入与标签错位、batch 补齐
    ├── training/
    │   ├── loss.py                # 忽略 PAD 的交叉熵
    │   ├── scheduler.py           # 可选：论文的学习率预热和衰减
    │   └── trainer.py             # 前向、反向、更新和 checkpoint
    └── inference/
        └── decoding.py            # 从 BOS 开始贪心生成
```

[快速开始](#快速开始) · [按文件学习](#按文件学习) · [数据与评估](#数据与评估)

## 快速开始

需要 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。`.python-version` 指定 Python 3.12。
第一次安装依赖和下载语料需要联网。已下载代码的读者可直接在项目根目录运行安装命令。

### 1. 安装并准备语料

```bash
git clone git@github.com:SuperChenQiang/Transformer.git
cd Transformer
uv sync --locked --no-dev
uv run --locked --no-dev python scripts/prepare_translation.py
```

GitHub 克隆需要仓库访问权限和已配置的 SSH。准备脚本会在 `data/tatoeba-en-zh/` 生成训练、验证和测试 TSV，
以及带原始句子署名的 `attribution.tsv`。默认快照得到 **28,961 个训练句对、500 条验证样本和 500 条测试样本**。

### 2. 从零训练模型

```bash
uv run --locked --no-dev python scripts/train.py --config configs/translation.yaml
```

默认训练 20 轮，自动选择 CUDA、Apple MPS 或 CPU。也可以用 `--device cpu` 指定设备。
如果只想确认环境能跑，先用 `--epochs 1 --output-dir outputs/translation-smoke`，再用完整配置训练。
每轮会显示训练和验证 loss，并在 `outputs/checkpoints/en-zh/` 保存最新 `model.pt`。
两个分词器 JSON 和 `run.json` 也保存在同一目录。移动模型时请保留这四个文件。

### 3. 翻译一句英文，再评估测试集

```bash
uv run --locked --no-dev python scripts/translate.py --checkpoint outputs/checkpoints/en-zh/model.pt --text "I love you"
uv run --locked --no-dev python scripts/evaluate.py --config configs/translation.yaml --checkpoint outputs/checkpoints/en-zh/model.pt --examples 3
```

第一条会从英文生成中文。第二条打印语料级 BLEU、chrF、评分签名和几条预测与参考译文。
`I love you` 对应的句子在语料训练集中，适合演示命令。新句子的表现请看独立测试集。

一次 Apple Silicon MPS 实跑 20 轮得到以下结果，其他设备上的输出可能有所差异：

```text
I love you.       → 我爱你。
Good morning.     → 早上好。
I am a student.   → 我是学生。
```

在 500 条独立测试句上，BLEU 为 **35.13**（`tok:zh`），chrF 为 **29.96**。
例如，一个测试句 `I didn't mean to imply anything.` 被译为「我没想暗示什么。」，
但习语 `So far, so good.` 在这次试跑中译错了。

### 4. 继续训练

```bash
uv run --locked --no-dev python scripts/train.py --config configs/translation.yaml --resume outputs/checkpoints/en-zh/model.pt --epochs 25
```

`--epochs` 表示总轮数。第 20 轮结束后执行这条命令会再跑 5 轮。
想重新训练一个模型，用 `--output-dir` 指定新目录。旧的复制任务 checkpoint 仍可用于复制配置。

## 按文件学习

先跑一次翻译，再按下面的顺序读模型。每行对应一个 Python 文件，先重写列出来的关键部分。
配置解析、语料下载和命令行入口可以等你要改训练设置时再看。

| 顺序 | Python 文件 | 优先看什么 | 怎么验证 |
| --- | --- | --- | --- |
| 1 | [attention.py](src/transformer/layers/attention.py) | `scaled_dot_product_attention()`、`MultiHeadAttention.forward()` | 从 `QKᵀ / √d_head` 算权重，再乘 V。打印拆头后的 `[B, H, L, D/H]`。 |
| 2 | [masks.py](src/transformer/layers/masks.py) | `create_padding_mask()`、`create_causal_mask()` | 取 3 个位置，确认未来位置和 PAD 列会被屏蔽，`True` 表示屏蔽。 |
| 3 | [embeddings.py](src/transformer/layers/embeddings.py) | `TokenEmbedding.forward()`、`SinusoidalPositionalEncoding.forward()` | 从词元编号得到 `[B, L, D]`，观察同一个词在不同位置加上的编码。 |
| 4 | [feed_forward.py](src/transformer/layers/feed_forward.py) | `PositionwiseFeedForward.forward()` | 跟踪 `D → d_ff → D`，确认每个位置用同一套参数独立计算。 |
| 5 | [encoder.py](src/transformer/layers/encoder.py) | `EncoderLayer.forward()` | 找到源序列自注意力和两次残差加层归一化。 |
| 6 | [decoder.py](src/transformer/layers/decoder.py) | `DecoderLayer.forward()` | 区分带因果掩码的目标自注意力，以及读取源句的交叉注意力。 |
| 7 | [model.py](src/transformer/model.py) | `encode()`、`decode()`、`forward()` | 先走一次完整前向，确认输出为 `[B, T, V_tgt]` 的 logits。 |
| 8 | [dataset.py](src/transformer/data/dataset.py) | `ParallelTextDataset.__getitem__()` | 打印目标输入 `[BOS, 我, 爱, 你]` 和标签 `[我, 爱, 你, EOS]`，看清一位错位。 |
| 9 | [loss.py](src/transformer/training/loss.py) | `LabelSmoothedCrossEntropyLoss.forward()` | 修改 PAD 位置的 logits，确认损失保持不变。 |
| 10 | [trainer.py](src/transformer/training/trainer.py) | `Trainer._run_epoch()` | 跟一批数据走完前向、反向和参数更新。 |
| 11 | [decoding.py](src/transformer/inference/decoding.py) | `greedy_decode()` | 打印每轮目标前缀，确认新词接回去，遇 EOS 停止。 |

**先记住三种注意力的来源。** 编码器自注意力的 Q、K、V 都来自源序列。
解码器自注意力的 Q、K、V 都来自目标前缀。交叉注意力的 Q 来自解码器，K、V 来自编码器输出。
训练时模型使用已知目标前缀，翻译时从 BOS 开始，把自己生成的词元接回去。

第一个练习只需要两个位置。把下面的 Python 片段放进 VS Code 的临时文件，用项目解释器运行，
再尝试去掉 `mask`，观察第一行的权重如何变化。

```python
import torch

from transformer.layers.attention import scaled_dot_product_attention
from transformer.layers.masks import create_causal_mask

q = k = torch.eye(2)
v = torch.tensor([[10.0], [20.0]])
mask = create_causal_mask(2)[0, 0]
output, weights = scaled_dot_product_attention(q, k, v, mask=mask)

print(weights)
torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0]))
torch.testing.assert_close(output[0], v[0])
```

## 数据与评估

默认语料来自 [ManyThings 的 Tatoeba 英中句对](https://www.manythings.org/anki/)，使用 2026-02-13 快照。
原始语料采用 [CC BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/)，逐句署名随数据保存在
`attribution.tsv`。

准备数据时将中文转为简体，按归一化英文源句分组，确保同一源句的多个译文留在同一数据划分。
词表只从训练集建立。英文切词会分开标点并忽略句末句号，中文按字符切分，生成后直接拼接。
这是用于学习的短句翻译模型，长句、罕见词和新的主题都可能翻译不准。BLEU 和 chrF 反映与参考译文的重合程度，
应结合预测样例一起看。这里的结果不与论文 WMT 成绩直接比较。

可以将自己的英中语料保存为三个 UTF-8 TSV 文件，每行是 `英文<Tab>中文`，无表头。
复制 [translation.yaml](configs/translation.yaml)，修改 `data.train_path`、`data.validation_path`、
`data.test_path` 和 `training.checkpoint_dir`，训练和评估时通过 `--config` 指向该文件。

资料：[Transformer 原论文](https://arxiv.org/html/1706.03762v7) ·
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) ·
[MiniMind](https://github.com/jingyaogong/minimind)（快速上手的组织参考）

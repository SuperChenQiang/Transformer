# Transformer 从零实现

用 PyTorch 复现 [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) 的 Encoder–Decoder Transformer。
你可以先在 CPU 上跑完一个小任务，再沿着数据流亲手实现注意力、编码器、解码器和训练循环。

[快速开始](#快速开始) · [学习路线](#学习路线) · [使用自己的翻译数据](#使用自己的翻译数据)

## 快速开始

需要 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。仓库的 `.python-version` 指定了
Python 3.12。首次安装依赖需要联网。复制任务的数据由代码生成，训练可以在 CPU 上完成。

### 1. 安装

从 GitHub 克隆需要有仓库访问权限，并为 GitHub 配置 SSH。已经下载代码的读者可直接在仓库根目录执行最后一行。

```bash
git clone git@github.com:SuperChenQiang/Transformer.git
cd Transformer
uv sync --locked --no-dev
```

### 2. 从零训练

[configs/tiny.yaml](configs/tiny.yaml) 使用复制任务。例如，输入 `red blue green yellow`，目标就是同一句话。
代码用固定种子生成 192 条训练样本、32 条验证样本和 32 条测试样本，三个集合的句子互不重叠。

```bash
uv run --locked --no-dev python scripts/train.py --device cpu --epochs 10 --output-dir outputs/quickstart
```

看到 `epoch=10/10 step=120` 就说明 10 轮已跑完。每轮都会显示训练和验证 loss，并保存最新的
`model.pt`。目录中还会生成 `src_tokenizer.json`、`tgt_tokenizer.json` 和 `run.json`。
复制或移动模型时，请把这四个文件放在一起。

### 3. 生成和评估

```bash
uv run --locked --no-dev python scripts/translate.py --device cpu --checkpoint outputs/quickstart/model.pt --text "I love you"
uv run --locked --no-dev python scripts/evaluate.py --device cpu --checkpoint outputs/quickstart/model.pt
```

第一条生成一句话，第二条在 32 条测试样本上打印 **BLEU、评分签名和整句完全匹配率**。
一次 Apple Silicon CPU 试跑中，示例句成功复制，测试集完全匹配 `2/32`。
短训练适合检查流程，生成仍可能出现漏词或重复。可以用验证 loss 观察训练进展。

### 4. 从 checkpoint 继续

```bash
uv run --locked --no-dev python scripts/train.py --device cpu --resume outputs/quickstart/model.pt --epochs 12
```

`--epochs 12` 是总轮数。从第 10 轮恢复后会再训练两轮。想从头再做一次实验时，给 `--output-dir` 换一个新目录。

## 学习路线

第一次只跟下面这条主线。每一步选一个函数自己重写，用小张量核对行为，再进入下一步。
配置解析、设备选择、词表保存和命令行参数可以等需要修改时再看。

| 顺序 | 先看哪里 | 动手验证什么 |
| --- | --- | --- |
| 1. 注意力与掩码 | [`attention.py`](src/transformer/layers/attention.py) 的 `scaled_dot_product_attention`，然后看 [`masks.py`](src/transformer/layers/masks.py) | 先用两个词元算出注意力权重，再加因果掩码，确认未来位置的权重是 0。 |
| 2. 多头与模型 | [`MultiHeadAttention`](src/transformer/layers/attention.py)、[`EncoderLayer`](src/transformer/layers/encoder.py)、[`DecoderLayer`](src/transformer/layers/decoder.py)，最后读 [`Transformer.forward`](src/transformer/model.py) | 跟踪 `[B, L, D] → [B, H, L, D/H] → [B, L, D]`。找出三种注意力分别从哪里取 Q、K、V，再跑一次前向，确认 logits 的形状是 `[B, T, V]`。 |
| 3. 标签与训练 | [`ParallelTextDataset.__getitem__`](src/transformer/data/dataset.py)、[`loss.py`](src/transformer/training/loss.py)、[`Trainer._run_epoch`](src/transformer/training/trainer.py) | 打印一个 batch 的目标输入和标签，确认标签向前错开一位。给标签加上 PAD，检查它对损失的影响，再用同一个小 batch 训练几次。 |
| 4. 自回归生成 | [`greedy_decode`](src/transformer/inference/decoding.py) | 打印每轮的目标前缀，确认编码器只执行一次，每轮把新词接回前缀，到 EOS 或长度上限停止。 |

这些维度里，`B` 是批大小，`L` 是序列长度，`D` 是隐藏维度，`H` 是头数，`T` 是目标长度，`V` 是目标词表大小。
注意力的核心是 `Q @ K.T / sqrt(d_head) → mask → softmax → @ V`。编码器自注意力的 Q、K、V 来自源序列，
解码器自注意力的 Q、K、V 来自目标前缀，交叉注意力的 Q 来自解码器，K、V 来自编码器输出。
注意力外还有残差和层归一化，每层用前馈网络独立处理各个位置。第一次读到位置编码时，先跟踪它在哪里加到词嵌入上即可。

训练和生成的区别可以用一句话检查：

```text
源输入：      red blue <eos>
目标输入：    <bos> red blue
监督标签：    red blue <eos>

训练：按已知目标输入预测每个位置的下一个词。
生成：从 <bos> 开始，把自己刚预测的词接回去，再预测下一个。
```

### 第一个动手练习

在仓库根目录的 Bash 或 Zsh 中运行。也可以把 Python 部分粘贴到 VS Code 的临时脚本里，用项目的 `.venv` 运行。

```bash
uv run --locked --no-dev python - <<'PY'
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
PY
```

第一行权重应为 `[1, 0]`，所以第一个位置的输出等于第一个 Value。试着去掉 `mask=mask`，观察第一行权重如何变化。

## 使用自己的翻译数据

准备独立的训练、验证、测试文件。每个文件使用 UTF-8 TSV，每行是一个源句、一个 Tab、一个目标句，无表头。
复制 [configs/base.yaml](configs/base.yaml)，填写 `data.train_path`、`data.validation_path` 和 `data.test_path`，
并给 `training.checkpoint_dir` 选一个新目录。首次试自己的语料也可以使用 tiny 配置的模型尺寸。
训练和评估时用 `--config` 指向这份配置，生成和评估时用 `--checkpoint` 指向训练结果。

当前分词器按空白切分，中文等语料需要使用一致的预分词方式。小型复制任务和 TSV 流程用于学习和验证实现。
要复现论文的 WMT 成绩，还需准备相应语料、子词分词、训练规模和评估设置。

## 参考

- [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) — 先读第 3 节的模型结构，再看第 5.3 节的优化设置。
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — 论文公式和张量实现的对照阅读。
- [MiniMind](https://github.com/jingyaogong/minimind) — 本 README 的快速上手组织参考。

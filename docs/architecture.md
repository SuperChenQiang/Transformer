# 架构约定

默认流程用 Tatoeba 英中句对训练编码器—解码器 Transformer，支持句子翻译、续训和独立测试集评估。

## 已确定的边界

- 目标：使用 PyTorch 基础组件复现 Encoder–Decoder Transformer。
- 核心注意力自行实现；不直接用 `nn.Transformer` 代替学习目标。
- 第一版采用论文中的 Post-LN、ReLU 和正弦位置编码。
- `src/transformer` 是可安装的 Python 包，`scripts/` 只负责命令入口。
- 环境依赖由 uv 管理，安装方式见 README。
- `scripts/prepare_translation.py` 单独准备真实英中语料，训练脚本读取准备好的 TSV。

## 接口约定

| 对象 | 约定 |
| --- | --- |
| 词元编号 | `torch.long`，形状 `[B, L]` |
| 隐藏表示 | `[B, L, D]`，batch 位于第一个维度 |
| 多头表示 | `[B, H, L, D/H]` |
| 布尔掩码 | `True` 表示禁止关注，必须可广播到注意力分数 |
| 模型输出 | 原始 logits，形状 `[B, T, target_vocab_size]` |
| 特殊词元 | 初始约定 `<pad>=0`、`<unk>=1`、`<bos>=2`、`<eos>=3` |
| 目标序列 | 解码器输入以 BOS 开始，监督标签以 EOS 结束 |

源、目标分词器可以有各自的词表。词表大小从训练数据构建出的分词器获得，
不在模型配置模板中填写虚假的固定词表大小。

自定义掩码传入 PyTorch 参考 API 时，需要检查该 API 的布尔掩码语义，
不要假定所有 API 都与本项目相同。

## 第一版分词器

`WordTokenizer` 封装词表和编码解码规则。默认英译中使用英文切词和中文字符切分，复制任务使用空白切词。
训练集建立两个独立词表。验证、测试和推理复用保存的词表。

## 配置模板

`configs/translation.yaml` 是 README 使用的英译中配置，`configs/tiny.yaml` 是合成复制调试配置。
`configs/base.yaml` 提供论文 Base 模型的主要维度，使用者需自行填写数据路径和设定训练规模。


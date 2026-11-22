# 从哪个文件开始写

第一次读代码，可按 [README 的模型学习路线](../README.md#按文件学习) 找到重点。
下表记录从零搭建整套项目时的建议实现顺序。

| 顺序 | 文件 | 首先实现 | 对应验证文件 |
| --- | --- | --- | --- |
| 1 | `src/transformer/data/tokenizer.py` | WordTokenizer：词表、encode/decode、保存加载 | `tests/test_tokenizer.py` |
| 2 | `src/transformer/layers/attention.py`、`masks.py` | 单头注意力与掩码，再加多头投影 | `tests/test_attention.py`、`test_masks.py` |
| 3 | `src/transformer/layers/embeddings.py` | 词嵌入、缩放与正弦位置编码 | `tests/test_embeddings.py` |
| 4 | `src/transformer/layers/feed_forward.py` | 两层线性变换和 ReLU | `tests/test_layers.py` |
| 5 | `src/transformer/layers/encoder.py` | EncoderLayer，再堆叠 Encoder | `tests/test_layers.py` |
| 6 | `src/transformer/layers/decoder.py` | 因果自注意力和交叉注意力 | `tests/test_layers.py` |
| 7 | `src/transformer/config.py`、`model.py` | 配置校验、encode/decode、完整模型 | `tests/test_model.py` |
| 8 | `src/transformer/data/dataset.py` | 复制任务、collate_fn、目标错位 | `tests/test_dataset.py` |
| 9 | `src/transformer/training/loss.py`、`scheduler.py` | PAD 屏蔽、标签平滑、warmup 调度 | `tests/test_training.py` |
| 10 | `src/transformer/training/trainer.py` | 小批次过拟合、验证、checkpoint | `tests/test_training.py` |
| 11 | `src/transformer/inference/decoding.py` | 自回归贪心解码 | `tests/test_decoding.py` |
| 12 | `scripts/train.py`、`translate.py`、`evaluate.py` | 将已验证的组件接入命令入口 | 完整流程验证 |

## 每个模块的工作方式

1. 阅读模块和 [架构约定](architecture.md)，先确定张量形状及掩码含义。
2. 在自己的练习文件里重写关键函数，用小张量检查前向和边界条件。
3. 改动模型后，从 README 的快速开始走一遍数据、训练和推理。

## 训练验收顺序

固定张量前向/反向 → 固定小批次过拟合 → 合成复制任务 → 小规模真实翻译。
当前真实任务为 Tatoeba 英译中。用验证集观察训练进展，用独立测试集报告 BLEU、chrF 和具体译文。

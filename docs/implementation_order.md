# 从哪个文件开始写

目录已经准备好，不需要按目录的显示顺序开发。
先完成可独立验证的模块，再组装有依赖的模块。

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

1. 阅读文件里的 TODO 和 `docs/architecture.md`，确定输入输出。
2. 编写模块代码，并在对应测试文件添加真实的行为测试。
3. 运行该模块测试，再运行全部测试和 Ruff。
4. 检查差异，自己执行 Git add / commit / push。

例如 tokenizer 完成后：

```bash
uv run pytest tests/test_tokenizer.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff
git add src/transformer/data/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: implement word tokenizer"
git push
```

目前 `tests/test_tokenizer.py` 只有待办说明，单独运行它会因没有收集到测试而退出。
添加真实测试函数后再使用上面的模块测试命令。

## 训练验收顺序

固定张量前向/反向 → 固定小批次过拟合 → 合成复制任务 → 小规模真实翻译。
数据来源、词表、语言对、算力预算和评估协议在进入真实任务时确定。
模型能够前向或训练 loss 下降，不代表翻译质量已经通过评估。

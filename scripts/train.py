"""按 YAML 配置训练 Transformer，并保存模型及配套词表。"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from random import Random

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformer.config import TransformerConfig
from transformer.data.dataset import ParallelTextDataset, collate_batch
from transformer.data.tokenizer import WordTokenizer
from transformer.model import Transformer
from transformer.training.loss import LabelSmoothedCrossEntropyLoss
from transformer.training.scheduler import TransformerLRScheduler
from transformer.training.trainer import Trainer

SRC_VOCAB = "src_tokenizer.json"
TGT_VOCAB = "tgt_tokenizer.json"
RUN_INFO = "run.json"


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(
        not isinstance(config.get(name), dict) for name in ("data", "training")
    ):
        raise ValueError("配置必须包含 data 和 training 字典")
    return config


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def read_parallel_tsv(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or not all(field.strip() for field in fields):
                raise ValueError(f"{path}:{number} 应为两列非空文本，以 Tab 分隔")
            pairs.append((fields[0].strip(), fields[1].strip()))

    if not pairs:
        raise ValueError(f"{path} 没有可用的文本对")
    return pairs


def load_pairs(data: dict, split: str, seed: int) -> list[tuple[str, str]]:
    if data["task"] == "copy":
        rng = Random(seed)
        words = "red blue green yellow black white pink orange".split()
        seen = set()
        texts = []

        while len(texts) < 256:
            text = " ".join(rng.choices(words, k=rng.randint(4, 6)))
            if text not in seen:
                seen.add(text)
                texts.append(text)

        partitions = {
            "train": texts[:192],
            "validation": texts[192:224],
            "test": texts[224:],
        }
        return [(text, text) for text in partitions[split]]

    if data["task"] != "translation":
        raise ValueError("data.task 必须是 copy 或 translation")

    paths = {
        name: Path(data[f"{name}_path"]).resolve()
        for name in ("train", "validation", "test")
        if data.get(f"{name}_path")
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("训练、验证和测试必须使用不同文件")
    if split not in paths:
        raise ValueError(f"请设置 data.{split}_path，指向 UTF-8 TSV 文件")

    return read_parallel_tsv(paths[split])


def run(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    data, training = config["data"], config["training"]

    seed = int(training.get("seed", 42))
    torch.manual_seed(seed)
    device = select_device(args.device or training.get("device", "auto"))
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 16))
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs 和 batch_size 必须为正整数")

    resume = args.resume.resolve() if args.resume else None
    if resume is not None:
        output_dir = resume.parent
        if args.output_dir and args.output_dir.resolve() != output_dir:
            raise ValueError("续训时 --output-dir 必须与 checkpoint 所在目录一致")
    else:
        output_dir = args.output_dir or Path(training["checkpoint_dir"])

    checkpoint_path = resume or (output_dir / "model.pt")
    src_vocab = output_dir / SRC_VOCAB
    tgt_vocab = output_dir / TGT_VOCAB
    run_path = output_dir / RUN_INFO

    if resume is None and any(
        path.exists() for path in (checkpoint_path, src_vocab, tgt_vocab, run_path)
    ):
        raise ValueError("该目录已有训练产物，请使用 --resume 或指定新的 --output-dir")

    train_pairs = load_pairs(data, "train", seed)
    valid_pairs = load_pairs(data, "validation", seed)
    tokenizer_options = config.get("tokenizer", {})
    if tokenizer_options.get("type", "word") != "word":
        raise ValueError("当前支持 tokenizer.type: word")
    src_mode = tokenizer_options.get("src_mode", "word")
    tgt_mode = tokenizer_options.get("tgt_mode", "word")

    if resume is not None:
        src_tokenizer = WordTokenizer.load(src_vocab)
        tgt_tokenizer = WordTokenizer.load(tgt_vocab)
        if (src_tokenizer.mode, tgt_tokenizer.mode) != (src_mode, tgt_mode):
            raise ValueError("继续训练时请沿用原来的源端和目标端分词模式")
    else:
        src_tokenizer = WordTokenizer(mode=src_mode)
        tgt_tokenizer = WordTokenizer(mode=tgt_mode, lowercase=tgt_mode != "char")
        src_tokenizer.fit(src for src, _ in train_pairs)
        tgt_tokenizer.fit(tgt for _, tgt in train_pairs)

    model_config = TransformerConfig.from_yaml(
        args.config,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
    )
    if model_config.pad_id != src_tokenizer.pad_id or model_config.pad_id != tgt_tokenizer.pad_id:
        raise ValueError("模型与分词器的 PAD 编号必须一致")

    max_seq_len = int(data.get("max_seq_len", model_config.max_seq_len))
    if not 1 <= max_seq_len <= model_config.max_seq_len:
        raise ValueError("data.max_seq_len 必须在 1 和模型最大长度之间")

    run_info = {"task": data["task"], "max_seq_len": max_seq_len, "seed": seed}

    train_set = ParallelTextDataset(train_pairs, src_tokenizer, tgt_tokenizer, max_seq_len)
    valid_set = ParallelTextDataset(valid_pairs, src_tokenizer, tgt_tokenizer, max_seq_len)
    collate = partial(collate_batch, pad_id=model_config.pad_id)
    shuffle_rng = torch.Generator()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=shuffle_rng,
    )
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=collate)

    model = Transformer(model_config).to(device)
    warmup_steps = training.get("warmup_steps")
    optimizer = Adam(
        model.parameters(),
        lr=0.0 if warmup_steps is not None else float(training.get("learning_rate", 0.001)),
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.98))),
        eps=float(training.get("adam_eps", 1e-9)),
    )

    scheduler = None
    if warmup_steps is not None:
        scheduler = TransformerLRScheduler(optimizer, model_config.d_model, warmup_steps)

    criterion = LabelSmoothedCrossEntropyLoss(
        pad_id=model_config.pad_id,
        label_smoothing=float(training.get("label_smoothing", 0.0)),
    )
    trainer = Trainer(model, optimizer, criterion, scheduler)

    if resume is not None:
        if json.loads(run_path.read_text(encoding="utf-8")) != run_info:
            raise ValueError("续训的任务、序列长度和随机种子须与原训练一致")
        trainer.load_checkpoint(resume)
        print(f"已恢复 epoch={trainer.epoch}，step={trainer.global_step}")
    else:
        src_tokenizer.save(src_vocab)
        tgt_tokenizer.save(tgt_vocab)
        run_path.write_text(json.dumps(run_info, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"设备={device}，训练样本={len(train_set)}，验证样本={len(valid_set)}")
    while trainer.epoch < epochs:
        shuffle_rng.manual_seed(seed + trainer.epoch)
        train_loss = trainer.train_epoch(train_loader)
        valid_loss = trainer.evaluate(valid_loader)
        trainer.save_checkpoint(checkpoint_path)

        print(
            f"epoch={trainer.epoch}/{epochs} step={trainer.global_step} "
            f"train_loss={train_loss:.4f} val_loss={valid_loss:.4f}",
            flush=True,
        )

    print(f"checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/translation.yaml"))
    parser.add_argument("--resume", type=Path, help="从已有 checkpoint 继续训练")
    parser.add_argument("--epochs", type=int, help="总轮数，包含已经完成的轮数")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    try:
        run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

"""在测试集上生成文本并计算语料级 SacreBLEU。"""

import argparse
from pathlib import Path

import yaml
from sacrebleu.metrics import BLEU, CHRF
from train import load_pairs, read_config, select_device
from translate import generate_texts, load_model_bundle


def run(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    data = config["data"]

    model, src_tokenizer, tgt_tokenizer, info = load_model_bundle(
        args.checkpoint, select_device(args.device)
    )
    if info["task"] != data["task"]:
        raise ValueError("评估配置的任务必须与 checkpoint 一致")

    pairs = load_pairs(data, "test", info["seed"])
    limit = info["max_seq_len"]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else limit
    if args.batch_size < 1 or args.examples < 0:
        raise ValueError("batch_size 必须为正整数，examples 必须为非负整数")

    hypotheses = []
    references = [target for _, target in pairs]

    for start in range(0, len(pairs), args.batch_size):
        texts = [src for src, _ in pairs[start : start + args.batch_size]]
        hypotheses.extend(
            generate_texts(
                model,
                src_tokenizer,
                tgt_tokenizer,
                texts,
                max_seq_len=limit,
                max_new_tokens=max_new_tokens,
            )
        )

    bleu_tokenize = args.bleu_tokenize
    if bleu_tokenize is None:
        bleu_tokenize = "zh" if tgt_tokenizer.mode == "char" else "13a"
    metric = BLEU(tokenize=bleu_tokenize)
    score = metric.corpus_score(hypotheses, [references])

    print(f"task={data['task']} split=test samples={len(references)}")
    print(f"BLEU={score.score:.2f} ({metric.get_signature()})")

    if data["task"] == "copy":
        matches = sum(hyp == ref for hyp, ref in zip(hypotheses, references, strict=True))
        print(f"exact_match={matches}/{len(references)} ({matches / len(references):.1%})")
    else:
        chrf = CHRF()
        print(
            f"chrF={chrf.corpus_score(hypotheses, [references]).score:.2f} ({chrf.get_signature()})"
        )

    for (source, reference), hypothesis in zip(pairs[: args.examples], hypotheses):
        print(f"源句：{source}\n预测：{hypothesis}\n参考：{reference}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/translation.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--bleu-tokenize", help="覆盖 BLEU 的分词规则，中文默认 zh")
    parser.add_argument("--examples", type=int, default=0, help="显示前 N 个测试例子的译文")
    args = parser.parse_args()

    try:
        run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

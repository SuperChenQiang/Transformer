"""准备 Tatoeba 英译中语料，输出训练、验证和测试 TSV。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from random import Random
from urllib.request import Request, urlopen
from zipfile import ZipFile

from opencc import OpenCC

from transformer.data.tokenizer import WordTokenizer

URL = "https://www.manythings.org/anki/cmn-eng.zip"
SHA256 = "d606361aa7312dcdfb15ef4fc51cb1e7ed8bc2dc608a7b192626cd36e8403106"
DEFAULT_DIR = Path("data/tatoeba-en-zh")


def read_archive(path: Path, *, download: bool) -> bytes:
    if path.is_file():
        content = path.read_bytes()
    elif download:
        request = Request(
            URL, headers={"User-Agent": "Transformer-Translation-Tutorial/0.1", "Accept": "*/*"}
        )
        with urlopen(request, timeout=45) as response:
            content = response.read()
    else:
        raise FileNotFoundError(path)

    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("压缩包与 README 记录的语料版本不一致，请用 --archive 提供已验证的 ZIP")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return content


def extract_pairs(content: bytes) -> dict[tuple[str, ...], list[tuple[str, str, str, str]]]:
    """按归一化英文分组，保留每对句子的原文及署名。"""
    source_tokenizer = WordTokenizer(mode="en")
    target_tokenizer = WordTokenizer(mode="char", lowercase=False)
    simplify = OpenCC("t2s").convert
    groups: dict[tuple[str, ...], list[tuple[str, str, str, str]]] = {}
    seen = set()
    with ZipFile(io.BytesIO(content)) as archive:
        rows = archive.read("cmn.txt").decode("utf-8-sig").splitlines()

    for number, row in enumerate(rows, start=1):
        fields = row.split("\t")
        if len(fields) != 3 or not all(fields):
            raise ValueError(f"语料第 {number} 行缺少源句、译文或署名")
        source, original_target, attribution = fields
        target = simplify(original_target)
        key = tuple(source_tokenizer.tokenize(source))
        target_tokens = tuple(target_tokenizer.tokenize(target))
        if not (1 <= len(key) <= 10 and 1 <= len(target_tokens) <= 24):
            continue
        if not any("\u4e00" <= char <= "\u9fff" for char in target_tokens):
            continue
        pair_key = (key, target_tokens)
        if pair_key not in seen:
            seen.add(pair_key)
            groups.setdefault(key, []).append((source, target, original_target, attribution))
    return groups


def prepare(output_dir: Path, archive: Path | None, train_sources: int) -> None:
    if train_sources < 1:
        raise ValueError("--train-sources 必须至少为 1")
    content = read_archive(
        archive if archive is not None else output_dir / "cmn-eng.zip",
        download=archive is None,
    )
    groups = extract_pairs(content)
    keys = sorted(groups)
    Random(42).shuffle(keys)
    if len(keys) <= 1000:
        raise ValueError("有效源句不足，无法保留独立验证集和测试集")
    splits = {
        "train": keys[1000 : 1000 + train_sources],
        "validation": keys[:500],
        "test": keys[500:1000],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    with (output_dir / "attribution.tsv").open("w", encoding="utf-8") as credit_file:
        credit_file.write("split\tsource\toriginal_target\tattribution\n")
        for split, source_keys in splits.items():
            counts[split] = 0
            with (output_dir / f"{split}.tsv").open("w", encoding="utf-8") as data_file:
                for key in source_keys:
                    # 验证和测试每个源句留一个参考译文。
                    rows = groups[key] if split == "train" else groups[key][:1]
                    for source, target, original_target, attribution in rows:
                        data_file.write(f"{source}\t{target}\n")
                        credit_file.write(f"{split}\t{source}\t{original_target}\t{attribution}\n")
                        counts[split] += 1

    manifest = {
        "source": URL,
        "archive_sha256": SHA256,
        "split_seed": 42,
        "requested_train_sources": train_sources,
        "unique_sources": {split: len(source_keys) for split, source_keys in splits.items()},
        "pairs": counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{output_dir}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--archive", type=Path, help="使用本地已有的同版本压缩包")
    parser.add_argument("--train-sources", type=int, default=30000)
    args = parser.parse_args()
    try:
        prepare(args.output_dir, args.archive, args.train_sources)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

"""翻译命令入口骨架；等待模型加载、分词器和自回归解码实现。"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.parse_args()
    parser.error("翻译尚未实现。请先完成模型、checkpoint 加载和 decoding.py。")


if __name__ == "__main__":
    main()

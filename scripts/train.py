"""训练命令入口骨架；等待 tokenizer、dataset、model 和 trainer 实现。"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.yaml"))
    parser.add_argument("--resume", type=Path, help="后续支持：从 checkpoint 恢复训练")
    parser.parse_args()
    parser.error("训练尚未实现。请按 docs/implementation_order.md 填写模块。")


if __name__ == "__main__":
    main()

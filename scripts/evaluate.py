"""评估命令入口骨架；等待数据加载、生成和 SacreBLEU 评估实现。"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.parse_args()
    parser.error("评估尚未实现。后续必须记录数据划分、分词方式和指标 signature。")


if __name__ == "__main__":
    main()

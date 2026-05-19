"""Evaluation CLI."""

from __future__ import annotations

import argparse

from brats_ai.config import load_config
from brats_ai.data.dataset import discover_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect evaluation split availability.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    split = args.split or config.data.test_split
    records = discover_cases(config.data, split)
    missing_masks = sum(record.mask_path is None for record in records)
    print({"split": split, "cases": len(records), "missing_masks": missing_masks})


if __name__ == "__main__":
    main()


"""Training CLI."""

from __future__ import annotations

import argparse
import logging

from brats_ai.config import load_config, validate_config
from brats_ai.training.engine import run_dry_run, train
from brats_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BraTS segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging()
    config = load_config(args.config)
    for warning in validate_config(config):
        logging.warning(warning)
    if args.dry_run:
        print(run_dry_run(config))
        return
    print(f"best_checkpoint={train(config)}")


if __name__ == "__main__":
    main()


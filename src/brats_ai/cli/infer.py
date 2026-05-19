"""Inference CLI."""

from __future__ import annotations

import argparse

from brats_ai.config import load_config
from brats_ai.inference.predictor import SegmentationPredictor, save_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BraTS segmentation inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    predictor = SegmentationPredictor(config, args.checkpoint)
    output_path = save_prediction(predictor.predict_file(args.input), args.output)
    print(f"prediction={output_path}")


if __name__ == "__main__":
    main()


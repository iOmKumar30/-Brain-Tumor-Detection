"""Repository and config audit CLI."""

from __future__ import annotations

import argparse

from brats_ai.config import load_config, validate_config
from brats_ai.utils.dependencies import is_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit BraTS project configuration.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    warnings = validate_config(config)
    print(f"project={config.project.name}")
    print(f"data_root={config.data.root}")
    print(f"model={config.model.name}")
    print(
        "dependencies="
        + str({name: is_available(name) for name in ["torch", "monai", "numpy", "nibabel", "fastapi"]})
    )
    for warning in warnings:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()


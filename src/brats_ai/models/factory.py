"""Model factory."""

from __future__ import annotations

from brats_ai.config import DataConfig, ModelConfig
from brats_ai.models.transformers import build_monai_transformer
from brats_ai.models.unet3d import UNet3D


def build_model(model: ModelConfig, data: DataConfig):
    """Build a configured segmentation model."""

    name = model.name.lower()
    if name == "unet3d":
        return UNet3D(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            base_channels=model.base_channels,
            dropout=model.dropout,
        )
    if name in {"unetr", "swinunetr"}:
        return build_monai_transformer(model, data)
    raise ValueError(f"Unsupported model name: {model.name}")


"""MONAI transformer segmentation model builders."""

from __future__ import annotations

from brats_ai.config import DataConfig, ModelConfig
from brats_ai.exceptions import MissingDependencyError


def build_monai_transformer(model: ModelConfig, data: DataConfig):
    """Build UNETR or SwinUNETR from MONAI."""

    try:
        from monai.networks.nets import SwinUNETR, UNETR
    except ImportError as exc:
        raise MissingDependencyError(
            "Install MONAI and PyTorch to use transformer models: pip install monai torch"
        ) from exc

    name = model.name.lower()
    img_size = tuple(data.input_shape)
    if name == "unetr":
        return UNETR(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            img_size=img_size,
            feature_size=model.feature_size,
        )
    if name == "swinunetr":
        return SwinUNETR(
            in_channels=model.in_channels,
            out_channels=model.out_channels,
            img_size=img_size,
            feature_size=model.feature_size,
            use_checkpoint=model.use_checkpoint,
        )
    raise ValueError(f"Unsupported MONAI transformer model: {model.name}")


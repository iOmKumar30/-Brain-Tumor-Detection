"""Checkpoint loading and inference."""

from __future__ import annotations

from pathlib import Path

from brats_ai.config import ExperimentConfig
from brats_ai.data.preprocessing import ensure_channel_first, load_volume, zscore_nonzero
from brats_ai.exceptions import MissingDependencyError
from brats_ai.models import build_model
from brats_ai.utils.reproducibility import resolve_device


class SegmentationPredictor:
    """Reusable inference wrapper for CLI and API use."""

    def __init__(self, config: ExperimentConfig, checkpoint: str | Path | None = None):
        try:
            import torch
        except ImportError as exc:
            raise MissingDependencyError("Install PyTorch to run inference.") from exc

        self.torch = torch
        self.config = config
        self.device = torch.device(resolve_device(config.training.device))
        self.model = build_model(config.model, config.data).to(self.device)
        if checkpoint:
            payload = torch.load(checkpoint, map_location=self.device)
            state_dict = payload.get("model", payload)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_array(self, image):
        """Predict a label mask from an in-memory image volume."""

        image = ensure_channel_first(image, self.config.model.in_channels)
        image = zscore_nonzero(image)
        tensor = self.torch.as_tensor(image, dtype=self.torch.float32)[None].to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = self.torch.softmax(logits, dim=1)
            mask = probabilities.argmax(dim=1)
        return {
            "mask": mask.squeeze(0).detach().cpu().numpy(),
            "probabilities": probabilities.squeeze(0).detach().cpu().numpy(),
        }

    def predict_file(self, input_path: str | Path):
        """Predict from a `.npy`, `.npz`, or NIfTI file."""

        return self.predict_array(load_volume(input_path))


def save_prediction(prediction: dict[str, object], output_path: str | Path) -> Path:
    """Save prediction arrays as compressed NPZ."""

    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("Install numpy to save predictions.") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prediction)
    return path


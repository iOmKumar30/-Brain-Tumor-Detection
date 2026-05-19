"""MRI preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from brats_ai.exceptions import MissingDependencyError


def load_volume(path: str | Path) -> Any:
    """Load `.npy`, `.npz`, or NIfTI volumes."""

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("Install numpy and nibabel for medical volume loading.") from exc

    volume_path = Path(path)
    suffixes = "".join(volume_path.suffixes).lower()
    if volume_path.suffix == ".npy":
        return np.load(volume_path)
    if volume_path.suffix == ".npz":
        data = np.load(volume_path)
        key = "image" if "image" in data else data.files[0]
        return data[key]
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return nib.load(str(volume_path)).get_fdata(dtype=np.float32)
    raise ValueError(f"Unsupported volume format: {volume_path}")


def zscore_nonzero(volume: Any, eps: float = 1e-8) -> Any:
    """Z-score normalize nonzero MRI voxels channel-wise."""

    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("Install numpy for preprocessing.") from exc

    arr = np.asarray(volume, dtype=np.float32).copy()
    if arr.ndim == 3:
        mask = arr != 0
        if mask.any():
            arr[mask] = (arr[mask] - arr[mask].mean()) / (arr[mask].std() + eps)
        return arr

    channel_axis = 0 if arr.shape[0] <= 8 else -1
    if channel_axis == -1:
        arr = np.moveaxis(arr, -1, 0)
    for channel in range(arr.shape[0]):
        mask = arr[channel] != 0
        if mask.any():
            arr[channel][mask] = (arr[channel][mask] - arr[channel][mask].mean()) / (
                arr[channel][mask].std() + eps
            )
    return arr if channel_axis == 0 else np.moveaxis(arr, 0, -1)


def ensure_channel_first(volume: Any, expected_channels: int = 4) -> Any:
    """Return image data in `[C, H, W, D]` layout."""

    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("Install numpy for preprocessing.") from exc

    arr = np.asarray(volume)
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected 3D or 4D volume, got shape {arr.shape}.")
    if arr.shape[0] == expected_channels:
        return arr
    if arr.shape[-1] == expected_channels:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Cannot infer channel axis for shape {arr.shape}.")


def one_hot_to_label(mask: Any) -> Any:
    """Convert one-hot mask to integer labels when needed."""

    try:
        import numpy as np
    except ImportError as exc:
        raise MissingDependencyError("Install numpy for preprocessing.") from exc

    arr = np.asarray(mask)
    if arr.ndim == 4 and arr.shape[-1] <= 8:
        return np.argmax(arr, axis=-1).astype(np.int64)
    if arr.ndim == 4 and arr.shape[0] <= 8:
        return np.argmax(arr, axis=0).astype(np.int64)
    return arr.astype(np.int64)


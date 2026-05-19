"""BraTS dataset indexing and PyTorch dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brats_ai.config import DataConfig
from brats_ai.data.preprocessing import ensure_channel_first, load_volume, one_hot_to_label, zscore_nonzero
from brats_ai.exceptions import DatasetError, MissingDependencyError


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """A single image/mask pair."""

    case_id: str
    image_path: Path
    mask_path: Path | None = None


def discover_cases(data: DataConfig, split: str) -> list[CaseRecord]:
    """Discover cases from `root/split/images` and optional `root/split/masks`."""

    split_root = Path(data.root) / split
    image_dir = split_root / data.image_dir_name
    mask_dir = split_root / data.mask_dir_name
    if not image_dir.exists():
        raise DatasetError(f"Image directory does not exist: {image_dir}")

    image_files = sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix in {".npy", ".npz", ".nii"} or "".join(p.suffixes).endswith(".nii.gz")
    )
    if not image_files:
        raise DatasetError(f"No image volumes found in {image_dir}")

    records: list[CaseRecord] = []
    for image_path in image_files:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            stem_mask = mask_dir / f"{image_path.stem}.npy"
            mask_path = stem_mask if stem_mask.exists() else None
        records.append(CaseRecord(case_id=image_path.stem, image_path=image_path, mask_path=mask_path))
    return records


class BraTSDataset:
    """PyTorch-compatible dataset for preprocessed BraTS volumes."""

    def __init__(self, records: list[CaseRecord], expected_channels: int = 4, require_masks: bool = True):
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise MissingDependencyError("Install PyTorch to use BraTSDataset.") from exc

        if not issubclass(type(self), Dataset):
            self._torch = torch
        self.records = records
        self.expected_channels = expected_channels
        self.require_masks = require_masks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = ensure_channel_first(load_volume(record.image_path), self.expected_channels)
        image = zscore_nonzero(image)

        sample: dict[str, object] = {
            "case_id": record.case_id,
            "image": self._torch.as_tensor(image, dtype=self._torch.float32),
        }
        if record.mask_path is None:
            if self.require_masks:
                raise DatasetError(f"Missing mask for case {record.case_id}")
            return sample

        mask = one_hot_to_label(load_volume(record.mask_path))
        sample["mask"] = self._torch.as_tensor(mask, dtype=self._torch.long)
        return sample


def make_dataloader(records: list[CaseRecord], batch_size: int, expected_channels: int, shuffle: bool):
    """Build a PyTorch dataloader for discovered cases."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise MissingDependencyError("Install PyTorch to create dataloaders.") from exc

    dataset = BraTSDataset(records, expected_channels=expected_channels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

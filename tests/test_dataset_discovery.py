from brats_ai.config import DataConfig
from brats_ai.data.dataset import discover_cases


def test_discover_cases_matches_images_and_masks(tmp_path):
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "case_001.npy").write_bytes(b"placeholder")
    (mask_dir / "case_001.npy").write_bytes(b"placeholder")

    records = discover_cases(DataConfig(root=tmp_path), "train")

    assert len(records) == 1
    assert records[0].case_id == "case_001"
    assert records[0].mask_path is not None


def test_discover_cases_matches_image_mask_prefixes(tmp_path):
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "image_296.npy").write_bytes(b"placeholder")
    (mask_dir / "mask_296.npy").write_bytes(b"placeholder")

    records = discover_cases(DataConfig(root=tmp_path), "train")

    assert len(records) == 1
    assert records[0].mask_path == mask_dir / "mask_296.npy"

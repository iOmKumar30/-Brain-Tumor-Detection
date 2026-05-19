# Dataset Preparation

## Supported Layout

Place preprocessed BraTS volumes under `data/processed`:

```text
data/processed/
├── train/
│   ├── images/
│   │   ├── case_000.npy
│   │   └── case_001.npy
│   └── masks/
│       ├── case_000.npy
│       └── case_001.npy
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Images should be four-channel MRI volumes in either `[4, H, W, D]` or `[H, W, D, 4]` layout. Masks may be integer label maps `[H, W, D]` or one-hot masks.

## Labels

The default config assumes four classes:

- `0`: background
- `1`: tumor subregion 1
- `2`: tumor subregion 2
- `3`: tumor subregion 3

Adapt `data.num_classes` and `model.out_channels` if your label encoding differs.

## Preprocessing Rules

- Channel-first conversion is applied automatically.
- Nonzero MRI voxels are z-score normalized per channel.
- Masks are converted from one-hot to integer labels when needed.
- Data folders are git-ignored to prevent committing protected or large medical data.

## Leakage Prevention

Split at patient/case level before augmentation. Do not slice one patient into multiple splits. Keep all modalities and masks for a case in the same split.


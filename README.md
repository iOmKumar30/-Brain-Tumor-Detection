# Brain Tumor Detection and Segmentation

Production-ready BraTS-style brain tumor segmentation scaffold with reproducible training, MONAI transformer support, evaluation metrics, FastAPI inference, and a React web interface.

> Medical safety note: this project is for research, education, and portfolio demonstration. It is not a certified medical device and must not be used for clinical decisions without regulatory validation.

## What This Repository Contains

- Modular Python package under `src/brats_ai`
- BraTS `.npy` and NIfTI dataset loaders
- 3D UNet baseline and MONAI UNETR/SwinUNETR builders
- Deterministic training, validation, inference, and evaluation CLIs
- Dice, IoU, precision, recall/sensitivity, specificity, and Hausdorff metrics
- FastAPI backend for upload and inference orchestration
- React + TypeScript frontend for MRI upload and mask visualization
- Docker and docker-compose deployment setup
- Clean notebooks that call the package instead of hardcoded Kaggle paths
- Tests for config, metrics, and dataset split behavior

## Repository Layout

```text
.
├── api/                         # FastAPI app entrypoint
├── configs/                     # YAML experiment configs
├── data/                        # Dataset mount point, ignored by git
├── docs/                        # Architecture, dataset, deployment, audit notes
├── notebooks/                   # Reproducible notebook workflows
├── src/brats_ai/                # Core Python package
├── tests/                       # Unit and smoke tests
├── webapp/                      # React + TypeScript frontend
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Quick Start

Create an environment with Python 3.10 or 3.11. PyTorch/MONAI medical imaging stacks are not yet broadly reliable on Python 3.13, so use a supported ML runtime.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Prepare data using the layout in [docs/DATASET.md](docs/DATASET.md), then run a smoke check:

```bash
python -m brats_ai.cli.audit --config configs/default.yaml
python -m brats_ai.cli.train --config configs/default.yaml --dry-run
```

Train a baseline:

```bash
python -m brats_ai.cli.train --config configs/default.yaml
```

Train a transformer model:

```bash
python -m brats_ai.cli.train --config configs/transformer_swinunetr.yaml
```

Run inference:

```bash
python -m brats_ai.cli.infer \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --input data/processed/test/case_000/images.npy \
  --output outputs/case_000_prediction.npz
```

Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start the web UI:

```bash
cd webapp
npm install
npm run dev
```

## Model Options

| Model | Backend | Purpose |
| --- | --- | --- |
| `unet3d` | PyTorch | Lightweight baseline for reproducible experiments |
| `unetr` | MONAI | Transformer encoder for volumetric segmentation |
| `swinunetr` | MONAI | Windowed transformer architecture for high-quality BraTS segmentation |

## Validation Status

This refactor was validated in the current Codex environment with syntax checks. Full notebook execution, GPU training, and real BraTS benchmark generation require:

- BraTS dataset files
- PyTorch + MONAI environment
- GPU/CUDA runtime for practical training
- Git LFS checkpoint download via `git lfs pull`

See [docs/AUDIT.md](docs/AUDIT.md) for the discovered issues and remediation map.


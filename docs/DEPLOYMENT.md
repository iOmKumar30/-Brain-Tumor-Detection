# Deployment Guide

## Local API

```bash
pip install -r requirements.txt
export BRATS_CONFIG=configs/default.yaml
export BRATS_CHECKPOINT=checkpoints/best.pt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell:

```powershell
$env:BRATS_CONFIG="configs/default.yaml"
$env:BRATS_CHECKPOINT="checkpoints/best.pt"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker Compose

```bash
docker compose up --build
```

API: `http://localhost:8000`

Web UI: `http://localhost:5173`

## GPU Inference

Use a CUDA PyTorch base image and set:

```text
BRATS_DEVICE=cuda
```

Mount checkpoints read-only and keep uploads/outputs on persistent storage.

## Cloud Notes

- Vercel is suitable for the frontend only.
- Render can host the FastAPI CPU service for demos.
- Hugging Face Spaces can host a CPU demo or GPU-backed application.
- Production medical inference should use a GPU server with request limits, audit logs, encrypted storage, and PHI handling controls.


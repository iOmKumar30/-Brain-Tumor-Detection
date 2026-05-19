"""FastAPI inference service."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from brats_ai.config import load_config
from brats_ai.inference.predictor import SegmentationPredictor, save_prediction

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API dependencies with `pip install -e .[api]`.") from exc


CONFIG_PATH = os.getenv("BRATS_CONFIG", "configs/default.yaml")
CHECKPOINT_PATH = os.getenv("BRATS_CHECKPOINT")
MAX_UPLOAD_MB = int(os.getenv("BRATS_MAX_UPLOAD_MB", "512"))

app = FastAPI(
    title="Brain Tumor Segmentation API",
    version="0.1.0",
    description="BraTS-style tumor segmentation inference service.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: SegmentationPredictor | None = None


@app.on_event("startup")
def startup() -> None:
    """Load model once at service startup when a checkpoint is configured."""

    global _predictor
    if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
        config = load_config(CONFIG_PATH)
        _predictor = SegmentationPredictor(config, CHECKPOINT_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "config": CONFIG_PATH,
        "checkpoint_loaded": _predictor is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> FileResponse:
    """Upload a `.npy`, `.npz`, `.nii`, or `.nii.gz` volume and receive an NPZ prediction."""

    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not loaded. Set BRATS_CHECKPOINT to enable inference.",
        )
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds configured size limit.")

    suffix = "".join(Path(file.filename or "upload.npy").suffixes) or ".npy"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        output_path = Path(tmpdir) / "prediction.npz"
        input_path.write_bytes(payload)
        prediction = _predictor.predict_file(input_path)
        save_prediction(prediction, output_path)
        return FileResponse(output_path, filename="prediction.npz", media_type="application/octet-stream")


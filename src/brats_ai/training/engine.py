"""Training and validation loops."""

from __future__ import annotations

import logging
from pathlib import Path

from brats_ai.config import ExperimentConfig
from brats_ai.data.dataset import discover_cases, make_dataloader
from brats_ai.evaluation.metrics import mean_multiclass_metrics
from brats_ai.exceptions import MissingDependencyError
from brats_ai.models import build_model
from brats_ai.training.losses import DiceCrossEntropyLoss
from brats_ai.utils.reproducibility import resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError("Install PyTorch to train models.") from exc
    return torch


def run_dry_run(config: ExperimentConfig) -> dict[str, object]:
    """Validate config, dataset availability, and model instantiation without training."""

    seed_everything(config.project.seed, config.training.deterministic)
    torch = _require_torch()
    device = resolve_device(config.training.device)
    model = build_model(config.model, config.data)
    param_count = sum(p.numel() for p in model.parameters())
    dataset_status: dict[str, int | str] = {}
    for split in (config.data.train_split, config.data.val_split, config.data.test_split):
        try:
            records = discover_cases(config.data, split)
            missing_masks = sum(record.mask_path is None for record in records)
            dataset_status[split] = (
                len(records) if missing_masks == 0 else f"{len(records)} cases, {missing_masks} missing masks"
            )
        except Exception as exc:  # noqa: BLE001 - report non-fatal audit status
            dataset_status[split] = str(exc)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"device": device, "parameters": param_count, "dataset": dataset_status}


def train(config: ExperimentConfig) -> Path:
    """Train a segmentation model and return the best checkpoint path."""

    torch = _require_torch()
    seed_everything(config.project.seed, config.training.deterministic)
    device = torch.device(resolve_device(config.training.device))
    output_dir = Path(config.project.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_records = discover_cases(config.data, config.data.train_split)
    val_records = discover_cases(config.data, config.data.val_split)
    train_loader = make_dataloader(
        train_records, config.training.batch_size, config.model.in_channels, shuffle=True
    )
    val_loader = make_dataloader(val_records, config.training.batch_size, config.model.in_channels, shuffle=False)

    model = build_model(config.model, config.data).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    loss_fn = DiceCrossEntropyLoss(include_background=config.evaluation.include_background)
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision and device.type == "cuda")

    best_metric = float("-inf") if config.training.checkpoint_mode == "max" else float("inf")
    best_path = checkpoint_dir / "best.pt"
    patience_remaining = config.training.early_stopping_patience

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=config.training.mixed_precision and device.type == "cuda"):
                logits = model(image)
                loss = loss_fn(logits, mask) / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % config.training.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu()) * config.training.gradient_accumulation_steps

        scheduler.step()
        val_metrics = validate(model, val_loader, device, config)
        metric = val_metrics.dice
        improved = metric > best_metric if config.training.checkpoint_mode == "max" else metric < best_metric
        LOGGER.info(
            "epoch=%s train_loss=%.4f val_dice=%.4f val_iou=%.4f",
            epoch,
            running_loss / max(1, len(train_loader)),
            val_metrics.dice,
            val_metrics.iou,
        )
        if improved:
            best_metric = metric
            patience_remaining = config.training.early_stopping_patience
            torch.save({"model": model.state_dict(), "config": config, "metric": best_metric}, best_path)
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                LOGGER.info("Early stopping at epoch %s", epoch)
                break
    return best_path


def validate(model, dataloader, device, config: ExperimentConfig):
    """Validate a model and return averaged segmentation metrics."""

    torch = _require_torch()
    model.eval()
    metrics = []
    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(image)
            metrics.append(
                mean_multiclass_metrics(
                    logits,
                    mask,
                    num_classes=config.data.num_classes,
                    include_background=config.evaluation.include_background,
                )
            )
    if not metrics:
        raise RuntimeError("Validation dataloader produced no batches.")
    return type(metrics[0])(
        dice=sum(m.dice for m in metrics) / len(metrics),
        iou=sum(m.iou for m in metrics) / len(metrics),
        precision=sum(m.precision for m in metrics) / len(metrics),
        recall=sum(m.recall for m in metrics) / len(metrics),
        specificity=sum(m.specificity for m in metrics) / len(metrics),
    )

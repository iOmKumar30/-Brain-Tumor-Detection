"""Segmentation metrics with multiclass-safe semantics."""

from __future__ import annotations

from dataclasses import dataclass

from brats_ai.exceptions import MissingDependencyError


@dataclass(frozen=True, slots=True)
class MetricResult:
    dice: float
    iou: float
    precision: float
    recall: float
    specificity: float
    hausdorff95: float | None = None


def _as_label_tensor(prediction, target):
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError("Install PyTorch for metric computation.") from exc

    pred = prediction
    true = target
    if pred.ndim == true.ndim + 1:
        pred = pred.argmax(dim=1)
    if true.ndim == pred.ndim + 1:
        true = true.argmax(dim=1)
    return pred.long(), true.long(), torch


def binary_stats(prediction, target, class_index: int):
    """Return TP/FP/FN/TN for a single class."""

    pred, true, torch = _as_label_tensor(prediction, target)
    pred_c = pred == class_index
    true_c = true == class_index
    tp = torch.logical_and(pred_c, true_c).sum().float()
    fp = torch.logical_and(pred_c, ~true_c).sum().float()
    fn = torch.logical_and(~pred_c, true_c).sum().float()
    tn = torch.logical_and(~pred_c, ~true_c).sum().float()
    return tp, fp, fn, tn


def dice_score(prediction, target, class_index: int, eps: float = 1e-8) -> float:
    tp, fp, fn, _ = binary_stats(prediction, target, class_index)
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def iou_score(prediction, target, class_index: int, eps: float = 1e-8) -> float:
    tp, fp, fn, _ = binary_stats(prediction, target, class_index)
    return float((tp + eps) / (tp + fp + fn + eps))


def precision_score(prediction, target, class_index: int, eps: float = 1e-8) -> float:
    tp, fp, _, _ = binary_stats(prediction, target, class_index)
    return float((tp + eps) / (tp + fp + eps))


def recall_score(prediction, target, class_index: int, eps: float = 1e-8) -> float:
    tp, _, fn, _ = binary_stats(prediction, target, class_index)
    return float((tp + eps) / (tp + fn + eps))


def specificity_score(prediction, target, class_index: int, eps: float = 1e-8) -> float:
    _, fp, _, tn = binary_stats(prediction, target, class_index)
    return float((tn + eps) / (tn + fp + eps))


def mean_multiclass_metrics(prediction, target, num_classes: int, include_background: bool = False) -> MetricResult:
    """Average class-wise metrics, excluding background by default."""

    start = 0 if include_background else 1
    classes = list(range(start, num_classes))
    if not classes:
        classes = [0]
    dice = [dice_score(prediction, target, c) for c in classes]
    iou = [iou_score(prediction, target, c) for c in classes]
    precision = [precision_score(prediction, target, c) for c in classes]
    recall = [recall_score(prediction, target, c) for c in classes]
    specificity = [specificity_score(prediction, target, c) for c in classes]
    return MetricResult(
        dice=sum(dice) / len(dice),
        iou=sum(iou) / len(iou),
        precision=sum(precision) / len(precision),
        recall=sum(recall) / len(recall),
        specificity=sum(specificity) / len(specificity),
    )


def hausdorff95(prediction, target, class_index: int) -> float:
    """Compute 95th percentile Hausdorff distance for one class."""

    try:
        import numpy as np
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise MissingDependencyError("Install numpy and scipy for Hausdorff distance.") from exc

    pred, true, _ = _as_label_tensor(prediction, target)
    pred_points = np.argwhere((pred.detach().cpu().numpy() == class_index))
    true_points = np.argwhere((true.detach().cpu().numpy() == class_index))
    if len(pred_points) == 0 or len(true_points) == 0:
        return float("inf")
    distances = cdist(pred_points, true_points)
    surface_distances = np.concatenate([distances.min(axis=1), distances.min(axis=0)])
    return float(np.percentile(surface_distances, 95))

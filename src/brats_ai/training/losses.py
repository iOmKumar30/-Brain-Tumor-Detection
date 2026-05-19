"""Loss functions."""

from __future__ import annotations

from brats_ai.exceptions import MissingDependencyError


class DiceCrossEntropyLoss:
    """Combined Dice and cross-entropy loss for multiclass segmentation."""

    def __init__(self, include_background: bool = False, smooth: float = 1e-5, dice_weight: float = 0.5):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise MissingDependencyError("Install PyTorch to use training losses.") from exc

        self.torch = torch
        self.functional = functional
        self.include_background = include_background
        self.smooth = smooth
        self.dice_weight = dice_weight

    def __call__(self, logits, target):
        num_classes = logits.shape[1]
        probs = self.torch.softmax(logits, dim=1)
        one_hot = self.functional.one_hot(target.long(), num_classes=num_classes)
        one_hot = one_hot.movedim(-1, 1).float()
        start = 0 if self.include_background else 1
        dims = tuple(range(2, logits.ndim))
        intersection = (probs[:, start:] * one_hot[:, start:]).sum(dim=dims)
        denominator = probs[:, start:].sum(dim=dims) + one_hot[:, start:].sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice.mean()
        ce_loss = self.functional.cross_entropy(logits, target.long())
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss


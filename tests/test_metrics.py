import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")


def test_perfect_prediction_has_unit_dice():
    import torch

    from brats_ai.evaluation.metrics import dice_score, mean_multiclass_metrics

    target = torch.tensor([[[[0, 1], [2, 3]]]])
    prediction = target.clone()

    assert dice_score(prediction, target, class_index=1) == 1.0
    assert mean_multiclass_metrics(prediction, target, num_classes=4).dice == 1.0


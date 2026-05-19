# Benchmark Plan

## Required Metrics

| Metric | Implementation |
| --- | --- |
| Dice | `brats_ai.evaluation.metrics.dice_score` |
| IoU | `brats_ai.evaluation.metrics.iou_score` |
| Precision | `precision_score` |
| Recall / Sensitivity | `recall_score` |
| Specificity | `specificity_score` |
| Hausdorff95 | `hausdorff95` |

## Model Comparison Template

| Model | Params | Dice | IoU | HD95 | VRAM | Inference Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3D UNet | Run training | TBD | TBD | TBD | TBD | TBD |
| UNETR | Run training | TBD | TBD | TBD | TBD | TBD |
| SwinUNETR | Run training | TBD | TBD | TBD | TBD | TBD |

No real scores are claimed until the BraTS dataset and GPU training run are available.


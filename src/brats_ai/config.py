"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brats_ai.exceptions import ConfigurationError


@dataclass(slots=True)
class ProjectConfig:
    name: str = "brats-segmentation"
    seed: int = 42
    output_dir: Path = Path("artifacts/default")


@dataclass(slots=True)
class DataConfig:
    root: Path = Path("data/processed")
    format: str = "npy"
    image_dir_name: str = "images"
    mask_dir_name: str = "masks"
    modalities: list[str] = field(default_factory=lambda: ["flair", "t1", "t1ce", "t2"])
    num_classes: int = 4
    input_shape: tuple[int, int, int] = (128, 128, 128)
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    cache_rate: float = 0.0
    num_workers: int = 4


@dataclass(slots=True)
class ModelConfig:
    name: str = "unet3d"
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 24
    dropout: float = 0.1
    feature_size: int = 48
    use_checkpoint: bool = True


@dataclass(slots=True)
class TrainingConfig:
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    deterministic: bool = True
    early_stopping_patience: int = 20
    checkpoint_metric: str = "val_dice"
    checkpoint_mode: str = "max"
    log_every_n_steps: int = 10


@dataclass(slots=True)
class EvaluationConfig:
    include_background: bool = False
    hausdorff_percentile: int = 95
    threshold: float = 0.5


@dataclass(slots=True)
class TrackingConfig:
    tensorboard: bool = True
    wandb: bool = False
    experiment_name: str = "unet3d-baseline"


@dataclass(slots=True)
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - exercised in minimal envs
        return _load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config {path} must contain a YAML mapping.")
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small YAML subset parser for dependency-free audits."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ConfigurationError(f"Cannot parse config line: {raw_line}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _coerce_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(values) - field_names
    if unknown:
        raise ConfigurationError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")

    coerced = dict(values)
    if cls in {ProjectConfig, DataConfig}:
        for key in ("output_dir", "root"):
            if key in coerced:
                coerced[key] = Path(coerced[key])
    if cls is DataConfig and "input_shape" in coerced:
        coerced["input_shape"] = tuple(int(v) for v in coerced["input_shape"])
    return cls(**coerced)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into typed dataclasses."""

    config_path = Path(path)
    raw = _load_yaml(config_path)
    return ExperimentConfig(
        project=_coerce_dataclass(ProjectConfig, raw.get("project", {})),
        data=_coerce_dataclass(DataConfig, raw.get("data", {})),
        model=_coerce_dataclass(ModelConfig, raw.get("model", {})),
        training=_coerce_dataclass(TrainingConfig, raw.get("training", {})),
        evaluation=_coerce_dataclass(EvaluationConfig, raw.get("evaluation", {})),
        tracking=_coerce_dataclass(TrackingConfig, raw.get("tracking", {})),
    )


def validate_config(config: ExperimentConfig) -> list[str]:
    """Return human-readable warnings for non-fatal config issues."""

    warnings: list[str] = []
    if config.data.num_classes < 2:
        raise ConfigurationError("data.num_classes must be at least 2.")
    if config.model.in_channels != len(config.data.modalities):
        warnings.append(
            "model.in_channels differs from number of data.modalities; verify channel order."
        )
    if config.model.out_channels != config.data.num_classes:
        warnings.append("model.out_channels differs from data.num_classes; verify labels.")
    if config.training.batch_size < 1:
        raise ConfigurationError("training.batch_size must be positive.")
    if config.training.checkpoint_mode not in {"min", "max"}:
        raise ConfigurationError("training.checkpoint_mode must be 'min' or 'max'.")
    return warnings

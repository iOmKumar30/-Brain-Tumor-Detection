from brats_ai.config import DataConfig, ExperimentConfig, ModelConfig, validate_config


def test_default_config_validates_without_warnings():
    config = ExperimentConfig()

    assert validate_config(config) == []


def test_channel_mismatch_returns_warning():
    config = ExperimentConfig(model=ModelConfig(in_channels=1), data=DataConfig())

    warnings = validate_config(config)

    assert any("in_channels" in warning for warning in warnings)


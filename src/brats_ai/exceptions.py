"""Project-specific exceptions."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional ML dependency is required but unavailable."""


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


class DatasetError(RuntimeError):
    """Raised when dataset files or shapes are invalid."""


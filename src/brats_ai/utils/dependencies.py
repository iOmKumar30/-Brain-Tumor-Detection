"""Optional dependency helpers."""

from __future__ import annotations

from importlib.util import find_spec


def is_available(module_name: str) -> bool:
    """Return whether a Python module can be imported."""

    return find_spec(module_name) is not None


def require(module_name: str, install_hint: str) -> None:
    """Raise a clear error if an optional dependency is missing."""

    if not is_available(module_name):
        from brats_ai.exceptions import MissingDependencyError

        raise MissingDependencyError(f"Missing dependency '{module_name}'. {install_hint}")


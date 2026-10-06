"""Module: storage
Stage: Library
Author: KafuuChino
Date: 2026-10-06
Description: Canonical project root and storage path resolution utilities.
"""

import os
from pathlib import Path
from typing import Optional, Union


def get_project_root() -> Path:
    """Get the canonical root directory of the project repository.

    Ascends from the location of this module to locate the project root directory
    containing ``pyproject.toml`` or ``src/agilab_lib``. Can also be overridden
    by the ``AGILAB_PROJECT_ROOT`` environment variable.

    Returns:
        Absolute Path to the project root directory.

    Raises:
        RuntimeError: If the project root cannot be determined.
    """
    env_root = os.environ.get("AGILAB_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "agilab_lib"
        ).is_dir():
            return parent

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file() or (
            parent / "src" / "agilab_lib"
        ).is_dir():
            return parent

    if len(current.parents) >= 3:
        return current.parents[3]

    raise RuntimeError("Could not determine project root directory.")


def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolve a relative or absolute path canonically against the project root.

    If the provided path is absolute, it is resolved and returned directly.
    If the path is relative and starts with ``Visual_Navigation_System``, that prefix
    is stripped to prevent redundant nesting within the project tree. Relative paths
    are anchored deterministically to ``get_project_root()``.

    Args:
        path: Path string or Path object to resolve.

    Returns:
        Canonical absolute Path anchored to the project root.
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    parts = list(p.parts)
    while parts and parts[0] == "Visual_Navigation_System":
        parts.pop(0)

    clean_path = Path(*parts) if parts else Path(".")
    return (get_project_root() / clean_path).resolve()


def get_outputs_dir(subdir: Optional[Union[str, Path]] = None) -> Path:
    """Get the canonical outputs directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory path or name inside outputs.

    Returns:
        Absolute Path to the canonical outputs directory or subdirectory.
    """
    base = resolve_project_path("outputs")
    return (base / subdir).resolve() if subdir else base


def get_checkpoints_dir(subdir: Optional[Union[str, Path]] = None) -> Path:
    """Get the canonical checkpoints directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory path or name inside checkpoints.

    Returns:
        Absolute Path to the canonical checkpoints directory or subdirectory.
    """
    base = resolve_project_path("checkpoints")
    return (base / subdir).resolve() if subdir else base


def get_data_dir(subdir: Optional[Union[str, Path]] = None) -> Path:
    """Get the canonical data directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory path or name inside data.

    Returns:
        Absolute Path to the canonical data directory or subdirectory.
    """
    base = resolve_project_path("data")
    return (base / subdir).resolve() if subdir else base


def get_cache_dir(subdir: Optional[Union[str, Path]] = None) -> Path:
    """Get the canonical cache directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory path or name inside cache.

    Returns:
        Absolute Path to the canonical cache directory or subdirectory.
    """
    base = resolve_project_path(".cache")
    return (base / subdir).resolve() if subdir else base

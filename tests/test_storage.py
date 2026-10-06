"""Module: test_storage
Stage: Tests
Author: KafuuChino
Date: 2026-10-06
Description: Unit tests for canonical project root and storage path resolution seam.
"""

from pathlib import Path

import pytest
from agilab_lib.datasets.dual_source_dataset import DualSourceVideoDataset
from agilab_lib.utils.storage import (
    get_cache_dir,
    get_checkpoints_dir,
    get_data_dir,
    get_outputs_dir,
    get_project_root,
    resolve_project_path,
)


def test_get_project_root_finds_valid_directory() -> None:
    """Verify get_project_root identifies an existing directory with project markers."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.is_absolute()
    assert root.is_dir()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "agilab_lib").is_dir()


def test_get_project_root_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify AGILAB_PROJECT_ROOT environment variable overrides root discovery.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    custom_root = tmp_path / "mock_project"
    custom_root.mkdir()
    monkeypatch.setenv("AGILAB_PROJECT_ROOT", str(custom_root))

    assert get_project_root() == custom_root.resolve()


def test_resolve_project_path_absolute(tmp_path: Path) -> None:
    """Verify resolve_project_path preserves and resolves absolute paths.

    Args:
        tmp_path: Temporary directory fixture.
    """
    abs_dir = tmp_path / "custom_data"
    abs_dir.mkdir()

    resolved_path = resolve_project_path(abs_dir)
    assert resolved_path == abs_dir.resolve()
    assert resolved_path.is_absolute()

    resolved_str = resolve_project_path(str(abs_dir))
    assert resolved_str == abs_dir.resolve()


def test_resolve_project_path_relative() -> None:
    """Verify resolve_project_path anchors relative paths to project root."""
    root = get_project_root()

    resolved_data = resolve_project_path("data/one_path")
    assert resolved_data == (root / "data" / "one_path").resolve()

    resolved_outputs = resolve_project_path(Path("outputs/runs"))
    assert resolved_outputs == (root / "outputs" / "runs").resolve()

    resolved_empty = resolve_project_path("")
    assert resolved_empty == root.resolve()

    resolved_dot = resolve_project_path(".")
    assert resolved_dot == root.resolve()


def test_resolve_project_path_strips_project_prefix() -> None:
    """Verify resolve_project_path strips redundant Visual_Navigation_System prefix."""
    root = get_project_root()

    resolved_root = resolve_project_path("Visual_Navigation_System")
    assert resolved_root == root.resolve()

    resolved_root_path = resolve_project_path(Path("Visual_Navigation_System"))
    assert resolved_root_path == root.resolve()

    resolved_data = resolve_project_path("Visual_Navigation_System/data/one_path")
    assert resolved_data == (root / "data" / "one_path").resolve()

    resolved_checkpoints = resolve_project_path(
        Path("Visual_Navigation_System/checkpoints/model.pt")
    )
    assert resolved_checkpoints == (root / "checkpoints" / "model.pt").resolve()


def test_resolve_project_path_invariant_to_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify path resolution is invariant to changes in current working directory.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    parent_dir = root.parent

    # Test when cwd is a temporary directory
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path.resolve()
    assert resolve_project_path("data/360") == (root / "data" / "360").resolve()
    assert (
        resolve_project_path("Visual_Navigation_System/outputs")
        == (root / "outputs").resolve()
    )

    # Test when cwd is the parent workspace directory
    monkeypatch.chdir(parent_dir)
    assert Path.cwd() == parent_dir.resolve()
    assert resolve_project_path("data/360") == (root / "data" / "360").resolve()
    assert (
        resolve_project_path("Visual_Navigation_System/data/360")
        == (root / "data" / "360").resolve()
    )


def test_convenience_directory_helpers() -> None:
    """Verify outputs, checkpoints, data, and cache directory helpers."""
    root = get_project_root()

    assert get_outputs_dir() == (root / "outputs").resolve()
    assert get_outputs_dir("eval") == (root / "outputs" / "eval").resolve()
    assert (
        get_outputs_dir(Path("eval/run1"))
        == (root / "outputs" / "eval" / "run1").resolve()
    )

    assert get_checkpoints_dir() == (root / "checkpoints").resolve()
    assert get_checkpoints_dir("vae") == (root / "checkpoints" / "vae").resolve()

    assert get_data_dir() == (root / "data").resolve()
    assert get_data_dir("one_path") == (root / "data" / "one_path").resolve()

    assert get_cache_dir() == (root / ".cache").resolve()
    assert get_cache_dir("latents") == (root / ".cache" / "latents").resolve()


def test_dual_source_dataset_resolve_dir_delegation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify DualSourceVideoDataset._resolve_dir delegates to canonical resolution.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()

    # Relative paths resolve to project root
    assert (
        DualSourceVideoDataset._resolve_dir("data/one_path")
        == (root / "data" / "one_path").resolve()
    )
    assert (
        DualSourceVideoDataset._resolve_dir("Visual_Navigation_System/data/360")
        == (root / "data" / "360").resolve()
    )

    # Absolute paths are preserved
    abs_dir = tmp_path / "videos"
    abs_dir.mkdir()
    assert DualSourceVideoDataset._resolve_dir(abs_dir) == abs_dir.resolve()

    # Resolution is invariant to cwd changes
    monkeypatch.chdir(tmp_path)
    assert (
        DualSourceVideoDataset._resolve_dir("data/one_path")
        == (root / "data" / "one_path").resolve()
    )

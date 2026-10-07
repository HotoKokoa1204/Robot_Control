"""Module: test_storage_confinement
Stage: Tests
Author: KafuuChino
Date: 2026-10-06
Description: End-to-end integration tests for storage seam isolation,
    directory confinement, and clean git status verification.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    get_cache_dir,
    get_checkpoints_dir,
    get_data_dir,
    get_outputs_dir,
    get_project_root,
    resolve_project_path,
)
from scripts.extract_keyframes import extract_keyframes
from scripts.generate_video import generate_video
from scripts.reconstruct_video import reconstruct_video


def _snapshot_directory_state(directory: Path) -> Dict[str, Tuple[int, str]]:
    """Capture snapshot of directory file paths, sizes, and content hashes.

    Args:
        directory: Root path of the directory to snapshot.

    Returns:
        Mapping of relative path string to (file_size, sha256_hash).
    """
    if not directory.exists():
        return {}
    snapshot: Dict[str, Tuple[int, str]] = {}
    for p in directory.rglob("*"):
        if p.is_file():
            rel_str = p.relative_to(directory).as_posix()
            content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            snapshot[rel_str] = (p.stat().st_size, content_hash)
    return snapshot


def _is_relative_to(path: Path, base: Path) -> bool:
    """Check whether path is relative to or inside base directory.

    Args:
        path: Path to test.
        base: Target parent directory.

    Returns:
        True if path is inside base, False otherwise.
    """
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _get_main_repo_root() -> Path:
    """Derive canonical repository root, handling worktree checkouts dynamically.

    Returns:
        Path to main repository root.
    """
    root = get_project_root()
    if root.parent.name in (".worktrees", "worktrees"):
        return root.parent.parent
    return root


def _get_outer_workspace_root() -> Path:
    """Derive outer workspace root containing the project repository.

    Returns:
        Path to outer workspace root.
    """
    return _get_main_repo_root().parent


@pytest.fixture
def baseline_data_dir() -> Path:
    """Ensure data/ directory exists with ground-truth assets and clean up after.

    Returns:
        Path to canonical data directory.
    """
    data_dir = get_data_dir()
    created_dir = False
    dummy_files: List[Path] = []

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        created_dir = True

    sample_file = data_dir / "sample_ground_truth.txt"
    if not sample_file.exists():
        sample_file.write_text("ground-truth sample data", encoding="utf-8")
        dummy_files.append(sample_file)

    try:
        yield data_dir
    finally:
        for f in dummy_files:
            if f.is_file():
                f.unlink()
        if created_dir and data_dir.is_dir():
            try:
                data_dir.rmdir()
            except OSError:
                pass


def test_storage_confinement_data_directory_untouched(
    baseline_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify scripts write exclusively to outputs/ and leave data/ untouched.

    Snapshots the data/ directory prior to script executions, runs
    extract_keyframes, generate_video, reconstruct_video, and a subprocess
    invocation with synthetic/dummy configs, asserts all outputs land in
    outputs/, and confirms the data/ directory snapshot is strictly unchanged.

    Args:
        baseline_data_dir: Canonical data directory fixture.
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    outputs_dir = get_outputs_dir()
    test_outputs_dir = outputs_dir / "confinement_e2e_test"
    test_outputs_dir.mkdir(parents=True, exist_ok=True)

    initial_snapshot = _snapshot_directory_state(baseline_data_dir)
    assert len(initial_snapshot) > 0, "Initial data snapshot should not be empty."

    try:
        # 1. extract_keyframes execution
        cfg_kf = OmegaConf.load(root / "configs" / "extract_keyframes.yaml")
        cfg_kf.output_json = "outputs/confinement_e2e_test/keyframes.json"
        cfg_kf.video_path = "non_existent_video.mp4"
        cfg_kf.use_dummy_if_missing = True

        out_kf = extract_keyframes(cfg_kf)
        assert out_kf.is_file(), f"Expected keyframes output file at {out_kf}"
        assert _is_relative_to(out_kf, outputs_dir), (
            f"{out_kf} not inside {outputs_dir}"
        )
        assert not _is_relative_to(out_kf, baseline_data_dir), (
            f"{out_kf} leaked into data/"
        )

        # 2. generate_video execution
        cfg_gen = OmegaConf.load(root / "configs" / "generate_video.yaml")
        cfg_gen.output_video = "outputs/confinement_e2e_test/generated_video.mp4"
        cfg_gen.video_path = "non_existent_video.mp4"
        cfg_gen.keyframes_json = "non_existent_kf.json"
        cfg_gen.use_dummy_if_missing = True
        cfg_gen.interp_steps = 1
        cfg_gen.use_chained_transformer = False
        cfg_gen.use_rrdn = False

        out_gen_str = generate_video(cfg_gen)
        out_gen = Path(out_gen_str)
        assert out_gen.is_file(), f"Expected generated video file at {out_gen}"
        assert _is_relative_to(out_gen, outputs_dir), (
            f"{out_gen} not inside {outputs_dir}"
        )
        assert not _is_relative_to(out_gen, baseline_data_dir), (
            f"{out_gen} leaked into data/"
        )

        # 3. reconstruct_video execution
        cfg_recon = OmegaConf.load(root / "configs" / "reconstruct_video.yaml")
        dummy_ckpt = tmp_path / "dummy_vae.pt"
        vae = VAE(latent_dim=int(cfg_recon.latent_dim))
        torch.save(vae.state_dict(), dummy_ckpt)

        dummy_video = tmp_path / "dummy_recon.mp4"
        dummy_video.touch()

        cfg_recon.vae_checkpoint = str(dummy_ckpt)
        cfg_recon.video_path = str(dummy_video)
        cfg_recon.output_video = "outputs/confinement_e2e_test/reconstructed.mp4"
        cfg_recon.sample_frames_dir = (
            "outputs/confinement_e2e_test/reconstructed_samples"
        )

        fake_frame = np.zeros((108, 192, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1
        mock_cap.read.side_effect = [(True, fake_frame), (False, None)]
        monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

        mock_writer = MagicMock()
        monkeypatch.setattr("cv2.VideoWriter", lambda *args, **kwargs: mock_writer)
        monkeypatch.setattr("cv2.imwrite", lambda *args, **kwargs: True)

        out_recon_video, out_recon_samples = reconstruct_video(cfg_recon)
        assert _is_relative_to(out_recon_video, outputs_dir)
        assert _is_relative_to(out_recon_samples, outputs_dir)
        assert not _is_relative_to(out_recon_video, baseline_data_dir)
        assert not _is_relative_to(out_recon_samples, baseline_data_dir)

        # 4. Subprocess execution
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{root}{os.pathsep}{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        )
        subproc_cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from omegaconf import OmegaConf; "
                "from scripts.extract_keyframes import extract_keyframes; "
                "from agilab_lib.utils.storage import get_project_root; "
                "root = get_project_root(); "
                "cfg = OmegaConf.load(root / 'configs' / 'extract_keyframes.yaml'); "
                "cfg.video_path = 'none.mp4'; "
                "cfg.use_dummy_if_missing = True; "
                "cfg.output_json = 'outputs/confinement_e2e_test/subproc_kf.json'; "
                "extract_keyframes(cfg)"
            ),
        ]
        proc = subprocess.run(
            subproc_cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        subproc_out = outputs_dir / "confinement_e2e_test" / "subproc_kf.json"
        assert subproc_out.is_file()
        assert _is_relative_to(subproc_out, outputs_dir)

        # 5. Verify data/ snapshot state remains strictly identical
        final_snapshot = _snapshot_directory_state(baseline_data_dir)
        added_files = set(final_snapshot) - set(initial_snapshot)
        removed_files = set(initial_snapshot) - set(final_snapshot)
        modified_files = {
            k
            for k in set(initial_snapshot) & set(final_snapshot)
            if initial_snapshot[k] != final_snapshot[k]
        }

        assert not added_files, f"Leaked new files in data/: {added_files}"
        assert not removed_files, f"Files deleted from data/: {removed_files}"
        assert not modified_files, f"Files modified in data/: {modified_files}"
        assert final_snapshot == initial_snapshot

    finally:
        if test_outputs_dir.is_dir():
            shutil.rmtree(test_outputs_dir, ignore_errors=True)


def test_invariant_workspace_root_function_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify script function execution from arbitrary external cwd does not leak
    outputs, checkpoints, or data into external cwd.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    external_cwd = tmp_path / "outer_workspace"
    external_cwd.mkdir()

    monkeypatch.chdir(external_cwd)
    assert Path.cwd() == external_cwd.resolve()

    cfg_kf = OmegaConf.load(root / "configs" / "extract_keyframes.yaml")
    cfg_kf.output_json = "outputs/workspace_test/kf_func.json"
    cfg_kf.video_path = "non_existent_video.mp4"
    cfg_kf.use_dummy_if_missing = True

    saved_path = extract_keyframes(cfg_kf)
    expected_path = get_outputs_dir("workspace_test/kf_func.json")

    try:
        assert saved_path == expected_path
        assert saved_path.is_file()

        # Assert no outputs, checkpoints, or data created in external_cwd
        assert not (external_cwd / "outputs").exists()
        assert not (external_cwd / "checkpoints").exists()
        assert not (external_cwd / "data").exists()
        assert list(external_cwd.iterdir()) == []
    finally:
        if expected_path.is_file():
            expected_path.unlink()
        if expected_path.parent.is_dir():
            try:
                expected_path.parent.rmdir()
            except OSError:
                pass


def test_invariant_workspace_root_subprocess_invocation(tmp_path: Path) -> None:
    """Verify script subprocess invocation from arbitrary external cwd routes
    artifacts to project outputs/ without creating outputs or checkpoints in cwd.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    root = get_project_root()
    external_cwd = tmp_path / "outer_workspace_proc"
    external_cwd.mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{root}{os.pathsep}{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    subproc_cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from omegaconf import OmegaConf; "
            "from scripts.extract_keyframes import extract_keyframes; "
            "from agilab_lib.utils.storage import get_project_root; "
            "root = get_project_root(); "
            "cfg = OmegaConf.load(root / 'configs' / 'extract_keyframes.yaml'); "
            "cfg.video_path = 'none.mp4'; "
            "cfg.use_dummy_if_missing = True; "
            "cfg.output_json = 'outputs/workspace_test/kf_subproc.json'; "
            "extract_keyframes(cfg)"
        ),
    ]
    proc = subprocess.run(
        subproc_cmd,
        cwd=str(external_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0

    expected_path = get_outputs_dir("workspace_test/kf_subproc.json")
    try:
        assert expected_path.is_file()

        # Assert outer workspace remains clean
        assert not (external_cwd / "outputs").exists()
        assert not (external_cwd / "checkpoints").exists()
        assert not (external_cwd / "data").exists()
        assert list(external_cwd.iterdir()) == []
    finally:
        if expected_path.is_file():
            expected_path.unlink()
        if expected_path.parent.is_dir():
            try:
                expected_path.parent.rmdir()
            except OSError:
                pass


def test_invariant_workspace_root_hydra_cli_invocation(tmp_path: Path) -> None:
    """Verify Hydra CLI invocation from external cwd respects configured output
    destinations and does not pollute external cwd.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    root = get_project_root()
    external_cwd = tmp_path / "outer_workspace_hydra"
    external_cwd.mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{root}{os.pathsep}{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    hydra_run_dir = (get_outputs_dir("hydra_test_run")).as_posix()

    subproc_cmd = [
        sys.executable,
        str(root / "scripts" / "extract_keyframes.py"),
        "video_path=none.mp4",
        "use_dummy_if_missing=true",
        "output_json=outputs/workspace_test/kf_hydra_cli.json",
        f"hydra.run.dir={hydra_run_dir}",
    ]
    proc = subprocess.run(
        subproc_cmd,
        cwd=str(external_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0

    expected_path = get_outputs_dir("workspace_test/kf_hydra_cli.json")
    try:
        assert expected_path.is_file()
        assert not (external_cwd / "outputs").exists()
        assert not (external_cwd / "checkpoints").exists()
        assert not (external_cwd / "data").exists()
        assert list(external_cwd.iterdir()) == []
    finally:
        if expected_path.is_file():
            expected_path.unlink()
        if expected_path.parent.is_dir():
            try:
                expected_path.parent.rmdir()
            except OSError:
                pass
        shutil.rmtree(root / "outputs" / "hydra_test_run", ignore_errors=True)


def test_invariant_workspace_root_invocation_from_outer_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify executing script function when cwd is outer workspace root
    (Robot_Control) does not create outputs, checkpoints, or data in Robot_Control.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    outer_workspace = _get_outer_workspace_root()
    if not outer_workspace.is_dir():
        pytest.skip(f"Outer workspace root {outer_workspace} not found.")

    monkeypatch.chdir(outer_workspace)
    assert Path.cwd() == outer_workspace.resolve()

    cfg_kf = OmegaConf.load(root / "configs" / "extract_keyframes.yaml")
    cfg_kf.output_json = "outputs/outer_workspace_test/kf.json"
    cfg_kf.video_path = "non_existent.mp4"
    cfg_kf.use_dummy_if_missing = True

    saved_path = extract_keyframes(cfg_kf)
    expected_path = get_outputs_dir("outer_workspace_test/kf.json")

    try:
        assert saved_path == expected_path
        assert saved_path.is_file()
        assert not (outer_workspace / "outputs").exists()
        assert not (outer_workspace / "checkpoints").exists()
        assert not (outer_workspace / "data").exists()
    finally:
        if expected_path.is_file():
            expected_path.unlink()
        if expected_path.parent.is_dir():
            try:
                expected_path.parent.rmdir()
            except OSError:
                pass


def test_outer_workspace_root_contains_no_leaked_directories() -> None:
    """Verify the outer workspace root does not contain outputs,
    checkpoints, or data directories.
    """
    outer_workspace = _get_outer_workspace_root()
    if not outer_workspace.is_dir():
        pytest.skip(f"Outer workspace root {outer_workspace} not accessible.")

    assert not (outer_workspace / "outputs").exists(), (
        f"Leaked outputs in outer workspace: {outer_workspace / 'outputs'}"
    )
    assert not (outer_workspace / "checkpoints").exists(), (
        f"Leaked checkpoints in outer workspace: {outer_workspace / 'checkpoints'}"
    )
    assert not (outer_workspace / "data").exists(), (
        f"Leaked data in outer workspace: {outer_workspace / 'data'}"
    )


def test_all_default_configs_produce_no_outputs_in_data() -> None:
    """Verify all default configuration files produce output targets under outputs/,
    checkpoints/, or .cache/, and strictly none under data/.
    """
    root = get_project_root()
    config_dir = root / "configs"
    yaml_files = sorted(config_dir.glob("*.yaml"))
    assert len(yaml_files) >= 10, (
        f"Expected at least 10 config files, found {len(yaml_files)}"
    )

    data_dir = get_data_dir()
    outputs_dir = get_outputs_dir()
    cache_dir = get_cache_dir()
    checkpoints_dir = get_checkpoints_dir()

    output_keys = {"output_json", "output_video", "sample_frames_dir"}
    cache_keys = {"cache_path"}
    checkpoint_keys = {"output_checkpoint"}

    for yf in yaml_files:
        cfg = OmegaConf.load(yf)
        assert isinstance(cfg, DictConfig)

        # Check explicit output keys
        for key in output_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"{yf.name} key '{key}' starts with data/: {val}"
                )
                resolved = resolve_project_path(val)
                assert _is_relative_to(resolved, outputs_dir), (
                    f"{yf.name} key '{key}' resolved to {resolved}, "
                    "expected under outputs/"
                )
                assert not _is_relative_to(resolved, data_dir), (
                    f"{yf.name} key '{key}' resolved inside data/: {resolved}"
                )

        # Check cache keys
        for key in cache_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"{yf.name} cache key '{key}' starts with data/: {val}"
                )
                resolved = resolve_project_path(val)
                assert _is_relative_to(resolved, cache_dir) or _is_relative_to(
                    resolved, outputs_dir
                ), (
                    f"{yf.name} cache key '{key}' resolved to {resolved}, "
                    "expected under .cache/ or outputs/"
                )
                assert not _is_relative_to(resolved, data_dir), (
                    f"{yf.name} cache key '{key}' resolved inside data/: {resolved}"
                )

        # Check checkpoint keys
        for key in checkpoint_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"{yf.name} checkpoint key '{key}' starts with data/: {val}"
                )
                resolved = resolve_project_path(val)
                assert _is_relative_to(resolved, checkpoints_dir), (
                    f"{yf.name} checkpoint key '{key}' resolved to {resolved}, "
                    "expected under checkpoints/"
                )
                assert not _is_relative_to(resolved, data_dir), (
                    f"{yf.name} checkpoint key '{key}' resolved inside data/: "
                    f"{resolved}"
                )

        # Check output_dir
        if "output_dir" in cfg and cfg["output_dir"] is not None:
            val = str(cfg["output_dir"])
            assert not val.startswith("data/"), (
                f"{yf.name} output_dir starts with data/: {val}"
            )
            resolved = resolve_project_path(val)
            assert _is_relative_to(resolved, outputs_dir) or _is_relative_to(
                resolved, checkpoints_dir
            ), (
                f"{yf.name} output_dir resolved to {resolved}, "
                "expected under outputs/ or checkpoints/"
            )
            assert not _is_relative_to(resolved, data_dir), (
                f"{yf.name} output_dir resolved inside data/: {resolved}"
            )


def test_git_status_cleanliness_no_artifact_clutter() -> None:
    """Verify git status reports no untracked artifact clutter in repository."""
    root = get_project_root()
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    status_output = proc.stdout.strip()
    lines = [line.strip() for line in status_output.splitlines() if line.strip()]

    artifact_extensions = {".mp4", ".png", ".npy", ".pt", ".pth", ".log", ".db3"}
    forbidden_prefixes = ("outputs/", "checkpoints/", ".cache/", "data/")

    untracked_clutter: List[str] = []
    for line in lines:
        if line.startswith("??"):
            path_str = line[2:].strip().strip('"').replace("\\", "/")
            if any(path_str.startswith(prefix) for prefix in forbidden_prefixes):
                untracked_clutter.append(path_str)
            elif any(path_str.endswith(ext) for ext in artifact_extensions):
                untracked_clutter.append(path_str)

    assert not untracked_clutter, (
        f"Untracked artifact clutter in git status: {untracked_clutter}"
    )


def test_main_repo_git_status_cleanliness() -> None:
    """Verify git status in main repo reports strictly empty status."""
    main_repo = _get_main_repo_root()
    if not (main_repo.is_dir() and (main_repo / ".git").exists()):
        pytest.skip("Main repo directory not found.")

    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(main_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    status_output = proc.stdout.strip()
    assert not status_output, (
        f"Git status in main repo is not strictly empty:\n{status_output}"
    )


def test_gitignore_guards_outputs_and_checkpoints_from_clutter() -> None:
    """Verify .gitignore actively prevents outputs, checkpoints, and caches
    from being tracked as untracked clutter.
    """
    root = get_project_root()
    test_output_file = root / "outputs" / "temp_confinement_artifact.mp4"
    test_ckpt_file = root / "checkpoints" / "temp_confinement_ckpt.pt"
    test_cache_file = root / ".cache" / "temp_confinement_cache.npy"

    test_output_file.parent.mkdir(parents=True, exist_ok=True)
    test_ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    test_cache_file.parent.mkdir(parents=True, exist_ok=True)

    test_output_file.write_bytes(b"dummy_artifact")
    test_ckpt_file.write_bytes(b"dummy_ckpt")
    test_cache_file.write_bytes(b"dummy_cache")

    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        status_lines = proc.stdout.strip().splitlines()
        for line in status_lines:
            assert "temp_confinement_artifact.mp4" not in line
            assert "temp_confinement_ckpt.pt" not in line
            assert "temp_confinement_cache.npy" not in line
    finally:
        for f in (test_output_file, test_ckpt_file, test_cache_file):
            if f.is_file():
                f.unlink()


def test_storage_confinement_writing_output_to_data_raises_permission_error() -> None:
    """Verify writing outputs to data/ raises PermissionError across APIs
    and scripts.
    """
    root = get_project_root()

    # 1. API level: ensure_writable_output_path
    with pytest.raises(
        PermissionError,
        match="Storage boundary violation: data/ is strictly read-only",
    ):
        ensure_writable_output_path("data/unauthorized_output.json")

    with pytest.raises(
        PermissionError,
        match="Storage boundary violation: data/ is strictly read-only",
    ):
        ensure_writable_output_path(root / "data" / "sub" / "output.mp4")

    # 2. Script level: extract_keyframes with output targeting data/
    cfg_kf = OmegaConf.load(root / "configs" / "extract_keyframes.yaml")
    cfg_kf.output_json = "data/unauthorized_kf.json"
    cfg_kf.video_path = "non_existent.mp4"
    cfg_kf.use_dummy_if_missing = True

    with pytest.raises(
        PermissionError,
        match="Storage boundary violation: data/ is strictly read-only",
    ):
        extract_keyframes(cfg_kf)

    # 3. Script level: generate_video with output targeting data/
    cfg_gen = OmegaConf.load(root / "configs" / "generate_video.yaml")
    cfg_gen.output_video = "data/unauthorized_gen.mp4"
    cfg_gen.video_path = "non_existent.mp4"
    cfg_gen.keyframes_json = "non_existent.json"
    cfg_gen.use_dummy_if_missing = True
    cfg_gen.interp_steps = 1
    cfg_gen.use_chained_transformer = False
    cfg_gen.use_rrdn = False

    with pytest.raises(
        PermissionError,
        match="Storage boundary violation: data/ is strictly read-only",
    ):
        generate_video(cfg_gen)

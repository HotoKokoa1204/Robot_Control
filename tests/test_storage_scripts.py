"""Module: test_storage_scripts
Stage: Tests
Author: KafuuChino
Date: 2026-10-06
Description: Unit and integration tests for script defaults and config
    storage seam routing.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import (
    get_checkpoints_dir,
    get_outputs_dir,
    get_project_root,
    resolve_project_path,
)
from scripts.extract_keyframes import extract_keyframes
from scripts.generate_video import generate_video
from scripts.reconstruct_video import reconstruct_video


def test_all_configs_default_outputs_route_to_outputs_or_cache() -> None:
    """Verify all YAML configs route outputs/caches away from data/ into outputs/
    or .cache/."""
    root = get_project_root()
    config_dir = root / "configs"
    assert config_dir.is_dir(), f"Config directory missing: {config_dir}"

    yaml_files = list(config_dir.glob("*.yaml"))
    assert len(yaml_files) >= 4, f"Found too few config files: {len(yaml_files)}"

    output_keys = {"output_json", "output_video", "sample_frames_dir"}
    cache_keys = {"cache_path"}
    checkpoint_keys = {"output_checkpoint"}

    for yf in yaml_files:
        cfg = OmegaConf.load(yf)
        assert isinstance(cfg, DictConfig)

        # Output keys must point inside outputs/
        for key in output_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"Config {yf.name} key '{key}' points to data/: {val}"
                )
                assert val.startswith("outputs/"), (
                    f"Config {yf.name} key '{key}' should point to outputs/: {val}"
                )

        # In extract_latents.yaml, output_dir is an output cache
        if yf.name == "extract_latents.yaml":
            out_dir = str(cfg.output_dir)
            assert not out_dir.startswith("data/"), (
                f"extract_latents.yaml output_dir points to data/: {out_dir}"
            )
            assert out_dir.startswith("outputs/"), (
                f"extract_latents.yaml output_dir should point to outputs/: {out_dir}"
            )

        # Cache keys must point inside .cache/ or outputs/
        for key in cache_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"Config {yf.name} cache key '{key}' points to data/: {val}"
                )
                assert val.startswith(".cache/") or val.startswith("outputs/"), (
                    f"Config {yf.name} cache key '{key}' must be under .cache/ or "
                    f"outputs/: {val}"
                )

        # Checkpoint keys must point inside checkpoints/
        for key in checkpoint_keys:
            if key in cfg and cfg[key] is not None:
                val = str(cfg[key])
                assert not val.startswith("data/"), (
                    f"Config {yf.name} key '{key}' points to data/: {val}"
                )
                assert val.startswith("checkpoints/"), (
                    f"Config {yf.name} key '{key}' should point to checkpoints/: {val}"
                )


def test_specific_config_default_paths() -> None:
    """Verify specific config files have expected output path defaults."""
    root = get_project_root()

    # 1. extract_keyframes.yaml
    cfg_kf = OmegaConf.load(root / "configs" / "extract_keyframes.yaml")
    assert cfg_kf.output_json == "outputs/keyframes/keyframes.json"

    # 2. extract_latents.yaml
    cfg_latents = OmegaConf.load(root / "configs" / "extract_latents.yaml")
    assert cfg_latents.output_dir == "outputs/latents"

    # 3. generate_video.yaml
    cfg_gen = OmegaConf.load(root / "configs" / "generate_video.yaml")
    assert cfg_gen.output_video == "outputs/eval/generated_video.mp4"

    # 4. reconstruct_video.yaml
    cfg_recon = OmegaConf.load(root / "configs" / "reconstruct_video.yaml")
    assert cfg_recon.output_video == "outputs/eval/reconstructed_eval_vae.mp4"
    assert cfg_recon.sample_frames_dir == "outputs/samples/reconstructed_samples"

    # 5. train_vae.yaml
    cfg_vae = OmegaConf.load(root / "configs" / "train_vae.yaml")
    assert cfg_vae.cache_path == ".cache/vae_dataset_cache_88v.npy"


def test_extract_keyframes_routes_to_outputs(tmp_path: Path) -> None:
    """Verify extract_keyframes resolves output path canonically and creates
    parent directories.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    root = get_project_root()
    cfg_path = root / "configs" / "extract_keyframes.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Use a unique subfolder inside outputs to avoid conflicts
    rel_output = "outputs/test_keyframes/test_kf.json"
    cfg.output_json = rel_output
    cfg.video_path = "non_existent_video_path.mp4"
    cfg.use_dummy_if_missing = True

    resolved_expected = get_outputs_dir("test_keyframes/test_kf.json")
    saved_path = extract_keyframes(cfg)

    try:
        assert saved_path == resolved_expected
        assert saved_path.is_file()
        assert saved_path.parent.is_dir()

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "keyframe_indices" in data
        assert isinstance(data["keyframe_indices"], list)
    finally:
        if saved_path.is_file():
            saved_path.unlink()
        if saved_path.parent.is_dir():
            saved_path.parent.rmdir()


def test_generate_video_routes_to_outputs() -> None:
    """Verify generate_video routes output path to outputs/ and creates parent
    directory."""
    root = get_project_root()
    cfg_path = root / "configs" / "generate_video.yaml"
    cfg = OmegaConf.load(cfg_path)

    rel_output = "outputs/test_gen/test_video.mp4"
    cfg.output_video = rel_output
    cfg.video_path = "non_existent_video.mp4"
    cfg.keyframes_json = "non_existent_kf.json"
    cfg.use_dummy_if_missing = True
    cfg.interp_steps = 1
    cfg.use_chained_transformer = False
    cfg.use_rrdn = False

    expected_path = get_outputs_dir("test_gen/test_video.mp4")
    out_video_str = generate_video(cfg)

    try:
        assert Path(out_video_str) == expected_path
        assert expected_path.parent.is_dir()
    finally:
        if expected_path.is_file():
            expected_path.unlink()
        if expected_path.parent.is_dir():
            expected_path.parent.rmdir()


def test_reconstruct_video_routes_to_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify reconstruct_video routes video and sample frames to outputs/ directory.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    cfg_path = root / "configs" / "reconstruct_video.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Create dummy VAE checkpoint
    dummy_ckpt = tmp_path / "dummy_vae.pt"
    vae = VAE(latent_dim=int(cfg.latent_dim))
    torch.save(vae.state_dict(), dummy_ckpt)

    # Create dummy source video file
    dummy_video = tmp_path / "dummy_source.mp4"
    dummy_video.touch()

    # Route outputs into test subdirectories under outputs/
    cfg.vae_checkpoint = str(dummy_ckpt)
    cfg.video_path = str(dummy_video)
    cfg.output_video = "outputs/test_recon_eval/eval_recon.mp4"
    cfg.sample_frames_dir = "outputs/test_recon_samples/sample_dir"

    # Mock cv2.VideoCapture to return a single frame
    fake_frame = np.zeros((108, 192, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 1
    mock_cap.read.side_effect = [(True, fake_frame), (False, None)]

    monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: mock_cap)

    # Mock cv2.VideoWriter
    mock_writer = MagicMock()
    monkeypatch.setattr("cv2.VideoWriter", lambda *args, **kwargs: mock_writer)

    # Mock cv2.imwrite
    monkeypatch.setattr("cv2.imwrite", lambda *args, **kwargs: True)

    out_video_path, out_sample_dir = reconstruct_video(cfg)

    expected_video = get_outputs_dir("test_recon_eval/eval_recon.mp4")
    expected_samples = get_outputs_dir("test_recon_samples/sample_dir")

    try:
        assert out_video_path == expected_video
        assert out_sample_dir == expected_samples
        assert expected_video.parent.is_dir()
        assert expected_samples.is_dir()
    finally:
        if expected_video.parent.is_dir():
            expected_video.parent.rmdir()
        if expected_samples.is_dir():
            expected_samples.rmdir()


def test_extract_latents_storage_resolution() -> None:
    """Verify extract_latents configuration resolves output_dir under outputs/."""
    root = get_project_root()
    cfg_path = root / "configs" / "extract_latents.yaml"
    cfg = OmegaConf.load(cfg_path)

    resolved_out = resolve_project_path(cfg.output_dir)
    assert resolved_out == get_outputs_dir("latents")
    assert resolved_out.is_absolute()

    resolved_data = resolve_project_path(cfg.data_root)
    assert resolved_data == (root / "data").resolve()


def test_training_scripts_output_checkpoints_route_to_checkpoints() -> None:
    """Verify training scripts have default checkpoint destinations routed under
    checkpoints/."""
    root = get_project_root()
    train_configs = [
        "train_angle_predictor.yaml",
        "train_forward.yaml",
        "train_multibranch.yaml",
        "train_rlt.yaml",
        "train_rotation.yaml",
        "train_vae.yaml",
    ]

    for cfg_name in train_configs:
        cfg = OmegaConf.load(root / "configs" / cfg_name)
        if "output_checkpoint" in cfg:
            resolved_ckpt = resolve_project_path(cfg.output_checkpoint)
            assert str(resolved_ckpt).startswith(str(get_checkpoints_dir()))
        if "output_dir" in cfg:
            resolved_dir = resolve_project_path(cfg.output_dir)
            assert str(resolved_dir).startswith(str(get_checkpoints_dir()))


def test_script_path_resolution_invariant_to_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify path resolution in scripts remains invariant when working
    directory changes.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    root = get_project_root()
    monkeypatch.chdir(tmp_path)

    # In a foreign working directory, relative config output paths must still
    # resolve to project root outputs/
    res_kf = resolve_project_path("outputs/keyframes/keyframes.json")
    assert res_kf == (root / "outputs" / "keyframes" / "keyframes.json").resolve()

    res_video = resolve_project_path("outputs/eval/generated_video.mp4")
    assert res_video == (root / "outputs" / "eval" / "generated_video.mp4").resolve()

    res_cache = resolve_project_path(".cache/vae_dataset_cache_88v.npy")
    assert res_cache == (root / ".cache" / "vae_dataset_cache_88v.npy").resolve()

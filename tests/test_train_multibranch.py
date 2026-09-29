"""
Module: test_train_multibranch
Stage: Test
Author: KafuuChino
Date: 2026-09-29
Description: Integration and unit tests for joint multi-branch training script,
    verifying Hydra config loading, AMP execution, gradient accumulation,
    telemetry logging, and standalone checkpoint exports.
"""

import math
from pathlib import Path

# Windows DLL initialization guard for PIL/torchvision
from PIL import Image  # isort: skip # noqa: F401

import pytest
import torch
from agilab_lib.datasets.dual_source_dataset import (
    DummyDualSourceVideoDataset,
)
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.models.rlt import (
    ForwardLatentTransformer,
    RotationLatentTransformer,
)
from agilab_lib.models.vae import VAE
from omegaconf import DictConfig, OmegaConf

from scripts.train_multibranch import (
    get_dataset,
    save_all_checkpoints,
    train_multibranch,
)


def test_config_loading() -> None:
    """Verify train_multibranch.yaml Hydra configuration parameters."""
    config_path = Path("configs/train_multibranch.yaml")
    assert config_path.exists(), f"Missing config file: {config_path}"

    cfg: DictConfig = OmegaConf.load(config_path)

    # Data settings
    assert cfg.one_path_dir == "data/one_path"
    assert cfg.rotation_dir == "data/360"
    assert cfg.img_height == 108
    assert cfg.img_width == 192

    # Model architecture parameters
    assert cfg.latent_dim == 512
    assert cfg.hidden_dim == 512
    assert cfg.num_blocks == 5
    assert cfg.block_inner_dim == 512

    # Optimization parameters
    assert float(cfg.lr) == 1e-4
    assert cfg.batch_size == 16
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.use_amp is True
    assert cfg.max_epochs == 50

    # Multi-branch loss weights
    assert float(cfg.alpha_perc) == 0.5
    assert float(cfg.beta_kl) == 0.0001
    assert float(cfg.w_fwd) == 1.0
    assert float(cfg.w_rot) == 1.0
    assert float(cfg.w_recon) == 1.0

    # Checkpoints
    assert cfg.output_dir == "checkpoints"
    assert cfg.vae_checkpoint is None
    assert cfg.save_interval_epochs == 5


def test_get_dataset_fallback(tmp_path: Path) -> None:
    """Verify get_dataset falls back to DummyDualSourceVideoDataset when missing."""
    cfg = OmegaConf.create(
        {
            "one_path_dir": str(tmp_path / "nonexistent_one_path"),
            "rotation_dir": str(tmp_path / "nonexistent_360"),
            "use_dummy_if_missing": True,
            "num_samples": 8,
            "img_height": 108,
            "img_width": 192,
            "seed": 123,
        }
    )

    dataset = get_dataset(cfg)
    assert isinstance(dataset, DummyDualSourceVideoDataset)
    assert len(dataset) == 8

    # When fallback is disabled, should raise FileNotFoundError
    cfg_strict = OmegaConf.create(
        {
            "one_path_dir": str(tmp_path / "nonexistent_one_path"),
            "rotation_dir": str(tmp_path / "nonexistent_360"),
            "use_dummy_if_missing": False,
        }
    )
    with pytest.raises(FileNotFoundError):
        get_dataset(cfg_strict)


def test_save_all_checkpoints_export(tmp_path: Path) -> None:
    """Verify composite and standalone checkpoint export and load compatibility."""
    out_dir = tmp_path / "test_ckpts"
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    paths = save_all_checkpoints(model, out_dir)

    assert paths["composite"].exists()
    assert paths["vae"].exists()
    assert paths["forward"].exists()
    assert paths["rotation"].exists()

    # Verify composite checkpoint can reload into JointNavigationModel
    loaded_composite = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    loaded_composite.load_state_dict(
        torch.load(paths["composite"], map_location="cpu", weights_only=True)
    )

    # Verify standalone VAE checkpoint can load into standalone VAE
    loaded_vae = VAE(latent_dim=64)
    loaded_vae.load_state_dict(
        torch.load(paths["vae"], map_location="cpu", weights_only=True)
    )

    # Verify standalone Forward transformer checkpoint
    loaded_fwd = ForwardLatentTransformer(
        latent_dim=64, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )
    loaded_fwd.load_state_dict(
        torch.load(paths["forward"], map_location="cpu", weights_only=True)
    )

    # Verify standalone Rotation transformer checkpoint
    loaded_rot = RotationLatentTransformer(
        latent_dim=64, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )
    loaded_rot.load_state_dict(
        torch.load(paths["rotation"], map_location="cpu", weights_only=True)
    )


def test_train_multibranch_loop_execution(tmp_path: Path) -> None:
    """Integration test: Verify training loop with gradient accumulation."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=8,
        img_height=108,
        img_width=192,
        seed=42,
    )

    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    out_dir = tmp_path / "checkpoints"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "use_amp": torch.cuda.is_available(),
            "max_epochs": 4,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 2,
            "vae_checkpoint": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )

    trained_model, history = train_multibranch(cfg, dataset=dataset, model=model)

    # Check telemetry keys
    for key in ("loss", "recon", "fwd", "rot", "perc", "kl"):
        assert key in history
        assert len(history[key]) == 4
        assert all(math.isfinite(val) for val in history[key])

    # Check loss trend: loss should decrease over training
    assert history["loss"][-1] < history["loss"][0]

    # Verify exported checkpoints exist
    assert (out_dir / "joint_navigation_model.pt").exists()
    assert (out_dir / "vae_512.pt").exists()
    assert (out_dir / "forward_transformer.pt").exists()
    assert (out_dir / "rotation_transformer.pt").exists()


def test_warm_start_vae_checkpoint(tmp_path: Path) -> None:
    """Verify warm-start loading of VAE weights during training initialization."""
    # Pre-train / create a mock VAE checkpoint
    src_vae = VAE(latent_dim=64)
    with torch.no_grad():
        src_vae.fc_mu.weight.fill_(0.777)

    ckpt_path = tmp_path / "warm_start_vae.pt"
    torch.save(src_vae.state_dict(), ckpt_path)

    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
        img_height=108,
        img_width=192,
        seed=42,
    )

    out_dir = tmp_path / "out"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 0.001,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "use_amp": False,
            "max_epochs": 1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 1,
            "vae_checkpoint": str(ckpt_path),
            "device": "cpu",
        }
    )

    trained_model, history = train_multibranch(cfg, dataset=dataset)
    assert isinstance(trained_model, JointNavigationModel)
    assert len(history["loss"]) == 1
    assert math.isfinite(history["loss"][0])


def test_gradient_accumulation_and_amp(tmp_path: Path) -> None:
    """Verify AMP execution and parameter updates with gradient accumulation."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
        img_height=108,
        img_width=192,
        seed=101,
    )
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    init_fwd_weight = model.forward_transformer.fc_in.weight.clone()

    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "use_amp": torch.cuda.is_available(),
            "max_epochs": 1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(tmp_path / "amp_out"),
            "save_interval_epochs": 1,
            "vae_checkpoint": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )

    trained_model, history = train_multibranch(cfg, dataset=dataset, model=model)
    assert not torch.allclose(
        trained_model.forward_transformer.fc_in.weight.cpu(),
        init_fwd_weight.cpu(),
    )
    assert len(history["loss"]) == 1
    assert math.isfinite(history["loss"][0])

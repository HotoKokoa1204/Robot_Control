"""
Module: test_train_multibranch
Stage: Test
Author: KafuuChino
Date: 2026-09-29
Description: Integration and unit tests for joint multi-branch training script,
    verifying Hydra config loading, three-stage curricular execution, AMP execution,
    gradient accumulation, telemetry logging, loss curve plotting, and milestone
    checkpoint exports.
"""

import math
from pathlib import Path
from typing import Any

# Windows DLL initialization guard for PIL/torchvision
from PIL import Image  # isort: skip # noqa: F401

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.datasets.dual_source_dataset import (
    DummyDualSourceVideoDataset,
)
from agilab_lib.models.joint_loss import JointNavigationLoss
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.models.rlt import (
    ForwardLatentTransformer,
    RotationLatentTransformer,
)
from agilab_lib.models.vae import VAE
from scripts.train_multibranch import (
    get_dataset,
    plot_loss_curves,
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
    assert cfg.preload_frames is True
    assert float(cfg.buffer_distance_meters) == 2.0

    # Model architecture parameters
    assert cfg.latent_dim == 512
    assert cfg.hidden_dim == 512
    assert cfg.num_blocks == 5
    assert cfg.block_inner_dim == 512

    # Optimization parameters
    assert float(cfg.lr) == 1e-4
    assert float(cfg.lr_trans) == 1e-4
    assert float(cfg.lr_vae_stage1) == 1e-4
    assert float(cfg.lr_vae_stage3) == 1e-5
    assert float(cfg.lr_vae) == 1e-5
    assert cfg.batch_size == 16
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.use_amp is True
    assert cfg.max_epochs == 150
    assert cfg.stage1_epochs == 50
    assert cfg.stage2_epochs == 50
    assert cfg.stage3_epochs == 50
    assert cfg.warmup_vae_epochs == 50

    # Multi-branch loss weights
    assert float(cfg.alpha_perc) == 0.5
    assert float(cfg.beta_kl) == 0.0001
    assert float(cfg.w_fwd) == 1.0
    assert float(cfg.w_rot) == 1.0
    assert float(cfg.w_recon) == 1.0
    assert float(cfg.w_latent) == 1.0
    assert float(cfg.w_latent_min) == 0.01
    assert float(cfg.w_latent_max) == 1.0
    assert cfg.w_latent_ramp_start_epoch == 61
    assert float(cfg.sanity_check_min_std) == 15.0

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


def test_plot_loss_curves(tmp_path: Path) -> None:
    """Verify plot_loss_curves generates valid image file."""
    history = {
        "loss": [1.0, 0.8, 0.6],
        "recon": [0.5, 0.0, 0.3],
        "fwd": [0.0, 0.4, 0.2],
        "rot": [0.0, 0.3, 0.1],
        "perc": [0.2, 0.1, 0.05],
        "kl": [0.01, 0.0, 0.005],
        "fwd_latent": [0.0, 0.05, 0.02],
        "rot_latent": [0.0, 0.04, 0.01],
    }
    plot_path = plot_loss_curves(history, tmp_path)
    assert plot_path is not None
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


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
            "lr": 0.001,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "use_amp": torch.cuda.is_available(),
            "max_epochs": 4,
            "warmup_vae_epochs": 0,
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
    for key in (
        "loss",
        "recon",
        "fwd",
        "rot",
        "perc",
        "kl",
        "fwd_latent",
        "rot_latent",
    ):
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
    init_fwd_weight = model.forward_transformer.blocks[0].fc3.weight.clone()

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
            "warmup_vae_epochs": 0,
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
        trained_model.forward_transformer.blocks[0].fc3.weight.cpu(),
        init_fwd_weight.cpu(),
    )
    assert len(history["loss"]) == 1
    assert math.isfinite(history["loss"][0])


def test_two_stage_warmup_and_sanity_gate(tmp_path: Path) -> None:
    """Verify Stage 1 VAE warm-up, sanity gate validation, and Stage 2 transition."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=8,
        img_height=108,
        img_width=192,
        seed=777,
    )
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    out_dir = tmp_path / "two_stage_out"
    # Stage 1: 2 epochs, Stage 2: 2 epochs (max_epochs=4)
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "lr_vae": 1e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "use_amp": torch.cuda.is_available(),
            "max_epochs": 4,
            "warmup_vae_epochs": 2,
            "w_latent_min": 0.01,
            "w_latent_max": 1.0,
            "w_latent_ramp_start_epoch": 3,
            "sanity_check_min_std": 0.1,  # Low threshold for random dummy images
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
    assert len(history["loss"]) == 4
    # In Stage 1 (epoch 1 and 2), dynamics loss should be 0
    # because Stage 1 reconstruction-only is active
    assert history["fwd"][0] == 0.0
    assert history["rot"][0] == 0.0
    # In Stage 2 (epoch 3 and 4), dynamics loss should be active (> 0)
    assert history["fwd"][2] > 0.0
    assert history["rot"][2] > 0.0


def test_three_stage_curricular_training_and_sanity_gate(tmp_path: Path) -> None:
    """Verify three-stage curriculum execution, boundary hooks, and milestones."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
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

    out_dir = tmp_path / "three_stage_out"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "lr_vae_stage1": 1e-4,
            "lr_vae_stage3": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "use_amp": False,
            "max_epochs": 3,
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "stage3_epochs": 1,
            "w_latent": 1.0,
            "sanity_check_min_std": 0.1,  # Low threshold for dummy images
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 1,
            "vae_checkpoint": None,
            "device": "cpu",
        }
    )

    trained_model, history = train_multibranch(cfg, dataset=dataset, model=model)
    assert len(history["loss"]) == 3

    # Stage 1 (epoch 0): Recon active, Dynamics inactive
    assert history["recon"][0] > 0.0
    assert history["fwd"][0] == 0.0
    assert history["rot"][0] == 0.0

    # Stage 2 (epoch 1): Recon inactive, Dynamics active
    assert history["recon"][1] == 0.0
    assert history["fwd"][1] > 0.0
    assert history["rot"][1] > 0.0

    # Stage 3 (epoch 2): Both Recon and Dynamics active
    assert history["recon"][2] > 0.0
    assert history["fwd"][2] > 0.0
    assert history["rot"][2] > 0.0

    # Milestone checkpoints verification
    stage1_ckpt = out_dir / "stage1_vae.pt"
    stage2_ckpt = out_dir / "stage2_transformers.pt"
    joint_ckpt = out_dir / "joint_navigation_model.pt"
    final_joint_ckpt = out_dir / "final_joint_model.pt"
    loss_curves_plot = out_dir / "loss_curves.png"

    assert stage1_ckpt.exists()
    assert stage2_ckpt.exists()
    assert joint_ckpt.exists()
    assert final_joint_ckpt.exists()
    assert loss_curves_plot.exists()

    # Verify stage1 checkpoint can load into standalone VAE
    loaded_vae = VAE(latent_dim=64)
    loaded_vae.load_state_dict(
        torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    )

    # Verify stage2 checkpoint contains both transformer state dicts
    stage2_payload = torch.load(stage2_ckpt, map_location="cpu", weights_only=True)
    assert "forward_transformer" in stage2_payload
    assert "rotation_transformer" in stage2_payload

    # Verify parameter state after stage 3: VAE trainable, transformers frozen
    assert trained_model.vae.fc_mu.weight.requires_grad is True
    assert trained_model.forward_transformer.blocks[0].fc1.weight.requires_grad is False
    assert (
        trained_model.rotation_transformer.blocks[0].fc1.weight.requires_grad is False
    )


def test_three_stage_curricular_w_latent_schedule(tmp_path: Path) -> None:
    """Verify that current_w_latent is set to 0.0 in Stage 1, 1.0 in Stage 2,
    and 0.0 in Stage 3.
    """
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
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

    out_dir = tmp_path / "w_latent_test_out"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "lr_trans": 1e-4,
            "lr_vae_stage1": 1e-4,
            "lr_vae_stage3": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "use_amp": False,
            "max_epochs": 3,
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "stage3_epochs": 1,
            "w_latent": 1.0,
            "sanity_check_min_std": 0.1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 1,
            "vae_checkpoint": None,
            "device": "cpu",
        }
    )

    observed_w_latents = []
    loss_fn = JointNavigationLoss()
    original_forward = loss_fn.forward_model

    def tracking_forward(*args: Any, **kwargs: Any) -> Any:
        observed_w_latents.append(loss_fn.w_latent)
        return original_forward(*args, **kwargs)

    loss_fn.forward_model = tracking_forward  # type: ignore[assignment]

    train_multibranch(cfg, dataset=dataset, model=model, loss_fn=loss_fn)

    # 4 samples / batch_size 2 = 2 batches per epoch (epochs: 0, 1, 2)
    assert len(observed_w_latents) == 6
    # Stage 1: w_latent = 0.0
    assert observed_w_latents[0] == 0.0
    assert observed_w_latents[1] == 0.0
    # Stage 2: w_latent = 1.0
    assert observed_w_latents[2] == 1.0
    assert observed_w_latents[3] == 1.0
    # Stage 3: w_latent = 0.0
    assert observed_w_latents[4] == 0.0
    assert observed_w_latents[5] == 0.0

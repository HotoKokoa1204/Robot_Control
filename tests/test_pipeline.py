"""Module: test_pipeline
Stage: Test
Author: KafuuChino
Date: 2026-10-12
Description: Integration and unit tests for CurricularTrainingPipeline, TrainingResult,
    and the train_multibranch CLI driver.
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from agilab_lib.training import (
    CurricularTrainingPipeline,
    TrainingResult,
)
from agilab_lib.training.pipeline import (
    _plot_loss_curves,
    _save_all_checkpoints,
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
    assert float(cfg.beta_kl) in (0.0, 0.0001)
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


def test_cli_driver_execution(tmp_path: Path) -> None:
    """Verify scripts/train_multibranch.py executes end-to-end as a CLI entrypoint."""
    out_dir = tmp_path / "cli_driver_ckpts"
    cmd = [
        sys.executable,
        "scripts/train_multibranch.py",
        f"output_dir={out_dir}",
        "max_epochs=1",
        "stage1_epochs=1",
        "stage2_epochs=0",
        "stage3_epochs=0",
        "num_samples=2",
        "batch_size=2",
        "use_dummy_if_missing=true",
        "use_amp=false",
        "sanity_check_min_std=0.001",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"CLI driver failed: {res.stderr}\n{res.stdout}"
    assert (out_dir / "stage1_vae.pt").exists()
    assert (out_dir / "final_joint_model.pt").exists()


def test_cli_driver_missing_data_strict_fails(tmp_path: Path) -> None:
    """Verify CLI driver exits with error when data dirs are missing."""
    missing_one_path = tmp_path / "missing_one_path"
    missing_360 = tmp_path / "missing_360"
    cmd = [
        sys.executable,
        "scripts/train_multibranch.py",
        f"one_path_dir={missing_one_path}",
        f"rotation_dir={missing_360}",
        "use_dummy_if_missing=false",
        "max_epochs=1",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode != 0
    assert "Missing data dirs" in res.stderr or "FileNotFoundError" in res.stderr


def test_save_all_checkpoints_export(tmp_path: Path) -> None:
    """Verify composite and standalone checkpoint export and load compatibility."""
    out_dir = tmp_path / "test_ckpts"
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    paths = _save_all_checkpoints(model, out_dir)

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
    plot_path = _plot_loss_curves(history, tmp_path)
    assert plot_path is not None
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_training_result_dataclass_defaults() -> None:
    """Verify default initialization and attributes of TrainingResult."""
    result = TrainingResult()
    assert result.history == {}
    assert result.checkpoint_paths == {}
    assert result.sanity_gate_passed is True
    assert result.sanity_metrics == {}


def test_train_multibranch_loop_execution(tmp_path: Path) -> None:
    """Integration test: Verify training loop execution with gradient accumulation."""
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
            "stage1_epochs": 0,
            "stage2_epochs": 4,
            "stage3_epochs": 0,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 2,
            "vae_checkpoint": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg)

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
        assert key in result.history
        assert len(result.history[key]) == 4
        assert all(math.isfinite(val) for val in result.history[key])

    # Check loss trend: loss should decrease over training
    assert result.history["loss"][-1] < result.history["loss"][0]

    # Verify exported checkpoints exist
    assert (out_dir / "joint_navigation_model.pt").exists()
    assert (out_dir / "vae_512.pt").exists()
    assert (out_dir / "forward_transformer.pt").exists()
    assert (out_dir / "rotation_transformer.pt").exists()


def test_warm_start_vae_checkpoint(tmp_path: Path) -> None:
    """Verify warm-start loading of VAE weights during training initialization."""
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
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    out_dir = tmp_path / "out"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 0.0,
            "lr_vae_stage1": 0.0,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "use_amp": False,
            "max_epochs": 1,
            "stage1_epochs": 1,
            "stage2_epochs": 0,
            "stage3_epochs": 0,
            "sanity_check_min_std": 0.1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 1,
            "vae_checkpoint": str(ckpt_path),
            "device": "cpu",
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg)
    assert isinstance(result, TrainingResult)
    assert len(result.history["loss"]) == 1
    assert math.isfinite(result.history["loss"][0])
    assert torch.allclose(
        model.vae.fc_mu.weight, torch.full_like(model.vae.fc_mu.weight, 0.777)
    )


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
            "stage1_epochs": 0,
            "stage2_epochs": 1,
            "stage3_epochs": 0,
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
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg)
    assert not torch.allclose(
        model.forward_transformer.blocks[0].fc3.weight.cpu(),
        init_fwd_weight.cpu(),
    )
    assert len(result.history["loss"]) == 1
    assert math.isfinite(result.history["loss"][0])


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
            "stage1_epochs": 2,
            "stage2_epochs": 2,
            "stage3_epochs": 0,
            "warmup_vae_epochs": 2,
            "w_latent_min": 0.01,
            "w_latent_max": 1.0,
            "w_latent_ramp_start_epoch": 3,
            "sanity_check_min_std": 0.1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 2,
            "vae_checkpoint": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg)
    history = result.history
    assert len(history["loss"]) == 4
    # In Stage 1 (epoch 1 and 2), dynamics loss should be 0
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
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg)
    history = result.history
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
    assert model.vae.fc_mu.weight.requires_grad is True
    assert model.forward_transformer.blocks[0].fc1.weight.requires_grad is False
    assert model.rotation_transformer.blocks[0].fc1.weight.requires_grad is False


def test_three_stage_curricular_w_latent_schedule(tmp_path: Path) -> None:
    """Verify that current_w_latent is set correctly across all 3 stages."""
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
    observed_w_latents: List[float] = []
    loss_fn = JointNavigationLoss()
    original_forward = loss_fn.forward_model

    def tracking_forward(*args: Any, **kwargs: Any) -> Any:
        observed_w_latents.append(loss_fn.w_latent)
        return original_forward(*args, **kwargs)

    loss_fn.forward_model = tracking_forward  # type: ignore[assignment]

    cfg = {
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
        "verbose": False,
        "loss_fn": loss_fn,
    }

    pipeline = CurricularTrainingPipeline()
    pipeline.fit(model=model, dataset=dataset, cfg=cfg)

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


def test_parameter_freezing_invariants_across_stages(tmp_path: Path) -> None:
    """Verify parameter freezing invariants across each curricular stage."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
        img_height=108,
        img_width=192,
        seed=123,
    )
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    stage_records: List[Tuple[int, bool, bool, bool]] = []

    def tracking_callback(
        epoch_idx: int, stage_idx: int, metrics: Dict[str, float]
    ) -> None:
        vae_trainable = any(p.requires_grad for p in model.vae.parameters())
        fwd_trainable = any(
            p.requires_grad for p in model.forward_transformer.parameters()
        )
        rot_trainable = any(
            p.requires_grad for p in model.rotation_transformer.parameters()
        )
        stage_records.append((stage_idx, vae_trainable, fwd_trainable, rot_trainable))

    cfg = {
        "latent_dim": 64,
        "hidden_dim": 32,
        "num_blocks": 2,
        "block_inner_dim": 32,
        "batch_size": 2,
        "use_amp": False,
        "max_epochs": 3,
        "stage1_epochs": 1,
        "stage2_epochs": 1,
        "stage3_epochs": 1,
        "sanity_check_min_std": 0.1,
        "output_dir": str(tmp_path / "invariants_out"),
        "device": "cpu",
        "verbose": False,
        "epoch_callback": tracking_callback,
    }

    pipeline = CurricularTrainingPipeline()
    pipeline.fit(
        model=model,
        dataset=dataset,
        cfg=cfg,
    )

    assert len(stage_records) == 3

    # Stage 1: VAE is trainable, transformers are frozen
    stage1, vae_s1, fwd_s1, rot_s1 = stage_records[0]
    assert stage1 == 1
    assert vae_s1 is True
    assert fwd_s1 is False
    assert rot_s1 is False

    # Stage 2: VAE is frozen, transformers are trainable
    stage2, vae_s2, fwd_s2, rot_s2 = stage_records[1]
    assert stage2 == 2
    assert vae_s2 is False
    assert fwd_s2 is True
    assert rot_s2 is True

    # Stage 3: VAE is trainable, transformers are frozen
    stage3, vae_s3, fwd_s3, rot_s3 = stage_records[2]
    assert stage3 == 3
    assert vae_s3 is True
    assert fwd_s3 is False
    assert rot_s3 is False


def test_sanity_gate_failure_handling(tmp_path: Path) -> None:
    """Verify sanity gate raises RuntimeError on failure or records flag."""
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
        img_height=108,
        img_width=192,
        seed=999,
    )
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )

    out_dir = tmp_path / "sanity_fail_out"
    strict_cfg = {
        "max_epochs": 1,
        "stage1_epochs": 1,
        "stage2_epochs": 0,
        "stage3_epochs": 0,
        "sanity_check_min_std": 99999.0,  # Unattainable threshold
        "raise_on_sanity_fail": True,
        "output_dir": str(out_dir),
        "device": "cpu",
        "verbose": False,
    }

    pipeline = CurricularTrainingPipeline()

    with pytest.raises(RuntimeError, match="Sanity Check Failed"):
        pipeline.fit(model=model, dataset=dataset, cfg=strict_cfg)

    # When raise_on_sanity_fail is False, returns sanity_gate_passed=False
    soft_cfg = {
        "max_epochs": 1,
        "stage1_epochs": 1,
        "stage2_epochs": 0,
        "stage3_epochs": 0,
        "sanity_check_min_std": 99999.0,
        "raise_on_sanity_fail": False,
        "output_dir": str(out_dir),
        "device": "cpu",
        "verbose": False,
    }

    result = pipeline.fit(model=model, dataset=dataset, cfg=soft_cfg)
    assert result.sanity_gate_passed is False
    assert result.sanity_metrics["avg_std"] < 99999.0


def test_pipeline_with_dictconfig_and_warm_start(tmp_path: Path) -> None:
    """Verify pipeline works with Hydra DictConfig and loads warm-start VAE weights."""
    src_vae = VAE(latent_dim=64)
    with torch.no_grad():
        src_vae.fc_mu.weight.fill_(0.888)

    ckpt_path = tmp_path / "warm_start_vae.pt"
    torch.save(src_vae.state_dict(), ckpt_path)

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

    dict_cfg: DictConfig = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 0.0,
            "lr_vae_stage1": 0.0,
            "max_epochs": 1,
            "stage1_epochs": 1,
            "stage2_epochs": 0,
            "stage3_epochs": 0,
            "sanity_check_min_std": 0.1,
            "vae_checkpoint": str(ckpt_path),
            "output_dir": str(tmp_path / "dict_cfg_out"),
            "device": "cpu",
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=dict_cfg)

    assert isinstance(result, TrainingResult)
    assert len(result.history["loss"]) == 1
    # Verify VAE was warm-started
    assert torch.allclose(
        model.vae.fc_mu.weight, torch.full_like(model.vae.fc_mu.weight, 0.888)
    )


def test_empty_dataset_raises_value_error() -> None:
    """Verify ValueError is raised when provided dataset contains 0 samples."""
    empty_dataset = DummyDualSourceVideoDataset(
        num_samples=0,
        img_height=108,
        img_width=192,
    )
    model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    pipeline = CurricularTrainingPipeline()
    with pytest.raises(ValueError, match="0 samples"):
        pipeline.fit(model=model, dataset=empty_dataset)

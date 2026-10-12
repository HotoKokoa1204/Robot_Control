"""Module: test_training_pipeline
Stage: Test
Author: KafuuChino
Date: 2026-10-12
Description: Unit and integration tests for CurricularTrainingPipeline and
    TrainingResult.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
from agilab_lib import CurricularTrainingPipeline, TrainingResult
from agilab_lib.datasets.dual_source_dataset import (
    DummyDualSourceVideoDataset,
)
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.models.vae import VAE
from omegaconf import DictConfig, OmegaConf


def test_training_result_dataclass_defaults() -> None:
    """Verify default initialization and attributes of TrainingResult."""
    result = TrainingResult()
    assert result.history == {}
    assert result.checkpoint_paths == {}
    assert result.sanity_gate_passed is True
    assert result.sanity_metrics == {}


def test_training_pipeline_3_stage_synthetic_run(tmp_path: Path) -> None:
    """Verify 3-epoch synthetic run (1 epoch per stage) on CPU.

    Verifies:
    1. Training executes across Stage 1, Stage 2, and Stage 3.
    2. Telemetry tracking in TrainingResult.history across all stages.
    3. Milestone checkpoint exports and reload compatibility.
    4. Sanity gate validation and metrics recording.
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

    out_dir = tmp_path / "pipeline_3_stage_out"
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
    }

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model=model, dataset=dataset, cfg=cfg, device="cpu")

    # 1. Structure verification
    assert isinstance(result, TrainingResult)
    assert result.sanity_gate_passed is True
    assert "avg_std" in result.sanity_metrics
    assert result.sanity_metrics["avg_std"] >= 0.1

    # 2. History telemetry verification
    assert len(result.history["loss"]) == 3
    for key in (
        "loss",
        "recon",
        "fwd",
        "rot",
        "perc",
        "kl",
        "fwd_latent",
        "rot_latent",
        "latent_mse",
    ):
        assert key in result.history
        assert len(result.history[key]) == 3
        assert all(math.isfinite(val) for val in result.history[key])

    # Stage 1 (epoch 0): Recon active, Dynamics inactive
    assert result.history["recon"][0] > 0.0
    assert result.history["fwd"][0] == 0.0
    assert result.history["rot"][0] == 0.0

    # Stage 2 (epoch 1): Recon inactive, Dynamics active
    assert result.history["recon"][1] == 0.0
    assert result.history["fwd"][1] > 0.0
    assert result.history["rot"][1] > 0.0

    # Stage 3 (epoch 2): Both Recon and Dynamics active
    assert result.history["recon"][2] > 0.0
    assert result.history["fwd"][2] > 0.0
    assert result.history["rot"][2] > 0.0

    # 3. Milestone checkpoints verification
    assert "stage1_vae" in result.checkpoint_paths
    assert "stage2_transformers" in result.checkpoint_paths
    assert "final_joint_model" in result.checkpoint_paths

    stage1_ckpt = result.checkpoint_paths["stage1_vae"]
    stage2_ckpt = result.checkpoint_paths["stage2_transformers"]
    final_joint_ckpt = result.checkpoint_paths["final_joint_model"]

    assert stage1_ckpt.exists()
    assert stage2_ckpt.exists()
    assert final_joint_ckpt.exists()

    # Verify Stage 1 milestone loads cleanly into standalone VAE
    loaded_vae = VAE(latent_dim=64)
    loaded_vae.load_state_dict(
        torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    )

    # Verify Stage 2 milestone contains both transformer state dicts
    stage2_payload = torch.load(stage2_ckpt, map_location="cpu", weights_only=True)
    assert "forward_transformer" in stage2_payload
    assert "rotation_transformer" in stage2_payload

    # Verify final milestone loads cleanly into JointNavigationModel
    loaded_model = JointNavigationModel(
        latent_dim=64, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )
    loaded_model.load_state_dict(
        torch.load(final_joint_ckpt, map_location="cpu", weights_only=True)
    )

    # Verify post-stage 3 parameter state: VAE trainable, transformers frozen
    assert model.vae.fc_mu.weight.requires_grad is True
    assert model.forward_transformer.blocks[0].fc1.weight.requires_grad is False
    assert model.rotation_transformer.blocks[0].fc1.weight.requires_grad is False


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
        "verbose": False,
    }

    pipeline = CurricularTrainingPipeline()
    pipeline.fit(
        model=model,
        dataset=dataset,
        cfg=cfg,
        device="cpu",
        epoch_callback=tracking_callback,
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
        "verbose": False,
    }

    pipeline = CurricularTrainingPipeline()

    with pytest.raises(RuntimeError, match="Sanity Check Failed"):
        pipeline.fit(model=model, dataset=dataset, cfg=strict_cfg, device="cpu")

    # When raise_on_sanity_fail is False, returns sanity_gate_passed=False
    soft_cfg = {
        "max_epochs": 1,
        "stage1_epochs": 1,
        "stage2_epochs": 0,
        "stage3_epochs": 0,
        "sanity_check_min_std": 99999.0,
        "raise_on_sanity_fail": False,
        "output_dir": str(out_dir),
        "verbose": False,
    }

    result = pipeline.fit(model=model, dataset=dataset, cfg=soft_cfg, device="cpu")
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
            "verbose": False,
        }
    )

    pipeline = CurricularTrainingPipeline(cfg=dict_cfg)
    result = pipeline.fit(model=model, dataset=dataset, device="cpu")

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

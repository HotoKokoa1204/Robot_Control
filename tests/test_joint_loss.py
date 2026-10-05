"""
Module: test_joint_loss
Stage: Library
Author: KafuuChino
Date: 2026-10-05
Description: Unit tests for JointNavigationLoss three-stage curriculum execution,
    stage decoupling, and gradient isolation.
"""

from typing import Dict
from unittest.mock import MagicMock

from PIL import Image  # isort: skip # noqa: F401
import pytest
import torch
from agilab_lib.models import (
    JointLossOutput,
    JointNavigationLoss,
    JointNavigationModel,
)


def _generate_synthetic_batch(
    batch_size: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Helper to generate synthetic paired multi-branch video batches.

    Args:
        batch_size: Number of synthetic sample frames.
        device: Target compute device.

    Returns:
        Dictionary containing synthetic multi-branch tensors.
    """
    return {
        "recon_frame": torch.rand(batch_size, 3, 108, 192, device=device),
        "fwd_current": torch.rand(batch_size, 3, 108, 192, device=device),
        "fwd_target": torch.rand(batch_size, 3, 108, 192, device=device),
        "fwd_distance": torch.tensor(
            [[1.0], [2.5]], device=device, dtype=torch.float32
        )[:batch_size],
        "rot_current": torch.rand(batch_size, 3, 108, 192, device=device),
        "rot_target": torch.rand(batch_size, 3, 108, 192, device=device),
        "rot_sin_cos": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], device=device, dtype=torch.float32
        )[:batch_size],
    }


def test_stage1_loss_computation_and_no_transformer_passes() -> None:
    """Stage 1: Only Autoencoder recon is evaluated, no transformer passes."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss(alpha_perc=0.5, beta_kl=0.0001, w_recon=1.0)
    batch = _generate_synthetic_batch(batch_size=2)

    # Spy on transformers to ensure they are never invoked in stage 1
    model.forward_transformer.forward = MagicMock(
        side_effect=RuntimeError("Forward transformer called in Stage 1!")
    )
    model.rotation_transformer.forward = MagicMock(
        side_effect=RuntimeError("Rotation transformer called in Stage 1!")
    )

    out = loss_fn.forward_model(model, batch, stage=1)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)

    # Verify reconstruction metrics are active
    assert out.metrics["loss_recon"].item() > 0.0
    assert out.metrics["recon_mse"].item() > 0.0
    assert out.metrics["recon_perc"].item() >= 0.0
    assert out.metrics["recon_kl"].item() >= 0.0

    # Verify dynamics branches have zero loss
    assert out.metrics["loss_fwd"].item() == 0.0
    assert out.metrics["fwd_mse"].item() == 0.0
    assert out.metrics["fwd_perc"].item() == 0.0
    assert out.metrics["fwd_latent_mse"].item() == 0.0
    assert out.metrics["loss_rot"].item() == 0.0
    assert out.metrics["rot_mse"].item() == 0.0
    assert out.metrics["rot_perc"].item() == 0.0
    assert out.metrics["rot_latent_mse"].item() == 0.0

    # Total loss should match w_recon * loss_recon
    assert torch.isclose(out.total_loss, loss_fn.w_recon * out.metrics["loss_recon"])

    # Transformer methods were never invoked
    model.forward_transformer.forward.assert_not_called()
    model.rotation_transformer.forward.assert_not_called()


def test_stage2_loss_computation() -> None:
    """Stage 2: Reconstruction is zero, forward and rotation dynamics are active."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss(w_fwd=1.0, w_rot=1.0, w_latent=1.0, alpha_perc=0.5)
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=2)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)

    # Reconstruction branch must be 0
    assert out.metrics["loss_recon"].item() == 0.0
    assert out.metrics["recon_mse"].item() == 0.0
    assert out.metrics["recon_perc"].item() == 0.0
    assert out.metrics["recon_kl"].item() == 0.0

    # Dynamics branches must be non-zero
    assert out.metrics["loss_fwd"].item() > 0.0
    assert out.metrics["fwd_mse"].item() > 0.0
    assert out.metrics["fwd_latent_mse"].item() > 0.0
    assert out.metrics["loss_rot"].item() > 0.0
    assert out.metrics["rot_mse"].item() > 0.0
    assert out.metrics["rot_latent_mse"].item() > 0.0

    # Total loss equals w_fwd * loss_fwd + w_rot * loss_rot
    expected_total = (
        loss_fn.w_fwd * out.metrics["loss_fwd"]
        + loss_fn.w_rot * out.metrics["loss_rot"]
    )
    assert torch.isclose(out.total_loss, expected_total)


def test_stage3_loss_computation() -> None:
    """Test Stage 3: All branches (reconstruction, forward, rotation) are active."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss(
        w_recon=1.0, w_fwd=1.0, w_rot=1.0, w_latent=1.0, alpha_perc=0.5
    )
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=3)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)

    # All branches must have non-zero metrics
    assert out.metrics["loss_recon"].item() > 0.0
    assert out.metrics["loss_fwd"].item() > 0.0
    assert out.metrics["loss_rot"].item() > 0.0

    expected_total = (
        loss_fn.w_recon * out.metrics["loss_recon"]
        + loss_fn.w_fwd * out.metrics["loss_fwd"]
        + loss_fn.w_rot * out.metrics["loss_rot"]
    )
    assert torch.isclose(out.total_loss, expected_total)


def test_stage1_recon_only_backwards_compatibility() -> None:
    """Verify stage1_recon_only flag backwards-compatibility."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    # stage1_recon_only=True maps to Stage 1
    out_s1 = loss_fn.forward_model(model, batch, stage1_recon_only=True)
    assert out_s1.metrics["loss_recon"].item() > 0.0
    assert out_s1.metrics["loss_fwd"].item() == 0.0
    assert out_s1.metrics["loss_rot"].item() == 0.0

    # stage1_recon_only=False maps to Stage 3
    out_s3 = loss_fn.forward_model(model, batch, stage1_recon_only=False)
    assert out_s3.metrics["loss_recon"].item() > 0.0
    assert out_s3.metrics["loss_fwd"].item() > 0.0
    assert out_s3.metrics["loss_rot"].item() > 0.0


def test_stage2_gradient_isolation() -> None:
    """Stage 2 gradient isolation: transformers get grads, frozen VAE gets none."""
    model = JointNavigationModel()
    with torch.no_grad():
        for blk in model.forward_transformer.blocks:
            torch.nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
            torch.nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)
        for blk in model.rotation_transformer.blocks:
            torch.nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
            torch.nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)

    model.vae.requires_grad_(False)
    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=2)
    model.zero_grad()
    out.total_loss.backward()

    # Transformers receive non-None gradients
    fwd_grads = [
        p.grad for p in model.forward_transformer.parameters() if p.requires_grad
    ]
    assert len(fwd_grads) > 0
    assert any(g is not None and torch.any(g != 0) for g in fwd_grads)

    rot_grads = [
        p.grad for p in model.rotation_transformer.parameters() if p.requires_grad
    ]
    assert len(rot_grads) > 0
    assert any(g is not None and torch.any(g != 0) for g in rot_grads)

    # Frozen VAE receives NO gradients
    for name, param in model.vae.named_parameters():
        assert (
            param.grad is None
        ), f"VAE parameter {name} received unexpected gradient in stage 2"


def test_stage3_gradient_isolation() -> None:
    """Stage 3 gradient isolation: VAE gets grads when transformers are frozen."""
    model = JointNavigationModel()
    model.forward_transformer.requires_grad_(False)
    model.rotation_transformer.requires_grad_(False)

    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=3)
    model.zero_grad()
    out.total_loss.backward()

    # VAE encoder and decoder receive non-None gradients
    assert model.vae.encoder_cnn[0].weight.grad is not None
    assert model.vae.fc_mu.weight.grad is not None
    assert model.vae.fc_logvar.weight.grad is not None
    assert model.vae.fc_decode.weight.grad is not None
    assert model.vae.decoder_cnn[0].weight.grad is not None

    # Frozen transformers receive NO gradients
    for name, param in model.forward_transformer.named_parameters():
        assert (
            param.grad is None
        ), f"forward_transformer {name} received grad in stage 3"
    for name, param in model.rotation_transformer.named_parameters():
        assert (
            param.grad is None
        ), f"rotation_transformer {name} received grad in stage 3"


def test_invalid_stage_error() -> None:
    """Test that specifying an invalid stage raises ValueError."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    for invalid_stage in (0, 4, -1, 99):
        with pytest.raises(ValueError, match=r"Invalid stage"):
            loss_fn.forward_model(model, batch, stage=invalid_stage)


def test_missing_batch_keys_per_stage() -> None:
    """Test that missing required batch keys raises KeyError."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()

    # Stage 1 requires recon_frame
    with pytest.raises(KeyError, match=r"Batch must contain 'recon_frame'"):
        loss_fn.forward_model(model, {}, stage=1)

    # Stage 3 requires recon_frame
    with pytest.raises(KeyError, match=r"Batch must contain 'recon_frame'"):
        loss_fn.forward_model(model, {}, stage=3)

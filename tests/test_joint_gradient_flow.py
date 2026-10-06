"""
Module: test_joint_gradient_flow
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Unit tests for JointNavigationLoss and end-to-end gradient propagation
    across Encoder, Decoder, Forward Transformer, and Rotation Transformer.
"""

from typing import Any, Dict

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


def test_joint_gradient_flow_primary_seam() -> None:
    """Primary Seam Test: Verify end-to-end gradient flow through all 4 sub-networks.

    Executes a single loss.backward() pass and asserts that non-zero, finite
    gradients populate simultaneously across:
    1. Encoder (vae.encoder_cnn, vae.fc_mu, vae.fc_logvar)
    2. Visual Dynamics Decoder (vae.fc_decode, vae.decoder_cnn)
    3. Forward Latent Transformer (forward_transformer)
    4. Rotation Latent Transformer (rotation_transformer)
    Also verifies frozen VGG-16 parameters receive no gradients.
    """
    model = JointNavigationModel()
    # Initialize non-zero residual weights to verify end-to-end gradient propagation
    # across deep feedforward layers of residual blocks
    with torch.no_grad():
        for blk in model.forward_transformer.blocks:
            torch.nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
            torch.nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)
        for blk in model.rotation_transformer.blocks:
            torch.nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
            torch.nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)

    loss_fn = JointNavigationLoss(alpha_perc=0.5, beta_kl=0.0001, w_latent=1.0)

    batch = _generate_synthetic_batch(batch_size=2)

    # Execute all 3 branches
    recon_x, mu_t, logvar_t = model.forward_reconstruction(batch["recon_frame"])
    pred_fwd, pred_fwd_latent, _ = model.forward_translation(
        x_t=batch["fwd_current"],
        distance_meters=batch["fwd_distance"],
    )
    pred_rot, pred_rot_latent, _ = model.forward_rotation(
        x_t=batch["rot_current"],
        sin_cos=batch["rot_sin_cos"],
    )
    mu_fwd_target = model.vae.encode(batch["fwd_target"])[0]
    mu_rot_target = model.vae.encode(batch["rot_target"])[0]

    # Compute loss via explicit keyword arguments
    total_loss, metrics = loss_fn(
        recon_x=recon_x,
        x_t=batch["recon_frame"],
        mu_t=mu_t,
        logvar_t=logvar_t,
        pred_fwd=pred_fwd,
        target_fwd=batch["fwd_target"],
        pred_rot=pred_rot,
        target_rot=batch["rot_target"],
        pred_fwd_latent=pred_fwd_latent,
        mu_fwd_target=mu_fwd_target,
        pred_rot_latent=pred_rot_latent,
        mu_rot_target=mu_rot_target,
    )

    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    # Backpropagate gradients
    model.zero_grad()
    total_loss.backward()

    # 1. Verify Encoder gradients
    for name, param in model.vae.encoder_cnn.named_parameters():
        assert param.grad is not None, f"Encoder CNN param {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"Encoder CNN {name} non-finite grad"
        assert torch.any(param.grad != 0), f"Encoder CNN {name} has all-zero grad"

    for name, param in model.vae.fc_mu.named_parameters():
        assert param.grad is not None, f"fc_mu param {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"fc_mu {name} non-finite grad"
        assert torch.any(param.grad != 0), f"fc_mu {name} has all-zero grad"

    for name, param in model.vae.fc_logvar.named_parameters():
        assert param.grad is not None, f"fc_logvar param {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"fc_logvar {name} non-finite grad"
        assert torch.any(param.grad != 0), f"fc_logvar {name} has all-zero grad"

    # 2. Verify Visual Dynamics Decoder gradients
    for name, param in model.vae.fc_decode.named_parameters():
        assert param.grad is not None, f"fc_decode param {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"fc_decode {name} non-finite grad"
        assert torch.any(param.grad != 0), f"fc_decode {name} has all-zero grad"

    for name, param in model.vae.decoder_cnn.named_parameters():
        assert param.grad is not None, f"decoder_cnn param {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"decoder_cnn {name} non-finite grad"
        assert torch.any(param.grad != 0), f"decoder_cnn {name} has all-zero grad"

    # 3. Verify Forward Latent Transformer gradients
    for name, param in model.forward_transformer.named_parameters():
        assert param.grad is not None, f"forward_transformer {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"forward {name} non-finite grad"
        assert torch.any(param.grad != 0), f"forward {name} all-zero grad"

    # 4. Verify Rotation Latent Transformer gradients
    for name, param in model.rotation_transformer.named_parameters():
        assert param.grad is not None, f"rotation_transformer {name} missing grad"
        assert torch.isfinite(param.grad).all(), f"rotation {name} non-finite grad"
        assert torch.any(param.grad != 0), f"rotation {name} all-zero grad"

    # 5. Verify Frozen VGG-16 parameters received no gradients
    for name, param in loss_fn.perceptual_loss.named_parameters():
        assert param.requires_grad is False, f"VGG param {name} is not frozen"
        assert param.grad is None, f"VGG param {name} received non-null grad"


def test_joint_loss_via_forward_model_and_batch_dict() -> None:
    """Test integrated forward execution using model and batch dictionary."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()

    batch = _generate_synthetic_batch(batch_size=2)

    # Calling loss_fn(model, batch, stage=3) directly
    out = loss_fn(model, batch, stage=3)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)
    assert out.total_loss.item() > 0.0

    model.zero_grad()
    out.total_loss.backward()

    # Verify gradients reached all 4 networks
    assert model.vae.encoder_cnn[0].weight.grad is not None
    assert model.vae.decoder_cnn[0].weight.grad is not None
    assert model.forward_transformer.blocks[0].fc3.weight.grad is not None
    assert model.rotation_transformer.blocks[0].fc3.weight.grad is not None


def test_joint_loss_via_predictions_and_targets_dict() -> None:
    """Test computing loss from separate predictions and targets dictionaries."""
    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()

    batch = _generate_synthetic_batch(batch_size=2)
    recon_x, mu_t, logvar_t = model.forward_reconstruction(batch["recon_frame"])
    pred_fwd, pred_fwd_latent, _ = model.forward_translation(
        x_t=batch["fwd_current"],
        distance_meters=batch["fwd_distance"],
    )
    pred_rot, pred_rot_latent, _ = model.forward_rotation(
        x_t=batch["rot_current"],
        sin_cos=batch["rot_sin_cos"],
    )
    mu_fwd_target = model.vae.encode(batch["fwd_target"])[0]
    mu_rot_target = model.vae.encode(batch["rot_target"])[0]

    preds: Dict[str, Any] = {
        "recon_x": recon_x,
        "mu_t": mu_t,
        "logvar_t": logvar_t,
        "pred_fwd": pred_fwd,
        "pred_rot": pred_rot,
        "pred_fwd_latent": pred_fwd_latent,
        "pred_rot_latent": pred_rot_latent,
    }

    targets: Dict[str, Any] = {
        "recon_frame": batch["recon_frame"],
        "fwd_target": batch["fwd_target"],
        "rot_target": batch["rot_target"],
        "mu_fwd_target": mu_fwd_target,
        "mu_rot_target": mu_rot_target,
    }

    total_loss, metrics = loss_fn(preds, targets)
    assert torch.isfinite(total_loss)
    assert "loss_recon" in metrics
    assert "loss_fwd" in metrics
    assert "loss_rot" in metrics
    assert "fwd_latent_mse" in metrics
    assert "rot_latent_mse" in metrics

    model.zero_grad()
    total_loss.backward()
    assert model.vae.encoder_cnn[0].weight.grad is not None


def test_joint_loss_single_container_and_batch_mock() -> None:
    """Test loss computation with mock DualSourceBatch supporting attribute access."""

    class MockDualSourceBatch(dict):
        """Mock DualSourceBatch supporting property and key access."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

    raw_batch = _generate_synthetic_batch(batch_size=2)
    mock_batch = MockDualSourceBatch(**raw_batch)

    model = JointNavigationModel()
    loss_fn = JointNavigationLoss()

    out = loss_fn.forward_model(model, mock_batch)
    assert torch.isfinite(out.total_loss)
    assert isinstance(out, tuple)
    assert isinstance(out, JointLossOutput)


def test_configurable_loss_weights() -> None:
    """Test configurable branch weights, alpha_perc, and beta_kl settings."""
    model = JointNavigationModel()
    batch = _generate_synthetic_batch(batch_size=2)

    recon_x, mu_t, logvar_t = model.forward_reconstruction(batch["recon_frame"])
    pred_fwd, _, _ = model.forward_translation(
        x_t=batch["fwd_current"],
        distance_meters=batch["fwd_distance"],
    )
    pred_rot, _, _ = model.forward_rotation(
        x_t=batch["rot_current"],
        sin_cos=batch["rot_sin_cos"],
    )

    # 1. Test perceptual loss disabled (alpha_perc = 0.0)
    loss_fn_no_perc = JointNavigationLoss(alpha_perc=0.0)
    _, metrics_no_perc = loss_fn_no_perc(
        recon_x=recon_x,
        x_t=batch["recon_frame"],
        mu_t=mu_t,
        logvar_t=logvar_t,
        pred_fwd=pred_fwd,
        target_fwd=batch["fwd_target"],
        pred_rot=pred_rot,
        target_rot=batch["rot_target"],
    )
    assert torch.isclose(metrics_no_perc["recon_perc"], torch.tensor(0.0))
    assert torch.isclose(metrics_no_perc["fwd_perc"], torch.tensor(0.0))
    assert torch.isclose(metrics_no_perc["rot_perc"], torch.tensor(0.0))

    # 2. Test KL divergence disabled (beta_kl = 0.0)
    loss_fn_no_kl = JointNavigationLoss(beta_kl=0.0)
    _, metrics_no_kl = loss_fn_no_kl(
        recon_x=recon_x,
        x_t=batch["recon_frame"],
        mu_t=mu_t,
        logvar_t=logvar_t,
        pred_fwd=pred_fwd,
        target_fwd=batch["fwd_target"],
        pred_rot=pred_rot,
        target_rot=batch["rot_target"],
    )
    assert torch.isclose(metrics_no_kl["recon_kl"], torch.tensor(0.0))

    # 3. Test branch isolation via weights (w_fwd = 0.0, w_rot = 0.0)
    loss_fn_only_recon = JointNavigationLoss(w_fwd=0.0, w_rot=0.0)
    total_loss_recon, metrics_recon = loss_fn_only_recon(
        recon_x=recon_x,
        x_t=batch["recon_frame"],
        mu_t=mu_t,
        logvar_t=logvar_t,
        pred_fwd=pred_fwd,
        target_fwd=batch["fwd_target"],
        pred_rot=pred_rot,
        target_rot=batch["rot_target"],
    )
    assert torch.isclose(total_loss_recon, metrics_recon["loss_recon"])

    # Gradients should NOT propagate into forward or rotation transformers
    model.zero_grad()
    total_loss_recon.backward()
    for param in model.forward_transformer.parameters():
        assert param.grad is None
    for param in model.rotation_transformer.parameters():
        assert param.grad is None
    # But should propagate into VAE
    assert model.vae.encoder_cnn[0].weight.grad is not None


def test_metrics_dictionary_contents() -> None:
    """Verify keys, telemetry metrics, and multi-format access on JointLossOutput."""
    loss_fn = JointNavigationLoss(w_fwd=1.5, w_rot=2.0)
    batch = _generate_synthetic_batch(batch_size=2)

    total_loss, metrics = loss_fn(
        recon_x=batch["recon_frame"],
        x_t=batch["recon_frame"],
        mu_t=torch.zeros(2, 512),
        logvar_t=torch.zeros(2, 512),
        pred_fwd=batch["fwd_current"],
        target_fwd=batch["fwd_target"],
        pred_rot=batch["rot_current"],
        target_rot=batch["rot_target"],
    )

    required_keys = [
        "total_loss",
        "loss",
        "loss_recon",
        "recon_mse",
        "recon_perc",
        "recon_kl",
        "loss_fwd",
        "fwd_mse",
        "fwd_perc",
        "fwd_latent_mse",
        "loss_rot",
        "rot_mse",
        "rot_perc",
        "rot_latent_mse",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
        assert torch.isfinite(metrics[key]), f"Metric {key} is non-finite"
        assert metrics[key].item() >= 0.0, f"Metric {key} is negative"

    # Verify linear combination:
    # total_loss == loss_recon + 1.5 * loss_fwd + 2.0 * loss_rot
    expected_total = (
        metrics["loss_recon"] + 1.5 * metrics["loss_fwd"] + 2.0 * metrics["loss_rot"]
    )
    assert torch.isclose(total_loss, expected_total, atol=1e-5)

    # Verify JointLossOutput container behavior
    out = JointLossOutput(total_loss, metrics)
    # 1. Unpacking
    l_unpacked, m_unpacked = out
    assert l_unpacked is total_loss
    assert m_unpacked is metrics

    # 2. Attribute access
    assert out.total_loss is total_loss
    assert out.loss is total_loss
    assert out.metrics is metrics

    # 3. Dictionary-style indexing
    assert out["total_loss"] is total_loss
    assert out["loss_recon"] is metrics["loss_recon"]
    assert out.get("loss_fwd") is metrics["loss_fwd"]
    assert out.get("non_existent", "default") == "default"


def test_single_branch_execution() -> None:
    """Verify that JointNavigationLoss can operate on partial branch inputs."""
    loss_fn = JointNavigationLoss()
    frame_a = torch.rand(2, 3, 108, 192)
    frame_b = torch.rand(2, 3, 108, 192)

    # 1. Only reconstruction branch
    out_recon = loss_fn(
        recon_x=frame_a,
        x_t=frame_b,
        mu_t=torch.randn(2, 512),
        logvar_t=torch.randn(2, 512),
    )
    assert torch.isfinite(out_recon.total_loss)
    assert out_recon.metrics["loss_recon"].item() > 0.0
    assert torch.isclose(out_recon.metrics["loss_fwd"], torch.tensor(0.0))
    assert torch.isclose(out_recon.metrics["loss_rot"], torch.tensor(0.0))

    # 2. Only forward dynamics branch
    out_fwd = loss_fn(pred_fwd=frame_a, target_fwd=frame_b)
    assert torch.isfinite(out_fwd.total_loss)
    assert out_fwd.metrics["loss_fwd"].item() > 0.0
    assert torch.isclose(out_fwd.metrics["loss_recon"], torch.tensor(0.0))

    # 3. Only rotation dynamics branch
    out_rot = loss_fn(pred_rot=frame_a, target_rot=frame_b)
    assert torch.isfinite(out_rot.total_loss)
    assert out_rot.metrics["loss_rot"].item() > 0.0
    assert torch.isclose(out_rot.metrics["loss_recon"], torch.tensor(0.0))


def test_input_validation_errors() -> None:
    """Test error handling on shape mismatches and empty inputs."""
    loss_fn = JointNavigationLoss()
    frame_a = torch.rand(2, 3, 108, 192)
    frame_wrong_shape = torch.rand(2, 3, 64, 64)

    # Reconstruction mismatch
    with pytest.raises(ValueError, match="Reconstruction shape mismatch"):
        loss_fn(recon_x=frame_a, x_t=frame_wrong_shape)

    # Forward dynamics mismatch
    with pytest.raises(ValueError, match="Forward dynamics shape mismatch"):
        loss_fn(pred_fwd=frame_a, target_fwd=frame_wrong_shape)

    # Rotation dynamics mismatch
    with pytest.raises(ValueError, match="Rotation dynamics shape mismatch"):
        loss_fn(pred_rot=frame_a, target_rot=frame_wrong_shape)

    # Latent dimension mismatch
    with pytest.raises(ValueError, match="Latent vector shape mismatch"):
        loss_fn(
            recon_x=frame_a,
            x_t=frame_a,
            mu_t=torch.randn(2, 512),
            logvar_t=torch.randn(2, 256),
        )

    # Empty inputs
    with pytest.raises(ValueError, match="requires at least one active branch"):
        loss_fn()

    # Missing batch key
    model = JointNavigationModel()
    with pytest.raises(KeyError, match="must contain 'recon_frame'"):
        loss_fn.forward_model(model, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_gradient_flow() -> None:
    """Verify end-to-end gradient flow when executed on CUDA device."""
    device = torch.device("cuda")
    model = JointNavigationModel().to(device)
    loss_fn = JointNavigationLoss().to(device)

    batch = _generate_synthetic_batch(batch_size=2, device=device)
    out = loss_fn(model, batch, stage=3)

    assert out.total_loss.is_cuda
    assert torch.isfinite(out.total_loss)

    model.zero_grad()
    out.total_loss.backward()

    # Check CUDA gradients
    assert model.vae.encoder_cnn[0].weight.grad is not None
    assert model.vae.encoder_cnn[0].weight.grad.is_cuda
    assert model.forward_transformer.blocks[0].fc3.weight.grad is not None
    assert model.forward_transformer.blocks[0].fc3.weight.grad.is_cuda
    assert model.rotation_transformer.blocks[0].fc3.weight.grad is not None
    assert model.rotation_transformer.blocks[0].fc3.weight.grad.is_cuda


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

    # Freeze VAE
    model.vae.requires_grad_(False)

    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=2)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)
    assert out.total_loss.item() > 0.0

    model.zero_grad()
    out.total_loss.backward()

    # Transformers have requires_grad=True and must receive non-None gradients
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

    # VAE parameters must receive NO gradients
    for name, param in model.vae.named_parameters():
        assert (
            param.grad is None
        ), f"VAE parameter {name} received unexpected gradient in stage 2"


def test_stage3_gradient_isolation() -> None:
    """Stage 3 gradient isolation: VAE gets grads when transformers are frozen."""
    model = JointNavigationModel()
    # Freeze transformers
    model.forward_transformer.requires_grad_(False)
    model.rotation_transformer.requires_grad_(False)

    loss_fn = JointNavigationLoss()
    batch = _generate_synthetic_batch(batch_size=2)

    out = loss_fn.forward_model(model, batch, stage=3)
    assert isinstance(out, JointLossOutput)
    assert torch.isfinite(out.total_loss)
    assert out.total_loss.item() > 0.0

    model.zero_grad()
    out.total_loss.backward()

    # VAE encoder and decoder receive gradients
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

"""
Module: test_decoupled_transformers
Stage: Library
Author: KafuuChino
Date: 2026-09-14
Description: Unit tests for decoupled latent transformers with chained execution.
"""

from pathlib import Path

import pytest
import torch
from agilab_lib.models.rlt import (
    ChainedLatentTransformer,
    ForwardLatentTransformer,
    RotationLatentTransformer,
)


def test_rotation_latent_transformer_shapes() -> None:
    """Test RotationLatentTransformer output dimensions with various input forms."""
    batch_size = 4
    latent_dim = 128
    model = RotationLatentTransformer(
        latent_dim=latent_dim,
        hidden_dim=64,
        num_blocks=5,
        block_inner_dim=64,
    )
    assert len(model.blocks) == 5

    z = torch.randn(batch_size, latent_dim)

    # 1. 2D tensor angle_deg (B, 1)
    angle_2d = torch.tensor([[15.0], [-30.0], [45.0], [90.0]])
    out_2d = model(z, angle_deg=angle_2d)
    assert out_2d.shape == (batch_size, latent_dim)

    # 2. 1D tensor angle_deg (B,)
    angle_1d = torch.tensor([10.0, -20.0, 30.0, -40.0])
    out_1d = model(z, angle_deg=angle_1d)
    assert out_1d.shape == (batch_size, latent_dim)

    # 3. Scalar float angle
    out_scalar = model(z, angle_deg=45.0)
    assert out_scalar.shape == (batch_size, latent_dim)

    # 4. Direct 2D sin_cos tensor (B, 2)
    sin_cos = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071], [-1.0, 0.0]])
    out_sc = model(z, sin_cos=sin_cos)
    assert out_sc.shape == (batch_size, latent_dim)

    # 5. Default angle (None)
    out_default = model(z)
    assert out_default.shape == (batch_size, latent_dim)


def test_rotation_latent_transformer_gradients() -> None:
    """Test gradient propagation through RotationLatentTransformer."""
    model = RotationLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=5,
        block_inner_dim=32,
    )
    z = torch.randn(2, 64, requires_grad=True)
    angle = torch.tensor([[30.0], [-45.0]])

    out = model(z, angle_deg=angle)
    loss = out.sum()
    loss.backward()

    assert z.grad is not None
    assert torch.any(z.grad != 0.0)
    for p in model.parameters():
        assert p.grad is not None


def test_forward_latent_transformer_shapes() -> None:
    """Test ForwardLatentTransformer output dimensions with 2 residual blocks."""
    batch_size = 4
    latent_dim = 128
    model = ForwardLatentTransformer(
        latent_dim=latent_dim,
        hidden_dim=64,
        num_blocks=2,
        block_inner_dim=64,
    )
    assert len(model.blocks) == 2

    z = torch.randn(batch_size, latent_dim)

    # 1. 2D tensor distance_meters (B, 1)
    dist_2d = torch.tensor([[0.5], [1.2], [2.0], [0.1]])
    out_2d = model(z, distance_meters=dist_2d)
    assert out_2d.shape == (batch_size, latent_dim)

    # 2. 1D tensor distance_meters (B,)
    dist_1d = torch.tensor([0.2, 0.4, 0.6, 0.8])
    out_1d = model(z, distance_meters=dist_1d)
    assert out_1d.shape == (batch_size, latent_dim)

    # 3. Scalar float distance
    out_scalar = model(z, distance_meters=1.5)
    assert out_scalar.shape == (batch_size, latent_dim)

    # 4. Default distance (None)
    out_default = model(z)
    assert out_default.shape == (batch_size, latent_dim)


def test_forward_latent_transformer_gradients() -> None:
    """Test gradient propagation through ForwardLatentTransformer."""
    model = ForwardLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    z = torch.randn(2, 64, requires_grad=True)
    dist = torch.tensor([[1.0], [2.0]])

    out = model(z, distance_meters=dist)
    loss = out.sum()
    loss.backward()

    assert z.grad is not None
    assert torch.any(z.grad != 0.0)
    for p in model.parameters():
        assert p.grad is not None


def test_chained_latent_transformer_execution_order() -> None:
    """Verify non-commutativity: rotate_first != forward_first when theta, d != 0."""
    torch.manual_seed(42)
    model = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
    )
    z = torch.randn(3, 64)
    theta = torch.tensor([[45.0], [-60.0], [90.0]])
    dist = torch.tensor([[1.5], [0.8], [2.0]])

    out_rotate_first = model(
        latent=z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order="rotate_first",
    )
    out_forward_first = model(
        latent=z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order="forward_first",
    )

    assert out_rotate_first.shape == (3, 64)
    assert out_forward_first.shape == (3, 64)
    # Non-commutativity in SE(2): Compound motion order yields distinct visual states
    assert not torch.allclose(out_rotate_first, out_forward_first, atol=1e-3)


def test_single_action_invariance() -> None:
    """Verify single-action invariance when either distance or rotation is zero."""
    model = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
    )
    z = torch.randn(4, 64)

    # 1. Pure rotation (distance = 0.0): both orders must yield identical results
    theta = torch.tensor([[30.0], [-45.0], [15.0], [60.0]])
    zero_dist = torch.zeros(4, 1)

    out_rf_rot = model(
        z, angle_deg=theta, distance_meters=zero_dist, execution_order="rotate_first"
    )
    out_ff_rot = model(
        z, angle_deg=theta, distance_meters=zero_dist, execution_order="forward_first"
    )
    assert torch.allclose(out_rf_rot, out_ff_rot, atol=1e-6)

    # Also check with distance_meters=None
    out_none_dist_rf = model(
        z, angle_deg=theta, distance_meters=None, execution_order="rotate_first"
    )
    assert torch.allclose(out_rf_rot, out_none_dist_rf, atol=1e-6)

    # 2. Pure forward (angle = 0.0): both orders must yield identical results
    dist = torch.tensor([[1.0], [2.5], [0.5], [1.8]])
    zero_angle = torch.zeros(4, 1)

    out_rf_fwd = model(
        z, angle_deg=zero_angle, distance_meters=dist, execution_order="rotate_first"
    )
    out_ff_fwd = model(
        z, angle_deg=zero_angle, distance_meters=dist, execution_order="forward_first"
    )
    assert torch.allclose(out_rf_fwd, out_ff_fwd, atol=1e-6)

    # 3. No motion (angle = 0, distance = 0): returns identical input latent
    out_identity = model(
        z, angle_deg=0.0, distance_meters=0.0, execution_order="rotate_first"
    )
    assert torch.allclose(out_identity, z, atol=1e-6)


def test_chained_latent_transformer_return_intermediate() -> None:
    """Verify intermediate latent state extraction during chained execution."""
    model = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
    )
    z = torch.randn(2, 64)
    theta = torch.tensor([[25.0], [-15.0]])
    dist = torch.tensor([[1.0], [2.0]])

    # Rotate first
    out_rf, mid_rf = model(
        z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order="rotate_first",
        return_intermediate=True,
    )
    expected_mid_rf = model.rotation_model(z, angle_deg=theta)
    expected_out_rf = model.forward_model(expected_mid_rf, distance_meters=dist)
    assert torch.allclose(mid_rf, expected_mid_rf, atol=1e-6)
    assert torch.allclose(out_rf, expected_out_rf, atol=1e-6)

    # Forward first
    out_ff, mid_ff = model(
        z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order="forward_first",
        return_intermediate=True,
    )
    expected_mid_ff = model.forward_model(z, distance_meters=dist)
    expected_out_ff = model.rotation_model(expected_mid_ff, angle_deg=theta)
    assert torch.allclose(mid_ff, expected_mid_ff, atol=1e-6)
    assert torch.allclose(out_ff, expected_out_ff, atol=1e-6)


def test_chained_latent_transformer_invalid_order() -> None:
    """Verify ValueError is raised on unsupported execution order."""
    model = ChainedLatentTransformer(latent_dim=64, hidden_dim=32)
    z = torch.randn(1, 64)
    with pytest.raises(ValueError, match="Unsupported execution_order"):
        model(z, execution_order="invalid_order")


def test_chained_latent_transformer_checkpoint_loading(tmp_path: Path) -> None:
    """Verify loading weights into submodels via checkpoint paths fixture.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    rot_model = RotationLatentTransformer(
        latent_dim=64, hidden_dim=32, num_blocks=5, block_inner_dim=32
    )
    fwd_model = ForwardLatentTransformer(
        latent_dim=64, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )

    rot_ckpt = tmp_path / "rot.pt"
    fwd_ckpt = tmp_path / "fwd.pt"
    torch.save(rot_model.state_dict(), rot_ckpt)
    torch.save(fwd_model.state_dict(), fwd_ckpt)

    chained = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
        rotation_checkpoint=rot_ckpt,
        forward_checkpoint=fwd_ckpt,
    )

    z = torch.randn(2, 64)
    theta = torch.tensor([[10.0], [20.0]])
    assert torch.allclose(
        chained.rotation_model(z, angle_deg=theta),
        rot_model(z, angle_deg=theta),
        atol=1e-6,
    )

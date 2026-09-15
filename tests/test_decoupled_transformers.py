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
from agilab_lib.datasets.latent_dataset import CachedLatentDataset
from agilab_lib.models.rlt import (
    BaseLatentTransformer,
    ChainedLatentTransformer,
    ExecutionOrder,
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


def test_chained_latent_transformer_mixed_batch_invariance() -> None:
    """Verify predictions are batch-independent across mixed zero/non-zero actions."""
    model = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
    )
    # Batch with mixed motion profiles:
    # 0: pure forward (angle=0, dist=1.5)
    # 1: pure rotation (angle=30, dist=0)
    # 2: no motion (angle=0, dist=0)
    # 3: compound motion (angle=45, dist=2.0)
    z = torch.randn(4, 64)
    angles = torch.tensor([[0.0], [30.0], [0.0], [45.0]])
    dists = torch.tensor([[1.5], [0.0], [0.0], [2.0]])

    for order in (ExecutionOrder.ROTATE_FIRST, ExecutionOrder.FORWARD_FIRST):
        out_batch = model(
            z,
            angle_deg=angles,
            distance_meters=dists,
            execution_order=order,
        )

        for i in range(4):
            out_isolated = model(
                z[i : i + 1],
                angle_deg=angles[i : i + 1],
                distance_meters=dists[i : i + 1],
                execution_order=order,
            )
            assert torch.allclose(out_batch[i : i + 1], out_isolated, atol=1e-6)


def test_forward_first_intermediate_latent_zero_dist() -> None:
    """Verify intermediate latent state in forward_first with zero distance."""
    model = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        rotation_num_blocks=5,
        forward_num_blocks=2,
        block_inner_dim=32,
    )
    z = torch.randn(3, 64)
    theta = torch.tensor([[25.0], [-15.0], [40.0]])
    zero_dist = torch.zeros(3, 1)

    out_ff, mid_ff = model(
        z,
        angle_deg=theta,
        distance_meters=zero_dist,
        execution_order=ExecutionOrder.FORWARD_FIRST,
        return_intermediate=True,
    )

    # In forward_first when d=0, the first step (forward) produces no displacement.
    # Therefore, mid_ff must strictly equal the starting latent z.
    assert torch.allclose(mid_ff, z, atol=1e-6)
    # And the final state out_ff reflects the rotation applied to z.
    assert not torch.allclose(mid_ff, out_ff, atol=1e-3)


def test_execution_order_enum_and_string_equivalence() -> None:
    """Verify that ExecutionOrder Enum and string literals produce identical results."""
    model = ChainedLatentTransformer(latent_dim=64, hidden_dim=32)
    z = torch.randn(2, 64)
    theta = torch.tensor([[15.0], [-20.0]])
    dist = torch.tensor([[0.5], [1.0]])

    out_enum_rf = model(
        z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order=ExecutionOrder.ROTATE_FIRST,
    )
    out_str_rf = model(
        z, angle_deg=theta, distance_meters=dist, execution_order="rotate_first"
    )
    assert torch.allclose(out_enum_rf, out_str_rf, atol=1e-6)

    out_enum_ff = model(
        z,
        angle_deg=theta,
        distance_meters=dist,
        execution_order=ExecutionOrder.FORWARD_FIRST,
    )
    out_str_ff = model(
        z, angle_deg=theta, distance_meters=dist, execution_order="forward_first"
    )
    assert torch.allclose(out_enum_ff, out_str_ff, atol=1e-6)


def test_base_latent_transformer_shapes() -> None:
    """Test BaseLatentTransformer directly with arbitrary conditioning dimension."""
    base = BaseLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=3,
        block_inner_dim=32,
        cond_dim=4,
    )
    x = torch.randn(3, 64)
    cond = torch.randn(3, 4)
    out = base._forward_blocks(x, cond)
    assert out.shape == (3, 64)


def test_chained_latent_transformer_keyframe_deduplication() -> None:
    """Test keyframe deduplication logic in compound chained transformer execution."""
    model = ChainedLatentTransformer(latent_dim=64, hidden_dim=32)
    z_start = torch.randn(1, 64)

    def assemble_keyframes(
        angle: float, dist: float, order: str = "rotate_first"
    ) -> torch.Tensor:
        with torch.no_grad():
            res = model(
                latent=z_start,
                angle_deg=angle,
                distance_meters=dist,
                execution_order=order,
                return_intermediate=True,
            )
            z_final, z_intermediate = res
        key_latents = [z_start]
        if not torch.allclose(z_intermediate, z_start, atol=1e-4):
            key_latents.append(z_intermediate)
        if not torch.allclose(z_final, key_latents[-1], atol=1e-4):
            key_latents.append(z_final)
        if len(key_latents) == 1:
            key_latents.append(z_start)
        return torch.cat(key_latents, dim=0)

    # 1. Zero motion: both intermediate and final are identical to start
    kf_zero = assemble_keyframes(0.0, 0.0)
    assert kf_zero.shape == (2, 64)
    assert torch.allclose(kf_zero[0], kf_zero[1])

    # 2. Pure rotation (dist=0.0): final is identical to intermediate
    # This results in exactly 2 keyframes
    kf_rot = assemble_keyframes(45.0, 0.0)
    assert kf_rot.shape == (2, 64)
    assert not torch.allclose(kf_rot[0], kf_rot[1], atol=1e-3)

    # 3. Pure forward (angle=0.0): intermediate is identical to start
    # This results in exactly 2 keyframes
    kf_fwd = assemble_keyframes(0.0, 1.5)
    assert kf_fwd.shape == (2, 64)
    assert not torch.allclose(kf_fwd[0], kf_fwd[1], atol=1e-3)

    # 4. Compound motion (angle != 0 and dist != 0): distinct 3 keyframes
    kf_comp = assemble_keyframes(45.0, 1.5)
    assert kf_comp.shape == (3, 64)
    assert not torch.allclose(kf_comp[0], kf_comp[1], atol=1e-3)
    assert not torch.allclose(kf_comp[1], kf_comp[2], atol=1e-3)


def test_cached_latent_dataset_speed_and_buffer_trimming(tmp_path: Path) -> None:
    """Test buffer trimming and physical speed calibration in CachedLatentDataset.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    latent_dim = 16
    # 1. Short video: 100 frames (<= 2 * 72 = 144)
    short_latents = torch.randn(100, latent_dim)
    torch.save(
        {"video_name": "short.mp4", "latents": short_latents},
        tmp_path / "short.pt",
    )

    # 2. Long video: 200 frames (> 144)
    long_latents = torch.randn(200, latent_dim)
    torch.save(
        {"video_name": "long.mp4", "latents": long_latents},
        tmp_path / "long.pt",
    )

    # Instantiate dataset with 3.0m buffer at 2.5 m/s, 60 FPS
    dataset = CachedLatentDataset(
        cache_dir=tmp_path,
        mode="forward",
        video_fps=60.0,
        straight_video_speed_mps=2.5,
        buffer_distance_meters=3.0,
        samples_per_frame=2,
    )

    # step_distance = 2.5 / 60.0 = 0.041666...
    assert pytest.approx(dataset.step_distance_meters, rel=1e-4) == 2.5 / 60.0
    assert dataset.buffer_frames == 72
    assert len(dataset.records) == 2
    assert len(dataset) > 0

    # Verify all pairs come from the long video (index 1) and stay within [72, 128)
    for r_idx, i, j in dataset.pairs:
        rec_name = str(dataset.records[r_idx]["video_name"])
        assert rec_name == "long.mp4"
        assert 72 <= i < 128
        assert 72 < j < 128
        assert j > i
        assert (j - i) <= 72

    # Verify sample outputs have calibrated physical distance
    start_z, target_z, dist = dataset[0]
    assert start_z.shape == (latent_dim,)
    assert target_z.shape == (latent_dim,)
    assert dist.item() > 0.0

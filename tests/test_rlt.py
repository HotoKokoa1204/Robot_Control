"""
Module: test_rlt
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Residual Latent Transformer and latent datasets.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from agilab_lib.datasets.latent_dataset import (
    AngleDataset,
    CachedLatentDataset,
    DummyLatentHorizonDataset,
    DummyLatentPairDataset,
    InMemoryLatentOffsetDataset,
)
from agilab_lib.models.rlt import (
    ConditionedResidualBlock,
    ResidualLatentTransformer,
)


def test_conditioned_residual_block_shape() -> None:
    """Test ConditionedResidualBlock output dimensions with 3D condition vector."""
    block = ConditionedResidualBlock(dim=128, hidden_dim=64)
    x = torch.randn(4, 128)
    cond = torch.randn(4, 3)
    out = block(x, cond)
    assert out.shape == (4, 128)


def test_conditioned_residual_block_identity_zero_init() -> None:
    """Verify exact identity mapping from zero-init ConditionedResidualBlock."""
    block = ConditionedResidualBlock(dim=128, hidden_dim=64, cond_dim=3)
    assert torch.all(block.fc3.weight == 0.0)
    assert torch.all(block.fc3.bias == 0.0)

    # Input tensor with both positive and negative values
    x = torch.randn(4, 128)
    cond = torch.randn(4, 3)
    out = block(x, cond)

    assert torch.equal(out, x)
    assert torch.max(torch.abs(out - x)).item() == 0.0


def test_rlt_zero_motion_exact_identity() -> None:
    """Verify residual transformer yields exact numerical identity at zero motion."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=3,
        block_inner_dim=64,
    )
    z = torch.randn(8, 128)

    # 1. Scalar zero motion
    out_scalar = model(z, angle_deg=0.0, distance_meters=0.0)
    assert torch.equal(out_scalar, z)
    assert torch.max(torch.abs(out_scalar - z)).item() == 0.0

    # 2. Tensor zero motion
    angle_tensor = torch.zeros(8, 1)
    dist_tensor = torch.zeros(8, 1)
    out_tensor = model(z, angle_deg=angle_tensor, distance_meters=dist_tensor)
    assert torch.equal(out_tensor, z)
    assert torch.max(torch.abs(out_tensor - z)).item() == 0.0

    # 3. Direct unit vector sin_cos = [0, 1] with distance = 0
    sc = torch.tensor([[0.0, 1.0]]).expand(8, 2)
    out_sc = model(z, sin_cos=sc, distance_meters=dist_tensor)
    assert torch.equal(out_sc, z)
    assert torch.max(torch.abs(out_sc - z)).item() == 0.0


def test_rlt_negative_coordinate_preservation() -> None:
    """Verify negative latent coordinates are preserved without ReLU truncation."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=3,
        block_inner_dim=64,
    )
    # Strictly negative latent tensor
    z_neg = -torch.abs(torch.randn(4, 128)) - 0.5
    assert torch.all(z_neg < 0.0)

    out = model(z_neg, angle_deg=0.0, distance_meters=0.0)
    assert torch.all(out < 0.0)
    assert torch.equal(out, z_neg)
    assert torch.max(torch.abs(out - z_neg)).item() == 0.0


def test_rlt_direct_stream_architecture_and_reset_parameters() -> None:
    """Verify direct-stream architecture and reset_parameters zeroing contract."""
    latent_dim = 128
    model = ResidualLatentTransformer(
        latent_dim=latent_dim,
        hidden_dim=64,
        num_blocks=3,
        block_inner_dim=64,
    )
    # fc_in and fc_out must be Identity modules
    assert isinstance(model.fc_in, nn.Identity)
    assert isinstance(model.fc_out, nn.Identity)

    # All blocks must have dim == latent_dim
    for blk in model.blocks:
        assert isinstance(blk, ConditionedResidualBlock)
        assert blk.fc1.in_features == latent_dim + 3
        assert blk.fc3.out_features == latent_dim
        assert torch.all(blk.fc3.weight == 0.0)
        assert torch.all(blk.fc3.bias == 0.0)

    # Perturb weights and verify reset_parameters re-zeroes them
    for blk in model.blocks:
        blk.fc3.weight.data.fill_(1.0)
        blk.fc3.bias.data.fill_(1.0)
    model.reset_parameters()
    for blk in model.blocks:
        assert torch.all(blk.fc3.weight == 0.0)
        assert torch.all(blk.fc3.bias == 0.0)


def test_rlt_gradient_flow_non_zero_motion() -> None:
    """Verify gradients propagate cleanly through all blocks under non-zero motion."""
    model = ResidualLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=3,
        block_inner_dim=32,
    )
    # Initialize non-zero weights in blocks to verify end-to-end backprop
    for blk in model.blocks:
        nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
        nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)

    z = torch.randn(2, 64, requires_grad=True)
    angle = torch.tensor([[30.0], [-45.0]])
    distance = torch.tensor([[1.0], [0.5]])

    out = model(z, angle_deg=angle, distance_meters=distance)
    loss = (out**2).sum()
    loss.backward()

    assert z.grad is not None
    assert torch.any(z.grad != 0.0)

    for i, blk in enumerate(model.blocks):
        assert blk.fc1.weight.grad is not None, f"Block {i} fc1 weight grad is None"
        assert torch.any(
            blk.fc1.weight.grad != 0.0
        ), f"Block {i} fc1 weight grad is zero"
        assert blk.fc2.weight.grad is not None, f"Block {i} fc2 weight grad is None"
        assert torch.any(
            blk.fc2.weight.grad != 0.0
        ), f"Block {i} fc2 weight grad is zero"
        assert blk.fc3.weight.grad is not None, f"Block {i} fc3 weight grad is None"
        assert torch.any(
            blk.fc3.weight.grad != 0.0
        ), f"Block {i} fc3 weight grad is zero"


def test_rlt_pure_rotation_shape() -> None:
    """Test ResidualLatentTransformer forward pass with pure rotation motion."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=2,
        block_inner_dim=64,
    )
    z = torch.randn(4, 128)
    angle = torch.tensor([[10.0], [-20.0], [45.0], [0.0]])
    distance = torch.zeros(4, 1)

    out = model(z, angle, distance)
    assert out.shape == (4, 128)


def test_rlt_direct_sin_cos_shape() -> None:
    """Test ResidualLatentTransformer forward pass with direct 2D sin_cos input."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=2,
        block_inner_dim=64,
    )
    z = torch.randn(4, 128)
    sin_cos = torch.randn(4, 2)
    # Test with distance_meters=None
    out = model(z, sin_cos=sin_cos)
    assert out.shape == (4, 128)

    # Test with distance_meters provided
    dist = torch.ones(4, 1)
    out_with_dist = model(z, sin_cos=sin_cos, distance_meters=dist)
    assert out_with_dist.shape == (4, 128)


def test_rlt_pure_forward_shape() -> None:
    """Test ResidualLatentTransformer forward pass with pure linear forward motion."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=2,
        block_inner_dim=64,
    )
    z = torch.randn(4, 128)
    angle = torch.zeros(4, 1)
    distance = torch.tensor([[1.0], [2.5], [0.2], [0.0]])

    out = model(z, angle, distance)
    assert out.shape == (4, 128)


def test_rlt_combined_motion_shape() -> None:
    """Test ResidualLatentTransformer with simultaneous rotation and translation."""
    model = ResidualLatentTransformer(
        latent_dim=128,
        hidden_dim=64,
        num_blocks=2,
        block_inner_dim=64,
    )
    z = torch.randn(4, 128)
    angle = torch.tensor([[15.0], [-15.0], [30.0], [-45.0]])
    distance = torch.tensor([[1.2], [0.8], [2.0], [0.5]])

    out = model(z, angle, distance)
    assert out.shape == (4, 128)


def test_dummy_latent_pair_dataset() -> None:
    """Test DummyLatentPairDataset sample shapes and count."""
    dataset = DummyLatentPairDataset(num_samples=8, latent_dim=128)
    assert len(dataset) == 8
    z_i, z_j, angle = dataset[0]
    assert z_i.shape == (128,)
    assert z_j.shape == (128,)
    assert angle.shape == (1,)


def test_dummy_latent_horizon_dataset() -> None:
    """Test DummyLatentHorizonDataset sample shapes over prediction horizon."""
    dataset = DummyLatentHorizonDataset(num_samples=5, latent_dim=128, horizon=6)
    assert len(dataset) == 5
    z_in, z_future = dataset[0]
    assert z_in.shape == (128,)
    assert z_future.shape == (6, 128)


def test_angle_dataset() -> None:
    """Test AngleDataset pair extraction from precomputed latent array."""
    latents = np.random.randn(20, 128).astype(np.float32)
    dataset = AngleDataset(latent_array=latents, total_frames=20, samples_per_frame=2)
    assert len(dataset) == 40
    z_i, angle, z_j = dataset[0]
    assert z_i.shape == (128,)
    assert z_j.shape == (128,)
    assert angle.shape == (1,)


def test_in_memory_latent_offset_dataset_empty() -> None:
    """Test InMemoryLatentOffsetDataset initialization without video assets."""
    dataset = InMemoryLatentOffsetDataset()
    assert len(dataset) == 0


def test_cached_latent_dataset_rotation(tmp_path: Path) -> None:
    """Test CachedLatentDataset in rotation mode with temporary latent cache.

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    dummy_latents = torch.randn(10, 128)
    dummy_record = {
        "video_path": "dummy_360.mp4",
        "latents": dummy_latents,
        "total_frames": 10,
        "fps": 30.0,
    }
    torch.save(dummy_record, tmp_path / "rotation_sample.pt")

    dataset = CachedLatentDataset(
        cache_dir=tmp_path,
        mode="rotation",
        samples_per_frame=2,
        max_frame_offset=3,
    )
    assert len(dataset) == 10 * 2
    z_i, z_j, angle = dataset[0]
    assert z_i.shape == (128,)
    assert z_j.shape == (128,)
    assert angle.shape == (1,)


def test_cached_latent_dataset_forward(tmp_path: Path) -> None:
    """Test CachedLatentDataset in forward mode with step distance labeling.

    Args:
        tmp_path: Pytest temporary directory path fixture.
    """
    dummy_latents = torch.randn(12, 128)
    dummy_record = {
        "video_path": "dummy_path.mp4",
        "latents": dummy_latents,
        "total_frames": 12,
        "fps": 30.0,
    }
    torch.save(dummy_record, tmp_path / "forward_sample.pt")

    dataset = CachedLatentDataset(
        cache_dir=tmp_path,
        mode="forward",
        samples_per_frame=3,
        max_frame_offset=4,
        step_distance_meters=0.1,
    )
    assert len(dataset) == 12 * 3
    z_i, z_j, dist = dataset[0]
    assert z_i.shape == (128,)
    assert z_j.shape == (128,)
    assert dist.shape == (1,)


def test_cached_latent_dataset_calibrated_wrapping(tmp_path: Path) -> None:
    """Test calibrated circular shortest-path angle wrapping and full-range pairs.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    total_frames = 20
    dummy_latents = torch.randn(total_frames, 128)
    dummy_record = {
        "video_path": "calibrated_360.mp4",
        "latents": dummy_latents,
        "total_frames": total_frames,
        "fps": 60.0,
    }
    torch.save(dummy_record, tmp_path / "calibrated_sample.pt")

    # When max_frame_offset is None, pairs span the full rotation
    dataset = CachedLatentDataset(
        cache_dir=tmp_path,
        mode="rotation",
        max_frame_offset=None,
        samples_per_frame=4,
        return_sin_cos=True,
    )
    assert len(dataset) == total_frames * 4

    tensor_ds = dataset.to_tensor_dataset()
    assert len(tensor_ds) == len(dataset)

    # Check all samples
    for i in range(len(dataset)):
        _, _, sc = dataset[i]
        assert sc.shape == (2,)
        norm = torch.norm(sc, p=2).item()
        assert abs(norm - 1.0) < 1e-4

        # Verify angle in [-180, 180]
        angle_deg = math.atan2(sc[0].item(), sc[1].item()) * 180.0 / math.pi
        assert -180.0 <= angle_deg <= 180.0

    # Verify tensor dataset angles match
    _, _, all_sc = tensor_ds[:]
    assert all_sc.shape == (len(dataset), 2)
    norms = torch.norm(all_sc, p=2, dim=-1)
    assert torch.all(torch.abs(norms - 1.0) < 1e-4)

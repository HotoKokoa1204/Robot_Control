"""
Module: test_rlt
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Residual Latent Transformer and latent datasets.
"""

from pathlib import Path

import numpy as np
import torch
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

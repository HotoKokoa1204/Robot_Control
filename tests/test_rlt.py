"""
Module: test_rlt
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Residual Latent Transformer and latent datasets.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from agilab_lib.models.rlt import (
    ConditionedResidualBlock,
    ResidualLatentTransformer,
)


class DummyLatentPairDataset:
    """Mock latent pair dataset for unit tests."""

    def __init__(
        self,
        num_samples: int = 32,
        latent_dim: int = 128,
        return_sin_cos: bool = False,
    ) -> None:
        """Initialize mock dataset.

        Args:
            num_samples: Number of sample pairs.
            latent_dim: Dimension of latent vectors.
            return_sin_cos: Whether to return 2D sin/cos vector.
        """
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.return_sin_cos = return_sin_cos

    def __len__(self) -> int:
        """Return sample count."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random pair and target."""
        z_i = torch.randn(self.latent_dim, dtype=torch.float32)
        z_j = torch.randn(self.latent_dim, dtype=torch.float32)
        angle_deg = random.uniform(-180.0, 180.0)
        if self.return_sin_cos:
            rad = angle_deg * math.pi / 180.0
            target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor([angle_deg], dtype=torch.float32)
        return z_i, z_j, target


class DummyLatentHorizonDataset:
    """Mock latent horizon dataset for unit tests."""

    def __init__(
        self,
        num_samples: int = 20,
        latent_dim: int = 128,
        horizon: int = 10,
    ) -> None:
        """Initialize mock dataset.

        Args:
            num_samples: Number of samples.
            latent_dim: Dimension of latent vectors.
            horizon: Prediction horizon steps.
        """
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.horizon = horizon

    def __len__(self) -> int:
        """Return sample count."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random current and future latents."""
        z_in = torch.randn(self.latent_dim, dtype=torch.float32)
        z_future = torch.randn(self.horizon, self.latent_dim, dtype=torch.float32)
        return z_in, z_future


class AngleDataset:
    """Mock angle dataset for unit tests."""

    def __init__(
        self,
        latent_array: Union[np.ndarray, torch.Tensor],
        total_frames: Optional[int] = None,
        samples_per_frame: int = 10,
        max_angle: float = 15.0,
        return_sin_cos: bool = False,
    ) -> None:
        """Initialize mock angle dataset.

        Args:
            latent_array: Latent vector array.
            total_frames: Total frame count.
            samples_per_frame: Samples per base frame.
            max_angle: Max sampling angle.
            return_sin_cos: Whether to return sin/cos representation.
        """
        if isinstance(latent_array, np.ndarray):
            self.latent_array: torch.Tensor = torch.from_numpy(latent_array).float()
        else:
            self.latent_array = latent_array.float()

        if total_frames is None:
            total_frames = self.latent_array.shape[0]

        self.total_frames: int = total_frames
        self.ratio: float = float(self.total_frames) / 360.0
        self.return_sin_cos: bool = return_sin_cos
        self.data_pairs: List[Tuple[int, float]] = []

        for i in range(self.total_frames):
            angles = np.random.uniform(
                low=-max_angle, high=max_angle, size=samples_per_frame
            )
            for angle in angles:
                self.data_pairs.append((i, float(angle)))

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch sampled pair and angle target."""
        i, angle = self.data_pairs[idx]
        latent_i = self.latent_array[i]
        frame_offset = int(round(angle * self.ratio))
        j = max(0, min(self.total_frames - 1, i + frame_offset))
        latent_j = self.latent_array[j]
        if self.return_sin_cos:
            rad = angle * math.pi / 180.0
            target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor([angle], dtype=torch.float32)
        return latent_i, target, latent_j


class InMemoryLatentOffsetDataset:
    """Mock in-memory offset dataset for unit tests."""

    def __init__(self) -> None:
        """Initialize empty mock dataset."""
        self.data_pairs_indices: List[Tuple[int, int, int]] = []

    def __len__(self) -> int:
        """Return sample count."""
        return len(self.data_pairs_indices)


class CachedLatentDataset:
    """Mock cached latent dataset for unit tests."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        mode: str = "rotation",
        max_frame_offset: Optional[int] = None,
        samples_per_frame: int = 5,
        step_distance_meters: Optional[float] = None,
        return_sin_cos: bool = False,
        video_fps: float = 60.0,
        straight_video_speed_mps: float = 2.5,
        buffer_distance_meters: Optional[float] = None,
        buffer_frames: Optional[int] = None,
    ) -> None:
        """Initialize mock cached latent dataset.

        Args:
            cache_dir: Directory containing .pt files.
            mode: Motion mode.
            max_frame_offset: Max frame offset.
            samples_per_frame: Samples per frame.
            step_distance_meters: Physical step distance.
            return_sin_cos: Whether to return sin/cos representation.
            video_fps: Video FPS.
            straight_video_speed_mps: Forward speed.
            buffer_distance_meters: Buffer distance in meters.
            buffer_frames: Buffer frames.
        """
        self.mode: str = mode.lower()
        self.max_frame_offset: Optional[int] = max_frame_offset
        self.return_sin_cos: bool = return_sin_cos
        self.video_fps: float = float(video_fps)
        self.straight_video_speed_mps: float = float(straight_video_speed_mps)

        if step_distance_meters is not None:
            self.step_distance_meters: float = float(step_distance_meters)
        else:
            self.step_distance_meters = self.straight_video_speed_mps / max(
                self.video_fps, 1.0
            )

        if self.mode == "forward":
            if buffer_frames is not None:
                self.buffer_frames: int = max(0, int(buffer_frames))
            elif buffer_distance_meters is not None and buffer_distance_meters > 0:
                self.buffer_frames = int(
                    buffer_distance_meters / self.step_distance_meters
                )
            else:
                self.buffer_frames = 0
        else:
            self.buffer_frames = 0

        self.records: List[Dict[str, Any]] = []
        self.pairs: List[Tuple[int, int, int]] = []

        dir_path = Path(cache_dir)
        if dir_path.exists():
            pt_files = sorted(dir_path.rglob("*.pt"))
            for r_idx, pf in enumerate(pt_files):
                rec = torch.load(pf, weights_only=False)
                if isinstance(rec, dict) and "latents" in rec:
                    self.records.append(rec)
                    latents = rec["latents"]
                    num_frames = len(latents)

                    if self.mode == "forward" and self.buffer_frames > 0:
                        buf = self.buffer_frames
                        if num_frames <= 2 * buf:
                            continue
                        forward_limit = (
                            max_frame_offset if max_frame_offset is not None else buf
                        )
                        start_idx = buf
                        end_idx = num_frames - buf
                        for i in range(start_idx, end_idx):
                            for _ in range(samples_per_frame):
                                max_diff = min(end_idx - 1 - i, forward_limit)
                                if max_diff <= 0:
                                    continue
                                offset = random.randint(1, max_diff)
                                j = i + offset
                                self.pairs.append((r_idx, i, j))
                    else:
                        half_turn = num_frames // 2
                        effective_offset = (
                            half_turn
                            if max_frame_offset is None and self.mode == "rotation"
                            else (
                                max_frame_offset if max_frame_offset is not None else 15
                            )
                        )
                        for i in range(num_frames):
                            for _ in range(samples_per_frame):
                                offset = random.randint(
                                    -effective_offset, effective_offset
                                )
                                if self.mode == "rotation" and max_frame_offset is None:
                                    j = (i + offset) % num_frames
                                else:
                                    j = max(0, min(num_frames - 1, i + offset))
                                self.pairs.append((r_idx, i, j))

    def __len__(self) -> int:
        """Return pair count."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch start latent, target latent, and motion tensor."""
        r_idx, i, j = self.pairs[idx]
        rec = self.records[r_idx]
        latents = rec["latents"]
        z_i = latents[i]
        z_j = latents[j]
        frame_diff = j - i

        if self.mode == "rotation":
            total_frames = float(rec.get("total_frames", len(latents)))
            angle_raw = float(frame_diff) * (360.0 / max(total_frames, 1.0))
            angle = ((angle_raw + 180.0) % 360.0) - 180.0
            if self.return_sin_cos:
                rad = angle * math.pi / 180.0
                sc = [math.sin(rad), math.cos(rad)]
                return z_i, z_j, torch.tensor(sc, dtype=torch.float32)
            return z_i, z_j, torch.tensor([angle], dtype=torch.float32)
        else:
            dist = float(frame_diff) * self.step_distance_meters
            return z_i, z_j, torch.tensor([dist], dtype=torch.float32)

    def to_tensor_dataset(self) -> TensorDataset:
        """Convert pairs to TensorDataset."""
        if not self.pairs:
            dim = (
                int(self.records[0]["latents"].shape[-1])
                if self.records and "latents" in self.records[0]
                else 128
            )
            return TensorDataset(
                torch.empty((0, dim)),
                torch.empty((0, dim)),
                torch.empty((0, 2 if self.return_sin_cos else 1)),
            )
        z_i_list = []
        z_j_list = []
        motion_list = []
        for idx in range(len(self.pairs)):
            zi, zj, mot = self[idx]
            z_i_list.append(zi)
            z_j_list.append(zj)
            motion_list.append(mot)
        return TensorDataset(
            torch.stack(z_i_list),
            torch.stack(z_j_list),
            torch.stack(motion_list),
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
        assert torch.any(blk.fc1.weight.grad != 0.0), (
            f"Block {i} fc1 weight grad is zero"
        )
        assert blk.fc2.weight.grad is not None, f"Block {i} fc2 weight grad is None"
        assert torch.any(blk.fc2.weight.grad != 0.0), (
            f"Block {i} fc2 weight grad is zero"
        )
        assert blk.fc3.weight.grad is not None, f"Block {i} fc3 weight grad is None"
        assert torch.any(blk.fc3.weight.grad != 0.0), (
            f"Block {i} fc3 weight grad is zero"
        )


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

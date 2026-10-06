"""Module: test_dual_source_dataset
Stage: Test
Author: KafuuChino
Date: 2026-09-29
Description: Unit tests for DualSourceVideoDataset and DummyDualSourceVideoDataset.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from agilab_lib.datasets.dual_source_dataset import (
    DualSourceBatch,
    DualSourceVideoDataset,
    DummyDualSourceVideoDataset,
    dual_source_collate_fn,
)
from torch.utils.data import DataLoader


def _create_synthetic_video(
    video_path: Path,
    num_frames: int = 30,
    width: int = 192,
    height: int = 108,
    fps: float = 30.0,
) -> None:
    """Create a synthetic MP4 video file with color-graded frames.

    Args:
        video_path: Target path to save the generated video file.
        num_frames: Number of synthetic frames to write.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Video playback framerate.
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        color_val = int((i / max(num_frames - 1, 1)) * 255)
        frame = np.full((height, width, 3), color_val, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_dummy_dual_source_dataset_shapes_and_types() -> None:
    """Test DummyDualSourceVideoDataset item shapes, types, and ranges."""
    num_samples = 10
    h, w = 108, 192
    min_dist, max_dist = 0.1, 2.5
    dataset = DummyDualSourceVideoDataset(
        num_samples=num_samples,
        img_height=h,
        img_width=w,
        min_distance=min_dist,
        max_distance=max_dist,
        recon_source="forward",
    )

    assert len(dataset) == num_samples

    item = dataset[0]
    assert isinstance(item, DualSourceBatch)
    assert isinstance(item, dict)

    # Required keys verification
    required_keys = {
        "recon_frame",
        "fwd_current",
        "fwd_target",
        "fwd_distance",
        "rot_current",
        "rot_target",
        "rot_sin_cos",
    }
    assert required_keys.issubset(item.keys())

    # Tensor shape contracts
    assert item.recon_frame.shape == (3, h, w)
    assert item.fwd_current.shape == (3, h, w)
    assert item.fwd_target.shape == (3, h, w)
    assert item.fwd_distance.shape == (1,)
    assert item.rot_current.shape == (3, h, w)
    assert item.rot_target.shape == (3, h, w)
    assert item.rot_sin_cos.shape == (2,)

    # Dtype and value ranges
    assert item.recon_frame.dtype == torch.float32
    assert item.fwd_current.dtype == torch.float32
    assert item.fwd_target.dtype == torch.float32
    assert item.fwd_distance.dtype == torch.float32
    assert item.rot_current.dtype == torch.float32
    assert item.rot_target.dtype == torch.float32
    assert item.rot_sin_cos.dtype == torch.float32

    assert 0.0 <= item.recon_frame.min().item() <= 1.0
    assert 0.0 <= item.recon_frame.max().item() <= 1.0

    # Distance within configured bounds
    assert min_dist <= item.fwd_distance.item() <= max_dist

    # Unit vector magnitude check for sin_cos
    norm = torch.linalg.norm(item.rot_sin_cos).item()
    assert math.isclose(norm, 1.0, rel_tol=1e-4)


def test_dummy_dual_source_dataset_recon_sources() -> None:
    """Test DummyDualSourceVideoDataset reconstruction source routing."""
    # Forward source routing
    ds_fwd = DummyDualSourceVideoDataset(num_samples=5, recon_source="forward")
    item_fwd = ds_fwd[0]
    assert torch.allclose(item_fwd.recon_frame, item_fwd.fwd_current)

    # Rotation source routing
    ds_rot = DummyDualSourceVideoDataset(num_samples=5, recon_source="rotation")
    item_rot = ds_rot[0]
    assert torch.allclose(item_rot.recon_frame, item_rot.rot_current)

    # Random source routing
    ds_rand = DummyDualSourceVideoDataset(
        num_samples=10, recon_source="random", seed=123
    )
    for i in range(len(ds_rand)):
        item = ds_rand[i]
        is_fwd = torch.allclose(item.recon_frame, item.fwd_current)
        is_rot = torch.allclose(item.recon_frame, item.rot_current)
        assert is_fwd or is_rot


def test_dummy_dual_source_dataset_seed_reproducibility() -> None:
    """Test that setting a seed provides deterministic reproducible outputs."""
    ds1 = DummyDualSourceVideoDataset(num_samples=5, seed=42)
    ds2 = DummyDualSourceVideoDataset(num_samples=5, seed=42)

    for i in range(5):
        assert torch.allclose(ds1[i].recon_frame, ds2[i].recon_frame)
        assert torch.allclose(ds1[i].fwd_distance, ds2[i].fwd_distance)
        assert torch.allclose(ds1[i].rot_sin_cos, ds2[i].rot_sin_cos)


def test_dual_source_batch_access_and_device_transfer() -> None:
    """Test DualSourceBatch key/property access and .to(device) transfer."""
    batch = DualSourceBatch(
        recon_frame=torch.rand(2, 3, 108, 192),
        fwd_current=torch.rand(2, 3, 108, 192),
        fwd_target=torch.rand(2, 3, 108, 192),
        fwd_distance=torch.tensor([[1.0], [2.0]]),
        rot_current=torch.rand(2, 3, 108, 192),
        rot_target=torch.rand(2, 3, 108, 192),
        rot_sin_cos=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )

    # Property accessors
    assert batch.recon_frame.shape == (2, 3, 108, 192)
    assert batch.fwd_current.shape == (2, 3, 108, 192)
    assert batch.fwd_target.shape == (2, 3, 108, 192)
    assert batch.fwd_distance.shape == (2, 1)
    assert batch.rot_current.shape == (2, 3, 108, 192)
    assert batch.rot_target.shape == (2, 3, 108, 192)
    assert batch.rot_sin_cos.shape == (2, 2)

    # Dictionary key access
    assert batch["recon_frame"] is batch.recon_frame
    assert batch["fwd_distance"] is batch.fwd_distance

    # Device transfer
    cpu_batch = batch.to("cpu")
    assert isinstance(cpu_batch, DualSourceBatch)
    assert cpu_batch.recon_frame.device.type == "cpu"


def test_dataloader_batch_generation() -> None:
    """Test PyTorch DataLoader collation and batch tensor shape contracts."""
    dataset = DummyDualSourceVideoDataset(num_samples=16, img_height=108, img_width=192)
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert isinstance(batch, DualSourceBatch)
        assert batch.recon_frame.shape == (batch_size, 3, 108, 192)
        assert batch.fwd_current.shape == (batch_size, 3, 108, 192)
        assert batch.fwd_target.shape == (batch_size, 3, 108, 192)
        assert batch.fwd_distance.shape == (batch_size, 1)
        assert batch.rot_current.shape == (batch_size, 3, 108, 192)
        assert batch.rot_target.shape == (batch_size, 3, 108, 192)
        assert batch.rot_sin_cos.shape == (batch_size, 2)

    assert batch_count == 4

    # Test sample_batch direct method
    sampled = dataset.sample_batch(batch_size=6)
    assert isinstance(sampled, DualSourceBatch)
    assert sampled.recon_frame.shape == (6, 3, 108, 192)
    assert sampled.fwd_distance.shape == (6, 1)
    assert sampled.rot_sin_cos.shape == (6, 2)


def test_dual_source_collate_fn() -> None:
    """Test dual_source_collate_fn with custom dictionary samples."""
    samples = [
        {
            "recon_frame": torch.rand(3, 10, 10),
            "fwd_current": torch.rand(3, 10, 10),
            "fwd_target": torch.rand(3, 10, 10),
            "fwd_distance": torch.tensor([1.2]),
            "rot_current": torch.rand(3, 10, 10),
            "rot_target": torch.rand(3, 10, 10),
            "rot_sin_cos": torch.tensor([0.6, 0.8]),
            "custom_scalar": torch.tensor([42.0]),
        }
        for _ in range(3)
    ]
    collated = dual_source_collate_fn(samples)
    assert isinstance(collated, DualSourceBatch)
    assert collated.recon_frame.shape == (3, 3, 10, 10)
    assert collated.fwd_distance.shape == (3, 1)
    assert collated.rot_sin_cos.shape == (3, 2)
    assert collated["custom_scalar"].shape == (3, 1)


def test_dual_source_video_dataset_with_synthetic_videos(tmp_path: Path) -> None:
    """Test DualSourceVideoDataset using synthetic video files on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    fwd_dir = tmp_path / "one_path"
    rot_dir = tmp_path / "360"

    # Create synthetic video files
    _create_synthetic_video(fwd_dir / "vid1.mp4", num_frames=120, fps=30.0)
    _create_synthetic_video(rot_dir / "vid2.mp4", num_frames=120, fps=30.0)

    # Test on-demand reading (preload_frames=False)
    dataset_ondemand = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        img_height=54,
        img_width=96,
        video_fps=30.0,
        straight_video_speed_mps=1.5,
        samples_per_video=10,
        preload_frames=False,
        seed=101,
    )

    assert len(dataset_ondemand) == 10
    item = dataset_ondemand[0]
    assert item.recon_frame.shape == (3, 54, 96)
    assert item.fwd_current.shape == (3, 54, 96)
    assert item.fwd_target.shape == (3, 54, 96)
    assert item.fwd_distance.shape == (1,)
    assert item.rot_current.shape == (3, 54, 96)
    assert item.rot_target.shape == (3, 54, 96)
    assert item.rot_sin_cos.shape == (2,)

    # Verify rotation unit vector
    sc_norm = torch.linalg.norm(item.rot_sin_cos).item()
    assert math.isclose(sc_norm, 1.0, rel_tol=1e-4)

    # Verify straight-line buffer pruning: frames must be strictly in [40, 80)
    assert len(dataset_ondemand.fwd_pairs) == 10
    for _, t, j, dist in dataset_ondemand.fwd_pairs:
        assert 40 <= t < 80
        assert 40 < j < 80
        assert t < j
        assert math.isclose(dist, (j - t) * dataset_ondemand.step_distance_meters)

    # Test preloading (preload_frames=True)
    dataset_preloaded = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        img_height=54,
        img_width=96,
        samples_per_video=10,
        preload_frames=True,
        seed=101,
    )
    assert len(dataset_preloaded) == 10
    item_pre = dataset_preloaded[0]
    assert item_pre.recon_frame.shape == (3, 54, 96)

    # Test batch sampling from dataset
    sampled_batch = dataset_preloaded.sample_batch(batch_size=4)
    assert sampled_batch.recon_frame.shape == (4, 3, 54, 96)
    assert sampled_batch.fwd_distance.shape == (4, 1)
    assert sampled_batch.rot_sin_cos.shape == (4, 2)


def test_dual_source_video_dataset_missing_and_empty_dirs(tmp_path: Path) -> None:
    """Test error handling for non-existent and empty directories.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    empty_fwd = tmp_path / "empty_fwd"
    empty_rot = tmp_path / "empty_rot"
    empty_fwd.mkdir()
    empty_rot.mkdir()

    # Non-existent directories raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DualSourceVideoDataset(
            one_path_dir=tmp_path / "missing1",
            rotation_dir=tmp_path / "missing2",
        )

    # Empty directories with no videos raise ValueError
    with pytest.raises(ValueError):
        DualSourceVideoDataset(
            one_path_dir=empty_fwd,
            rotation_dir=empty_rot,
        )


def test_dual_source_video_dataset_with_repo_data() -> None:
    """Test DualSourceVideoDataset with real repository data if present."""
    fwd_dir = Path("data/one_path")
    rot_dir = Path("data/360")

    resolved_fwd = DualSourceVideoDataset._resolve_dir(fwd_dir)
    resolved_rot = DualSourceVideoDataset._resolve_dir(rot_dir)

    if not resolved_fwd.exists() or not resolved_rot.exists():
        pytest.skip("Repo video data not present in execution environment.")

    dataset = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        max_videos_per_source=1,
        samples_per_video=4,
        preload_frames=False,
    )

    assert len(dataset) >= 1
    batch = dataset.sample_batch(batch_size=2)
    assert batch.recon_frame.shape == (2, 3, 108, 192)
    assert batch.fwd_distance.shape == (2, 1)
    assert batch.rot_sin_cos.shape == (2, 2)


def test_dual_source_video_dataset_one_per_subfolder(tmp_path: Path) -> None:
    """Verify rotation_one_per_subfolder picks at most one video per subfolder."""
    fwd_dir = tmp_path / "fwd"
    fwd_dir.mkdir()
    rot_dir = tmp_path / "rot"
    rot_dir.mkdir()

    (fwd_dir / "vid1.mp4").touch()

    sub1 = rot_dir / "sub1"
    sub1.mkdir()
    (sub1 / "take1.mp4").touch()
    (sub1 / "take2.mp4").touch()

    sub2 = rot_dir / "sub2"
    sub2.mkdir()
    (sub2 / "take1.mp4").touch()
    (sub2 / "take2.mp4").touch()

    found_one = DualSourceVideoDataset._find_video_files(
        rot_dir, one_per_subfolder=True
    )
    assert len(found_one) == 2

    found_all = DualSourceVideoDataset._find_video_files(
        rot_dir, one_per_subfolder=False
    )
    assert len(found_all) == 4


def test_buffer_frames_calculation_and_defaults() -> None:
    """Verify buffer_distance_meters default, calculation, and overriding.

    Ensures that default 2.0 meters at 60 FPS / 2.5 m/s cruising speed yields
    exactly 48 frames, and explicit parameters correctly override.
    """
    # 1. DummyDualSourceVideoDataset defaults
    dummy_default = DummyDualSourceVideoDataset()
    assert dummy_default.buffer_distance_meters == 2.0
    assert dummy_default.step_distance_meters == 2.5 / 60.0
    assert dummy_default.buffer_frames == 48

    # 2. Custom buffer distance in meters
    dummy_1m = DummyDualSourceVideoDataset(buffer_distance_meters=1.0)
    assert dummy_1m.buffer_distance_meters == 1.0
    assert dummy_1m.buffer_frames == 24

    # 3. Explicit buffer_frames overriding buffer_distance_meters
    dummy_override = DummyDualSourceVideoDataset(
        buffer_distance_meters=2.0, buffer_frames=15
    )
    assert dummy_override.buffer_frames == 15

    # 4. Zero buffer disables margin trimming
    dummy_zero = DummyDualSourceVideoDataset(buffer_distance_meters=0.0)
    assert dummy_zero.buffer_frames == 0


def test_straight_path_buffer_pruning_bounds(tmp_path: Path) -> None:
    """Verify straight-path indexing excludes first and last buffer_frames.

    Checks:
    1. Default buffer_distance_meters=2.0 calculates buffer_frames=48 for
       60 FPS / 2.5 m/s.
    2. Base and target frame indices are strictly inside
       [buffer_frames, total_frames - buffer_frames).
    3. Neither base nor target frames ever fall in [0, buffer_frames)
       or [total_frames - buffer_frames, total_frames).
    4. Short videos with frames <= 2 * buffer_frames are safely skipped.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    fwd_dir = tmp_path / "one_path"
    rot_dir = tmp_path / "360"
    fwd_dir.mkdir(parents=True)
    rot_dir.mkdir(parents=True)

    total_frames = 150
    fps = 60.0
    speed = 2.5
    _create_synthetic_video(fwd_dir / "fwd.mp4", num_frames=total_frames, fps=fps)
    _create_synthetic_video(rot_dir / "rot.mp4", num_frames=total_frames, fps=fps)

    # Instantiate dataset with defaults (buffer_distance_meters=2.0)
    dataset = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        video_fps=fps,
        straight_video_speed_mps=speed,
        samples_per_video=200,
        min_forward_offset=1,
        max_forward_offset=30,
        seed=42,
    )

    expected_buffer = 48
    expected_start = expected_buffer
    expected_end = total_frames - expected_buffer  # 150 - 48 = 102

    assert dataset.buffer_distance_meters == 2.0
    assert dataset.buffer_frames == expected_buffer
    assert len(dataset.fwd_pairs) > 0

    lower_buffer_frames = set(range(0, expected_start))
    upper_buffer_frames = set(range(expected_end, total_frames))

    for _, t, j, dist in dataset.fwd_pairs:
        # Strict boundary assertion: base and target in [expected_start, expected_end)
        assert (
            expected_start <= t < expected_end
        ), f"Base frame {t} outside [{expected_start}, {expected_end})"
        assert (
            expected_start < j < expected_end
        ), f"Target frame {j} outside [{expected_start}, {expected_end})"
        assert t < j

        # Verify neither frame is in lower buffer [0, buffer_frames)
        assert t not in lower_buffer_frames, f"Base frame {t} in lower buffer!"
        assert j not in lower_buffer_frames, f"Target frame {j} in lower buffer!"

        # Verify neither frame is in upper buffer [end, total_frames)
        assert t not in upper_buffer_frames, f"Base frame {t} in upper buffer!"
        assert j not in upper_buffer_frames, f"Target frame {j} in upper buffer!"

        # Verify distance calculation
        expected_dist = (j - t) * dataset.step_distance_meters
        assert math.isclose(dist, expected_dist, rel_tol=1e-5)

    # Exhaustive sweep test (samples_per_video=None)
    dataset_exhaustive = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        video_fps=fps,
        straight_video_speed_mps=speed,
        samples_per_video=None,
        samples_per_frame=1,
        min_forward_offset=1,
        max_forward_offset=30,
        seed=42,
    )
    assert len(dataset_exhaustive.fwd_pairs) > 0
    for _, t, j, _ in dataset_exhaustive.fwd_pairs:
        assert expected_start <= t < expected_end
        assert expected_start < j < expected_end
        assert t not in lower_buffer_frames
        assert j not in lower_buffer_frames
        assert t not in upper_buffer_frames
        assert j not in upper_buffer_frames


def test_straight_path_short_video_without_valid_frames_skipped(
    tmp_path: Path,
) -> None:
    """Verify that a video too short to afford both buffer margins is skipped.

    If a video has num_frames <= 2 * buffer_frames, no valid pair can be formed
    outside the margins. The loader must not fall back to unpruned sampling.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    fwd_dir = tmp_path / "short_fwd"
    rot_dir = tmp_path / "short_rot"
    fwd_dir.mkdir(parents=True)
    rot_dir.mkdir(parents=True)

    # 80 frames is less than 2 * 48 = 96 frames needed for buffer margins
    _create_synthetic_video(fwd_dir / "short.mp4", num_frames=80, fps=60.0)
    _create_synthetic_video(rot_dir / "rot.mp4", num_frames=120, fps=60.0)

    with pytest.raises(ValueError, match="No valid forward frame pairs"):
        DualSourceVideoDataset(
            one_path_dir=fwd_dir,
            rotation_dir=rot_dir,
            video_fps=60.0,
            straight_video_speed_mps=2.5,
            buffer_distance_meters=2.0,
        )


def test_dummy_dual_source_video_dataset_max_rotation_deg() -> None:
    """Verify dummy dataset restricts angles when max_rotation_deg is set."""
    # Test restricted to +-30 degrees
    ds_30 = DummyDualSourceVideoDataset(
        num_samples=100,
        max_rotation_deg=30.0,
        seed=42,
    )
    for i in range(len(ds_30)):
        item = ds_30[i]
        s = item.rot_sin_cos[0].item()
        c = item.rot_sin_cos[1].item()
        deg = abs(math.degrees(math.atan2(s, c)))
        assert deg <= 30.0 + 1e-4, f"Sample {i} angle {deg:.2f} exceeds 30 deg!"

    # Test unrestricted (max_rotation_deg=None)
    ds_full = DummyDualSourceVideoDataset(
        num_samples=100,
        max_rotation_deg=None,
        seed=42,
    )
    large_angles = [
        abs(
            math.degrees(
                math.atan2(
                    ds_full[i].rot_sin_cos[0].item(), ds_full[i].rot_sin_cos[1].item()
                )
            )
        )
        for i in range(len(ds_full))
    ]
    assert any(
        a > 45.0 for a in large_angles
    ), "Full dataset should contain angles > 45 deg."


def test_dual_source_video_dataset_max_rotation_deg(tmp_path: Path) -> None:
    """Verify DualSourceVideoDataset limits rotation offsets to max_rotation_deg."""
    fwd_dir = tmp_path / "fwd"
    rot_dir = tmp_path / "rot"
    _create_synthetic_video(fwd_dir / "fwd.mp4", num_frames=150, fps=30.0)
    _create_synthetic_video(rot_dir / "rot.mp4", num_frames=150, fps=30.0)

    # Test with max_rotation_deg=30.0
    ds = DualSourceVideoDataset(
        one_path_dir=fwd_dir,
        rotation_dir=rot_dir,
        video_fps=30.0,
        straight_video_speed_mps=1.0,
        buffer_distance_meters=1.0,
        samples_per_video=50,
        max_rotation_deg=30.0,
        seed=123,
    )
    assert len(ds.rot_pairs) > 0
    for vp, t, j, (s, c) in ds.rot_pairs:
        deg = abs(math.degrees(math.atan2(s, c)))
        assert (
            deg <= 30.0 + 1e-4
        ), f"Rotation pair {t}->{j} angle {deg:.2f} exceeds 30 deg!"

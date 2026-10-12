"""Module: test_motion_indexing
Stage: Test
Author: KafuuChino
Date: 2026-10-12
Description: Unit tests for pure geometric motion pairing indexers.
"""

import math
import random
from typing import Set

from agilab_lib.datasets.motion_indexing import (
    MotionPairSampler,
    RotationPairIndexer,
    StraightPairIndexer,
)


def test_straight_indexer_buffer_and_step_distance() -> None:
    """Verify StraightPairIndexer buffer margins and step distance calculations."""
    # 1. Default settings (60 fps, 2.5 m/s, 2.0 m buffer)
    indexer = StraightPairIndexer(
        video_fps=60.0,
        straight_video_speed_mps=2.5,
        buffer_distance_meters=2.0,
    )
    assert math.isclose(indexer.step_distance_meters, 2.5 / 60.0, rel_tol=1e-6)
    assert indexer.buffer_frames == 48
    assert math.isclose(indexer.buffer_distance_meters, 2.0, rel_tol=1e-6)

    # 2. Explicit buffer_frames overriding buffer_distance_meters
    indexer_override = StraightPairIndexer(
        video_fps=30.0,
        straight_video_speed_mps=1.5,
        buffer_distance_meters=5.0,
        buffer_frames=12,
    )
    assert indexer_override.buffer_frames == 12
    expected_dist = 12 * (1.5 / 30.0)
    assert math.isclose(indexer_override.buffer_distance_meters, expected_dist)

    # 3. Explicit step_distance_meters overriding speed / fps
    indexer_explicit_step = StraightPairIndexer(
        step_distance_meters=0.05,
        buffer_distance_meters=1.0,
    )
    assert indexer_explicit_step.step_distance_meters == 0.05
    assert indexer_explicit_step.buffer_frames == 20

    # 4. Zero buffer disables margin trimming
    indexer_zero = StraightPairIndexer(buffer_distance_meters=0.0)
    assert indexer_zero.buffer_frames == 0
    assert indexer_zero.buffer_distance_meters == 0.0


def test_straight_indexer_valid_frame_range() -> None:
    """Verify valid search interval computation and sequence length checks."""
    indexer = StraightPairIndexer(
        buffer_frames=20,
        min_forward_offset=5,
    )

    # Long sequence: [20, 100 - 20) = [20, 80)
    start, end = indexer.compute_valid_frame_range(100)
    assert start == 20
    assert end == 80
    assert end - start > indexer.min_forward_offset

    # Sequence too short to afford margins
    start_short, end_short = indexer.compute_valid_frame_range(44)
    assert start_short == 20
    assert end_short == 24
    assert end_short - start_short <= indexer.min_forward_offset

    # Sequence with frames <= 2 * buffer_frames
    start_tiny, end_tiny = indexer.compute_valid_frame_range(30)
    assert end_tiny - start_tiny < 0


def test_straight_indexer_sample_pairs_bounds() -> None:
    """Verify sampled straight pairs respect buffer margins and offset limits."""
    n_frames = 200
    indexer = StraightPairIndexer(
        video_fps=60.0,
        straight_video_speed_mps=2.5,
        buffer_frames=30,
        min_forward_offset=2,
        max_forward_offset=40,
    )

    rng = random.Random(42)
    pairs = indexer.sample_pairs(
        n_frames=n_frames,
        samples_per_video=100,
        rng=rng,
    )

    assert len(pairs) == 100

    lower_buffer: Set[int] = set(range(0, 30))
    upper_buffer: Set[int] = set(range(170, 200))

    for t, j, dist in pairs:
        # Base and target strictly inside [30, 170)
        assert 30 <= t < 170
        assert 30 <= j < 170
        assert t < j

        # Never in lower or upper buffer zones
        assert t not in lower_buffer
        assert j not in lower_buffer
        assert t not in upper_buffer
        assert j not in upper_buffer

        # Offset within constraints
        offset = j - t
        assert 2 <= offset <= 40

        # Physical distance calculation matches
        expected_dist = offset * indexer.step_distance_meters
        assert math.isclose(dist, expected_dist, rel_tol=1e-5)


def test_straight_indexer_sweep_mode() -> None:
    """Verify exhaustive sweep mode (samples_per_video=None)."""
    n_frames = 60
    indexer = StraightPairIndexer(
        buffer_frames=10,
        min_forward_offset=2,
        max_forward_offset=5,
    )

    pairs = indexer.sample_pairs(
        n_frames=n_frames,
        samples_per_video=None,
        samples_per_frame=2,
        rng=random.Random(123),
    )

    # Valid base range: [10, 60 - 10 - 2) = [10, 48) -> 38 base frames
    # 2 samples per frame -> 76 total pairs
    expected_bases = set(range(10, 48))
    sampled_bases = {t for t, _, _ in pairs}
    assert sampled_bases == expected_bases
    assert len(pairs) == 38 * 2


def test_straight_indexer_short_or_empty_sequences() -> None:
    """Verify empty pair lists returned for invalid or short sequences."""
    indexer = StraightPairIndexer(buffer_frames=15, min_forward_offset=5)

    assert indexer.sample_pairs(0) == []
    assert indexer.sample_pairs(1) == []
    assert indexer.sample_pairs(30) == []  # 30 frames <= 2 * 15
    assert indexer.sample_pairs(34) == []  # 34 - 30 = 4 <= min_offset 5


def test_straight_indexer_synthetic_distance() -> None:
    """Verify synthetic uniform distance generation within bounds."""
    rng = random.Random(999)
    for _ in range(50):
        dist = StraightPairIndexer.sample_synthetic_distance(0.2, 3.5, rng)
        assert 0.2 <= dist <= 3.5


def test_rotation_indexer_effective_max_offset() -> None:
    """Verify calculation of effective max offset under degree and offset bounds."""
    # 1. Unbounded: half turn of sequence length
    indexer_unbounded = RotationPairIndexer(
        max_rotation_offset=None,
        max_rotation_deg=None,
    )
    assert indexer_unbounded.compute_effective_max_offset(100) == 50

    # 2. Capped by max_rotation_offset
    indexer_offset_cap = RotationPairIndexer(
        max_rotation_offset=15,
        max_rotation_deg=None,
    )
    assert indexer_offset_cap.compute_effective_max_offset(100) == 15

    # 3. Capped by max_rotation_deg (+-30 deg on 360 frames -> 30 frames)
    indexer_deg_cap = RotationPairIndexer(
        max_rotation_offset=None,
        max_rotation_deg=30.0,
    )
    # floor(360 * (30 / 360)) = 30
    assert indexer_deg_cap.compute_effective_max_offset(360) == 30

    # 4. Clamped to at least 1 even for tiny sequence
    indexer_tiny = RotationPairIndexer(max_rotation_deg=1.0)
    assert indexer_tiny.compute_effective_max_offset(10) == 1


def test_rotation_indexer_boundary_reflection() -> None:
    """Verify boundary reflection when target candidate exceeds sequence limits."""
    n_frames = 100

    # Case 1: Within bounds -> direct addition
    j1, off1 = RotationPairIndexer.resolve_target_frame(10, 5, n_frames)
    assert j1 == 15
    assert off1 == 5

    # Case 2: Underflow reflected (t=2, offset=-5 -> candidate_j = 2 - (-5) = 7)
    j2, off2 = RotationPairIndexer.resolve_target_frame(2, -5, n_frames)
    assert j2 == 7
    assert off2 == 5  # Sign inverted on reflection

    # Case 3: Overflow reflected (t=98, offset=5 -> candidate_j = 98 - 5 = 93)
    j3, off3 = RotationPairIndexer.resolve_target_frame(98, 5, n_frames)
    assert j3 == 93
    assert off3 == -5  # Sign inverted on reflection

    # Case 4: Double out of bounds (reflection also out) -> periodic modulo fallback
    # t=1, offset=-10: t + offset = -9; candidate_j = 1 - (-10) = 11 (in bounds)
    # To fail both: n_frames=5, t=0, offset=6 -> candidate_j = -6 (out of bounds)
    j4, off4 = RotationPairIndexer.resolve_target_frame(0, 6, 5)
    assert j4 == 1
    assert off4 == 6


def test_rotation_indexer_heading_calculation() -> None:
    """Verify heading angle in degrees and unit vector computation."""
    n_frames = 360

    # 0 offset: 0 degrees, sin=0, cos=1
    deg_0, (s_0, c_0) = RotationPairIndexer.compute_heading(0, n_frames)
    assert math.isclose(deg_0, 0.0, abs_tol=1e-5)
    assert math.isclose(s_0, 0.0, abs_tol=1e-5)
    assert math.isclose(c_0, 1.0, abs_tol=1e-5)

    # +90 offset: +90 degrees, sin=1, cos=0
    deg_90, (s_90, c_90) = RotationPairIndexer.compute_heading(90, n_frames)
    assert math.isclose(deg_90, 90.0, abs_tol=1e-5)
    assert math.isclose(s_90, 1.0, abs_tol=1e-5)
    assert math.isclose(c_90, 0.0, abs_tol=1e-5)

    # -45 offset: -45 degrees, sin=-sqrt(2)/2, cos=sqrt(2)/2
    deg_neg45, (s_neg45, c_neg45) = RotationPairIndexer.compute_heading(-45, n_frames)
    assert math.isclose(deg_neg45, -45.0, abs_tol=1e-5)
    assert math.isclose(s_neg45, -math.sqrt(2) / 2, abs_tol=1e-5)
    assert math.isclose(c_neg45, math.sqrt(2) / 2, abs_tol=1e-5)

    # Unit vector magnitude check
    assert math.isclose(math.hypot(s_neg45, c_neg45), 1.0, rel_tol=1e-6)


def test_rotation_indexer_angle_bounds() -> None:
    """Verify rotation pairs strictly respect max_rotation_deg bound."""
    indexer_30 = RotationPairIndexer(max_rotation_deg=30.0)
    assert indexer_30.is_within_angle_bound(29.5)
    assert indexer_30.is_within_angle_bound(30.0)
    assert not indexer_30.is_within_angle_bound(30.5)
    assert indexer_30.is_within_angle_bound(-30.0)
    assert not indexer_30.is_within_angle_bound(-30.1)

    # Sample pairs on 360 frames
    pairs = indexer_30.sample_pairs(
        n_frames=360,
        samples_per_video=100,
        rng=random.Random(42),
    )
    assert len(pairs) == 100
    for t, j, (s, c) in pairs:
        assert 0 <= t < 360
        assert 0 <= j < 360
        deg = abs(math.degrees(math.atan2(s, c)))
        assert deg <= 30.0 + 1e-4

    # Unbounded indexer allows larger angles
    indexer_unbounded = RotationPairIndexer(max_rotation_deg=None)
    pairs_unbounded = indexer_unbounded.sample_pairs(
        n_frames=360,
        samples_per_video=100,
        rng=random.Random(42),
    )
    angles = [abs(math.degrees(math.atan2(s, c))) for _, _, (s, c) in pairs_unbounded]
    assert any(a > 45.0 for a in angles)


def test_rotation_indexer_short_sequences() -> None:
    """Verify empty pair lists returned for sequences with 1 or 0 frames."""
    indexer = RotationPairIndexer()
    assert indexer.sample_pairs(0) == []
    assert indexer.sample_pairs(1) == []


def test_rotation_indexer_synthetic_heading() -> None:
    """Verify synthetic heading generates angles within configured bounds."""
    indexer_30 = RotationPairIndexer(max_rotation_deg=30.0)
    rng = random.Random(1234)

    for _ in range(50):
        angle_rad, (s, c) = indexer_30.sample_synthetic_heading(rng)
        deg = abs(math.degrees(angle_rad))
        assert deg <= 30.0 + 1e-4
        assert math.isclose(math.hypot(s, c), 1.0, rel_tol=1e-5)


def test_motion_pair_sampler_adapter() -> None:
    """Verify MotionPairSampler correctly delegates and exposes properties."""
    sampler = MotionPairSampler(
        video_fps=60.0,
        straight_video_speed_mps=2.5,
        buffer_distance_meters=2.0,
        max_rotation_deg=30.0,
    )

    # Property forwards
    assert sampler.buffer_frames == 48
    assert math.isclose(sampler.buffer_distance_meters, 2.0)
    assert math.isclose(sampler.step_distance_meters, 2.5 / 60.0)
    assert sampler.max_rotation_deg == 30.0

    # Delegation of straight pairs
    st_pairs = sampler.sample_straight_pairs(
        n_frames=120,
        samples_per_video=10,
        rng=random.Random(101),
    )
    assert len(st_pairs) == 10
    for t, j, dist in st_pairs:
        assert 48 <= t < 72
        assert 48 < j < 72
        assert math.isclose(dist, (j - t) * sampler.step_distance_meters)

    # Delegation of rotation pairs
    ro_pairs = sampler.sample_rotation_pairs(
        n_frames=120,
        samples_per_video=10,
        rng=random.Random(101),
    )
    assert len(ro_pairs) == 10
    for _, _, (s, c) in ro_pairs:
        assert math.isclose(math.hypot(s, c), 1.0, rel_tol=1e-5)

    # Delegation of synthetic conditions
    syn_dist = sampler.sample_synthetic_distance(0.1, 1.0)
    assert 0.1 <= syn_dist <= 1.0

    syn_rad, (s_syn, c_syn) = sampler.sample_synthetic_heading()
    assert abs(math.degrees(syn_rad)) <= 30.0 + 1e-4
    assert math.isclose(math.hypot(s_syn, c_syn), 1.0, rel_tol=1e-5)


def test_motion_pair_sampler_custom_indexers() -> None:
    """Verify MotionPairSampler accepts pre-instantiated custom indexers."""
    st_indexer = StraightPairIndexer(buffer_frames=10, step_distance_meters=0.1)
    ro_indexer = RotationPairIndexer(max_rotation_deg=15.0)

    sampler = MotionPairSampler(
        straight_indexer=st_indexer,
        rotation_indexer=ro_indexer,
    )

    assert sampler.straight_indexer is st_indexer
    assert sampler.rotation_indexer is ro_indexer
    assert sampler.buffer_frames == 10
    assert sampler.step_distance_meters == 0.1
    assert sampler.max_rotation_deg == 15.0

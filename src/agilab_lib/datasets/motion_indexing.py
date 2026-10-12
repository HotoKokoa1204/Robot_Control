"""Module: motion_indexing
Stage: Library
Author: KafuuChino
Date: 2026-10-12
Description: Geometric motion pairing indexers for straight and rotational paths.
"""

import math
import random
from typing import List, Optional, Tuple


class StraightPairIndexer:
    """Pure geometric indexer for forward straight-path video motion pairing.

    Calculates valid frame boundaries excluding accelerating and braking buffer
    margins, computes physical displacements from frame offsets, and generates
    paired frame index tuples (t, j, distance).
    """

    def __init__(
        self,
        video_fps: float = 60.0,
        straight_video_speed_mps: float = 2.5,
        step_distance_meters: Optional[float] = None,
        min_forward_offset: int = 1,
        max_forward_offset: int = 72,
        buffer_distance_meters: float = 2.0,
        buffer_frames: Optional[int] = None,
    ) -> None:
        """Initialize StraightPairIndexer.

        Args:
            video_fps: Video frame rate in frames per second.
            straight_video_speed_mps: Robot cruising velocity in meters/second.
            step_distance_meters: Physical displacement per frame. If None,
                computed as straight_video_speed_mps / max(video_fps, 1.0).
            min_forward_offset: Minimum frame offset between base and target frames.
            max_forward_offset: Maximum frame offset between base and target frames.
            buffer_distance_meters: Physical margin in meters to exclude from video
                ends (acceleration and deceleration zones). Defaults to 2.0.
            buffer_frames: Frame margin to exclude from video ends. Overrides
                buffer_distance_meters if specified.
        """
        self.video_fps = float(video_fps)
        self.straight_video_speed_mps = float(straight_video_speed_mps)

        if step_distance_meters is not None:
            self.step_distance_meters = float(step_distance_meters)
        else:
            self.step_distance_meters = self.straight_video_speed_mps / max(
                self.video_fps, 1.0
            )

        if buffer_frames is not None:
            self.buffer_frames = max(0, int(buffer_frames))
            self.buffer_distance_meters = self.buffer_frames * self.step_distance_meters
        elif buffer_distance_meters is not None and buffer_distance_meters > 0:
            self.buffer_distance_meters = float(buffer_distance_meters)
            self.buffer_frames = int(
                self.buffer_distance_meters / self.step_distance_meters
            )
        else:
            self.buffer_distance_meters = (
                float(buffer_distance_meters)
                if buffer_distance_meters is not None
                else 0.0
            )
            self.buffer_frames = 0

        self.min_forward_offset = max(1, int(min_forward_offset))
        self.max_forward_offset = max(1, int(max_forward_offset))

    def compute_valid_frame_range(self, n_frames: int) -> Tuple[int, int]:
        """Compute valid frame interval [start, end) excluding buffer margins.

        Args:
            n_frames: Total number of frames in the video sequence.

        Returns:
            Tuple of (start_frame, end_frame). If sequence is too short,
            end_frame - start_frame <= min_forward_offset.
        """
        start = self.buffer_frames
        end = n_frames - self.buffer_frames
        return start, end

    def compute_distance(self, offset: int) -> float:
        """Compute physical forward distance in meters for a given frame offset.

        Args:
            offset: Number of frames between base and target frames.

        Returns:
            Physical forward distance in meters.
        """
        return float(offset) * self.step_distance_meters

    def sample_pairs(
        self,
        n_frames: int,
        samples_per_video: Optional[int] = None,
        samples_per_frame: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int, float]]:
        """Generate straight motion frame index pairs and distances for a sequence.

        Args:
            n_frames: Total number of frames in the video sequence.
            samples_per_video: Target number of pairs to sample. If None,
                samples_per_frame is sampled for each valid base frame.
            samples_per_frame: Number of target samples generated per base frame.
            rng: Optional random.Random instance for deterministic sampling.

        Returns:
            List of tuples (t, j, distance) where t is base frame, j is target frame,
            and distance is physical displacement in meters.
        """
        if n_frames <= 1:
            return []

        s, e = self.compute_valid_frame_range(n_frames)
        if e - s <= self.min_forward_offset:
            return []

        active_rng = rng if rng is not None else random.Random()
        pairs: List[Tuple[int, int, float]] = []

        if samples_per_video is not None:
            for _ in range(samples_per_video):
                if e - 1 - self.min_forward_offset < s:
                    continue
                t = active_rng.randint(s, e - 1 - self.min_forward_offset)
                max_diff = min(e - 1 - t, self.max_forward_offset)
                if max_diff < self.min_forward_offset:
                    continue
                offset = active_rng.randint(self.min_forward_offset, max_diff)
                j = t + offset
                dist = self.compute_distance(offset)
                pairs.append((t, j, dist))
        else:
            for t in range(s, e - self.min_forward_offset):
                for _ in range(samples_per_frame):
                    max_diff = min(e - 1 - t, self.max_forward_offset)
                    if max_diff < self.min_forward_offset:
                        continue
                    offset = active_rng.randint(self.min_forward_offset, max_diff)
                    j = t + offset
                    dist = self.compute_distance(offset)
                    pairs.append((t, j, dist))

        return pairs

    @staticmethod
    def sample_synthetic_distance(
        min_distance: float,
        max_distance: float,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Sample a uniform distance value for synthetic mock datasets.

        Args:
            min_distance: Minimum displacement in meters.
            max_distance: Maximum displacement in meters.
            rng: Optional random.Random instance.

        Returns:
            Uniformly sampled distance in meters.
        """
        active_rng = rng if rng is not None else random.Random()
        return active_rng.uniform(min_distance, max_distance)


class RotationPairIndexer:
    """Pure geometric indexer for in-place rotation video motion pairing.

    Calculates angular offsets, enforces calibrated maximum angle bounds
    (e.g., +-30 degrees), resolves target indices with boundary reflection,
    and computes [sin(theta), cos(theta)] unit vectors.
    """

    def __init__(
        self,
        max_rotation_offset: Optional[int] = None,
        max_rotation_deg: Optional[float] = 30.0,
    ) -> None:
        """Initialize RotationPairIndexer.

        Args:
            max_rotation_offset: Maximum frame offset between rotation pair frames.
                If None, pairs cover full 360-degree rotation up to +-180 deg.
            max_rotation_deg: Maximum rotation angle in degrees (e.g. 30.0 restricts
                sampled pairs to [-30 deg, +30 deg]). Defaults to 30.0.
        """
        self.max_rotation_offset = (
            max(1, int(max_rotation_offset))
            if max_rotation_offset is not None
            else None
        )
        self.max_rotation_deg = (
            float(max_rotation_deg) if max_rotation_deg is not None else None
        )

    def compute_effective_max_offset(self, n_frames: int) -> int:
        """Calculate effective maximum frame offset given sequence constraints.

        Args:
            n_frames: Total number of frames in the rotation video.

        Returns:
            Maximum frame offset magnitude clamped by rotation constraints.
        """
        half_turn = n_frames // 2
        eff_max_offset = (
            min(self.max_rotation_offset, half_turn)
            if self.max_rotation_offset is not None
            else half_turn
        )
        if self.max_rotation_deg is not None:
            max_deg_offset = int(
                math.floor(float(n_frames) * (float(self.max_rotation_deg) / 360.0))
            )
            eff_max_offset = min(eff_max_offset, max_deg_offset)
        return max(1, eff_max_offset)

    @staticmethod
    def resolve_target_frame(t: int, offset: int, n_frames: int) -> Tuple[int, int]:
        """Resolve target frame index using boundary reflection and periodic wrapping.

        Args:
            t: Base frame index.
            offset: Candidate signed frame offset.
            n_frames: Sequence frame count.

        Returns:
            Tuple of (target_frame_index, effective_offset).
        """
        if 0 <= t + offset < n_frames:
            return t + offset, offset
        candidate_j = t - offset
        if 0 <= candidate_j < n_frames:
            return candidate_j, -offset
        return (t + offset) % n_frames, offset

    @staticmethod
    def compute_heading(
        offset: int,
        n_frames: int,
    ) -> Tuple[float, Tuple[float, float]]:
        """Compute heading angle in degrees and [sin, cos] vector for a frame offset.

        Args:
            offset: Signed frame offset from base to target.
            n_frames: Total number of frames in the sequence.

        Returns:
            Tuple of (angle_degrees, (sin_theta, cos_theta)).
        """
        angle_raw = float(offset) * (360.0 / max(float(n_frames), 1.0))
        angle = ((angle_raw + 180.0) % 360.0) - 180.0
        rad = angle * math.pi / 180.0
        sin_cos = (math.sin(rad), math.cos(rad))
        return angle, sin_cos

    def is_within_angle_bound(self, angle_deg: float) -> bool:
        """Check whether a rotation angle is within configured bounds.

        Args:
            angle_deg: Rotation angle in degrees in [-180, 180).

        Returns:
            True if angle satisfies configured max_rotation_deg, False otherwise.
        """
        if (
            self.max_rotation_deg is not None
            and abs(angle_deg) > float(self.max_rotation_deg) + 1e-4
        ):
            return False
        return True

    def sample_pairs(
        self,
        n_frames: int,
        samples_per_video: Optional[int] = None,
        samples_per_frame: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Generate rotation motion frame index pairs and sin/cos conditions.

        Args:
            n_frames: Total number of frames in the rotation video.
            samples_per_video: Target number of pairs to sample. If None,
                samples_per_frame is sampled for each base frame.
            samples_per_frame: Number of target samples generated per base frame.
            rng: Optional random.Random instance for deterministic sampling.

        Returns:
            List of tuples (t, j, (sin_theta, cos_theta)).
        """
        if n_frames <= 1:
            return []

        eff_max_offset = self.compute_effective_max_offset(n_frames)
        active_rng = rng if rng is not None else random.Random()
        pairs: List[Tuple[int, int, Tuple[float, float]]] = []

        if samples_per_video is not None:
            for _ in range(samples_per_video):
                t = active_rng.randint(0, n_frames - 1)
                offset = active_rng.randint(-eff_max_offset, eff_max_offset)
                if offset == 0:
                    offset = 1 if active_rng.random() < 0.5 else -1
                j, eff_offset = self.resolve_target_frame(t, offset, n_frames)
                angle, sin_cos = self.compute_heading(eff_offset, n_frames)
                if not self.is_within_angle_bound(angle):
                    continue
                pairs.append((t, j, sin_cos))
        else:
            for t in range(n_frames):
                for _ in range(samples_per_frame):
                    offset = active_rng.randint(-eff_max_offset, eff_max_offset)
                    if offset == 0:
                        offset = 1 if active_rng.random() < 0.5 else -1
                    j, eff_offset = self.resolve_target_frame(t, offset, n_frames)
                    angle, sin_cos = self.compute_heading(eff_offset, n_frames)
                    if not self.is_within_angle_bound(angle):
                        continue
                    pairs.append((t, j, sin_cos))

        return pairs

    def sample_synthetic_heading(
        self,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """Sample a synthetic random rotation angle and [sin, cos] unit vector.

        Args:
            rng: Optional random.Random instance.

        Returns:
            Tuple of (angle_radians, (sin_theta, cos_theta)).
        """
        active_rng = rng if rng is not None else random.Random()
        if self.max_rotation_deg is not None:
            max_rad = math.radians(float(self.max_rotation_deg))
            angle_rad = active_rng.uniform(-max_rad, max_rad)
        else:
            angle_rad = active_rng.uniform(-math.pi, math.pi)
        sin_cos = (math.sin(angle_rad), math.cos(angle_rad))
        return angle_rad, sin_cos


class MotionPairSampler:
    """Unified adapter coordinating straight and rotation motion pair indexers.

    Acts as a facade delegating geometric calculations to StraightPairIndexer
    and RotationPairIndexer, isolating mathematical indexing from disk video I/O.
    """

    def __init__(
        self,
        straight_indexer: Optional[StraightPairIndexer] = None,
        rotation_indexer: Optional[RotationPairIndexer] = None,
        video_fps: float = 60.0,
        straight_video_speed_mps: float = 2.5,
        step_distance_meters: Optional[float] = None,
        min_forward_offset: int = 1,
        max_forward_offset: int = 72,
        buffer_distance_meters: float = 2.0,
        buffer_frames: Optional[int] = None,
        max_rotation_offset: Optional[int] = None,
        max_rotation_deg: Optional[float] = 30.0,
    ) -> None:
        """Initialize MotionPairSampler.

        Args:
            straight_indexer: Optional pre-configured StraightPairIndexer.
            rotation_indexer: Optional pre-configured RotationPairIndexer.
            video_fps: Video frame rate in frames per second.
            straight_video_speed_mps: Robot cruising velocity in meters/second.
            step_distance_meters: Forward displacement per frame offset in meters.
            min_forward_offset: Minimum frame offset for straight pairs.
            max_forward_offset: Maximum frame offset for straight pairs.
            buffer_distance_meters: Margin in meters excluded from video ends.
            buffer_frames: Explicit frame count to exclude from video ends.
            max_rotation_offset: Maximum frame offset for rotation pairs.
            max_rotation_deg: Maximum rotation angle in degrees.
        """
        if straight_indexer is not None:
            self.straight_indexer = straight_indexer
        else:
            self.straight_indexer = StraightPairIndexer(
                video_fps=video_fps,
                straight_video_speed_mps=straight_video_speed_mps,
                step_distance_meters=step_distance_meters,
                min_forward_offset=min_forward_offset,
                max_forward_offset=max_forward_offset,
                buffer_distance_meters=buffer_distance_meters,
                buffer_frames=buffer_frames,
            )

        if rotation_indexer is not None:
            self.rotation_indexer = rotation_indexer
        else:
            self.rotation_indexer = RotationPairIndexer(
                max_rotation_offset=max_rotation_offset,
                max_rotation_deg=max_rotation_deg,
            )

    @property
    def buffer_frames(self) -> int:
        """Frame margin excluded from video ends."""
        return self.straight_indexer.buffer_frames

    @property
    def buffer_distance_meters(self) -> float:
        """Physical margin in meters excluded from video ends."""
        return self.straight_indexer.buffer_distance_meters

    @property
    def step_distance_meters(self) -> float:
        """Physical displacement per frame offset in meters."""
        return self.straight_indexer.step_distance_meters

    @property
    def min_forward_offset(self) -> int:
        """Minimum frame offset for forward pairs."""
        return self.straight_indexer.min_forward_offset

    @property
    def max_forward_offset(self) -> int:
        """Maximum frame offset for forward pairs."""
        return self.straight_indexer.max_forward_offset

    @property
    def max_rotation_offset(self) -> Optional[int]:
        """Maximum frame offset for rotation pairs."""
        return self.rotation_indexer.max_rotation_offset

    @property
    def max_rotation_deg(self) -> Optional[float]:
        """Maximum rotation angle bound in degrees."""
        return self.rotation_indexer.max_rotation_deg

    def sample_straight_pairs(
        self,
        n_frames: int,
        samples_per_video: Optional[int] = None,
        samples_per_frame: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int, float]]:
        """Sample straight-path forward motion pairs (t, j, distance).

        Args:
            n_frames: Total number of frames in the sequence.
            samples_per_video: Target number of pairs to sample.
            samples_per_frame: Samples per base frame if samples_per_video is None.
            rng: Optional random.Random instance.

        Returns:
            List of tuples (t, j, distance).
        """
        return self.straight_indexer.sample_pairs(
            n_frames=n_frames,
            samples_per_video=samples_per_video,
            samples_per_frame=samples_per_frame,
            rng=rng,
        )

    def sample_rotation_pairs(
        self,
        n_frames: int,
        samples_per_video: Optional[int] = None,
        samples_per_frame: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Sample in-place rotation motion pairs (t, j, (sin_theta, cos_theta)).

        Args:
            n_frames: Total number of frames in the sequence.
            samples_per_video: Target number of pairs to sample.
            samples_per_frame: Samples per base frame if samples_per_video is None.
            rng: Optional random.Random instance.

        Returns:
            List of tuples (t, j, (sin_theta, cos_theta)).
        """
        return self.rotation_indexer.sample_pairs(
            n_frames=n_frames,
            samples_per_video=samples_per_video,
            samples_per_frame=samples_per_frame,
            rng=rng,
        )

    def sample_synthetic_distance(
        self,
        min_distance: float,
        max_distance: float,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Sample synthetic distance using the straight pair indexer.

        Args:
            min_distance: Minimum displacement in meters.
            max_distance: Maximum displacement in meters.
            rng: Optional random.Random instance.

        Returns:
            Sampled distance in meters.
        """
        return self.straight_indexer.sample_synthetic_distance(
            min_distance=min_distance,
            max_distance=max_distance,
            rng=rng,
        )

    def sample_synthetic_heading(
        self,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, Tuple[float, float]]:
        """Sample synthetic rotation heading using the rotation pair indexer.

        Args:
            rng: Optional random.Random instance.

        Returns:
            Tuple of (angle_radians, (sin_theta, cos_theta)).
        """
        return self.rotation_indexer.sample_synthetic_heading(rng=rng)

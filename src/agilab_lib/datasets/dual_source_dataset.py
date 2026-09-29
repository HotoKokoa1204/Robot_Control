"""Module: dual_source_dataset
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Paired video datasets for concurrent straight and rotation dynamics.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DualSourceBatch(dict):
    """Structured container mapping keys to dual-source paired batch tensors.

    Inherits from dict so that standard key indexing (`batch['fwd_current']`),
    property access (`batch.fwd_current`), and `isinstance(batch, dict)` all
    work transparently with PyTorch's default collate and custom data pipelines.
    """

    def __init__(
        self,
        recon_frame: torch.Tensor,
        fwd_current: torch.Tensor,
        fwd_target: torch.Tensor,
        fwd_distance: torch.Tensor,
        rot_current: torch.Tensor,
        rot_target: torch.Tensor,
        rot_sin_cos: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Initialize DualSourceBatch.

        Args:
            recon_frame: Frame tensor for reconstruction branch, shape (3, H, W)
                or (B, 3, H, W).
            fwd_current: Base frame tensor for forward dynamics, shape (3, H, W)
                or (B, 3, H, W).
            fwd_target: Future frame tensor for forward dynamics, shape (3, H, W)
                or (B, 3, H, W).
            fwd_distance: Forward scalar distance tensor, shape (1,) or (B, 1).
            rot_current: Base frame tensor for rotation dynamics, shape (3, H, W)
                or (B, 3, H, W).
            rot_target: Future frame tensor for rotation dynamics, shape (3, H, W)
                or (B, 3, H, W).
            rot_sin_cos: Rotation unit vector tensor [sin(theta), cos(theta)],
                shape (2,) or (B, 2).
            **kwargs: Additional optional metadata or auxiliary tensors.
        """
        super().__init__(
            recon_frame=recon_frame,
            fwd_current=fwd_current,
            fwd_target=fwd_target,
            fwd_distance=fwd_distance,
            rot_current=rot_current,
            rot_target=rot_target,
            rot_sin_cos=rot_sin_cos,
            **kwargs,
        )

    @property
    def recon_frame(self) -> torch.Tensor:
        """Frame tensor for reconstruction branch."""
        return self["recon_frame"]

    @property
    def fwd_current(self) -> torch.Tensor:
        """Base frame tensor for forward dynamics."""
        return self["fwd_current"]

    @property
    def fwd_target(self) -> torch.Tensor:
        """Future target frame tensor for forward dynamics."""
        return self["fwd_target"]

    @property
    def fwd_distance(self) -> torch.Tensor:
        """Forward displacement scalar distance in meters."""
        return self["fwd_distance"]

    @property
    def rot_current(self) -> torch.Tensor:
        """Base frame tensor for rotation dynamics."""
        return self["rot_current"]

    @property
    def rot_target(self) -> torch.Tensor:
        """Future target frame tensor for rotation dynamics."""
        return self["rot_target"]

    @property
    def rot_sin_cos(self) -> torch.Tensor:
        """Rotation unit vector [sin(theta), cos(theta)]."""
        return self["rot_sin_cos"]

    def to(self, device: Union[str, torch.device]) -> "DualSourceBatch":
        """Transfer all batch tensors to the specified compute device.

        Args:
            device: Target torch device or device string identifier.

        Returns:
            New DualSourceBatch with all tensors moved to device.
        """
        transferred = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.items()
        }
        return DualSourceBatch(**transferred)


def dual_source_collate_fn(
    batch_list: Sequence[Dict[str, torch.Tensor]],
) -> DualSourceBatch:
    """Collate a sequence of sample dictionaries into a batched DualSourceBatch.

    Args:
        batch_list: Sequence of sample dictionaries or DualSourceBatch instances.

    Returns:
        Batched DualSourceBatch with stacked tensors along dimension 0.
    """
    keys = (
        "recon_frame",
        "fwd_current",
        "fwd_target",
        "fwd_distance",
        "rot_current",
        "rot_target",
        "rot_sin_cos",
    )
    collated: Dict[str, Any] = {}
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch_list], dim=0)

    first_item = batch_list[0]
    for key, val in first_item.items():
        if key not in collated:
            if isinstance(val, torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch_list], dim=0)
            else:
                collated[key] = [item[key] for item in batch_list]

    return DualSourceBatch(**collated)


class DummyDualSourceVideoDataset(Dataset[DualSourceBatch]):
    """Synthetic dataset generating paired multi-branch video batches for testing.

    Yields mock RGB video frames with shape (3, H, W) and normalized motion
    condition vectors for straight-path and rotation branches without requiring
    raw video assets on disk.
    """

    def __init__(
        self,
        num_samples: int = 32,
        img_height: int = 108,
        img_width: int = 192,
        min_distance: float = 0.05,
        max_distance: float = 3.0,
        recon_source: str = "forward",
        seed: Optional[int] = None,
    ) -> None:
        """Initialize DummyDualSourceVideoDataset.

        Args:
            num_samples: Number of synthetic paired samples to generate.
            img_height: Target frame height in pixels.
            img_width: Target frame width in pixels.
            min_distance: Minimum forward displacement in meters.
            max_distance: Maximum forward displacement in meters.
            recon_source: Modality supplying reconstruction frame ('forward',
                'rotation', or 'random').
            seed: Optional random seed for reproducible sample generation.
        """
        super().__init__()
        self.num_samples = int(num_samples)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.recon_source = str(recon_source).lower()
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def __len__(self) -> int:
        """Return total number of synthetic paired samples.

        Returns:
            Synthetic sample count.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> DualSourceBatch:
        """Generate a synthetic paired sample item.

        Args:
            idx: Index of sample.

        Returns:
            DualSourceBatch containing mock frame tensors and motion conditions.
        """
        if self.seed is not None:
            gen_rng = random.Random(self.seed + idx * 7919)
            torch_gen = torch.Generator().manual_seed(self.seed + idx * 7919)
            fwd_current = torch.rand(
                3,
                self.img_height,
                self.img_width,
                generator=torch_gen,
                dtype=torch.float32,
            )
            fwd_target = torch.rand(
                3,
                self.img_height,
                self.img_width,
                generator=torch_gen,
                dtype=torch.float32,
            )
            dist_val = gen_rng.uniform(self.min_distance, self.max_distance)
            rot_current = torch.rand(
                3,
                self.img_height,
                self.img_width,
                generator=torch_gen,
                dtype=torch.float32,
            )
            rot_target = torch.rand(
                3,
                self.img_height,
                self.img_width,
                generator=torch_gen,
                dtype=torch.float32,
            )
            angle_rad = gen_rng.uniform(-math.pi, math.pi)
            use_fwd = gen_rng.random() < 0.5
        else:
            fwd_current = torch.rand(
                3, self.img_height, self.img_width, dtype=torch.float32
            )
            fwd_target = torch.rand(
                3, self.img_height, self.img_width, dtype=torch.float32
            )
            dist_val = self._rng.uniform(self.min_distance, self.max_distance)
            rot_current = torch.rand(
                3, self.img_height, self.img_width, dtype=torch.float32
            )
            rot_target = torch.rand(
                3, self.img_height, self.img_width, dtype=torch.float32
            )
            angle_rad = self._rng.uniform(-math.pi, math.pi)
            use_fwd = self._rng.random() < 0.5

        fwd_distance = torch.tensor([dist_val], dtype=torch.float32)
        rot_sin_cos = torch.tensor(
            [math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32
        )

        if self.recon_source == "forward":
            recon_frame = fwd_current.clone()
        elif self.recon_source == "rotation":
            recon_frame = rot_current.clone()
        else:
            recon_frame = fwd_current.clone() if use_fwd else rot_current.clone()

        return DualSourceBatch(
            recon_frame=recon_frame,
            fwd_current=fwd_current,
            fwd_target=fwd_target,
            fwd_distance=fwd_distance,
            rot_current=rot_current,
            rot_target=rot_target,
            rot_sin_cos=rot_sin_cos,
        )

    def sample_batch(self, batch_size: int = 16) -> DualSourceBatch:
        """Sample an ad-hoc collated batch without creating a DataLoader.

        Args:
            batch_size: Number of paired samples in the extracted batch.

        Returns:
            Batched DualSourceBatch with shape (B, ...).
        """
        indices = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        samples = [self[i] for i in indices]
        return dual_source_collate_fn(samples)


class DualSourceVideoDataset(Dataset[DualSourceBatch]):
    """Dual-source paired video dataset sampling straight paths and in-place rotations.

    Samples concurrent pairs from forward motion videos and rotation videos,
    providing aligned visual dynamics for multi-branch latent navigation training.
    """

    def __init__(
        self,
        one_path_dir: Union[str, Path] = "data/one_path",
        rotation_dir: Union[str, Path] = "data/360",
        img_height: int = 108,
        img_width: int = 192,
        video_fps: float = 60.0,
        straight_video_speed_mps: float = 2.5,
        step_distance_meters: Optional[float] = None,
        max_forward_offset: int = 72,
        min_forward_offset: int = 1,
        max_rotation_offset: Optional[int] = None,
        buffer_distance_meters: Optional[float] = None,
        buffer_frames: Optional[int] = None,
        samples_per_video: Optional[int] = 50,
        samples_per_frame: int = 1,
        max_videos_per_source: Optional[int] = None,
        preload_frames: bool = False,
        recon_source: str = "forward",
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize DualSourceVideoDataset.

        Args:
            one_path_dir: Directory containing straight-path forward videos.
            rotation_dir: Directory containing 360 in-place rotation videos.
            img_height: Target frame height in pixels.
            img_width: Target frame width in pixels.
            video_fps: Video frame rate in frames per second.
            straight_video_speed_mps: Robot cruise forward velocity in meters/sec.
            step_distance_meters: Forward displacement per frame offset. If None,
                computed as straight_video_speed_mps / video_fps.
            max_forward_offset: Maximum frame offset between forward pair frames.
            min_forward_offset: Minimum frame offset between forward pair frames.
            max_rotation_offset: Maximum frame offset between rotation pair frames.
                If None, pairs cover full 360-degree rotation up to +-180 deg.
            buffer_distance_meters: Physical margin in meters to exclude from video
                ends to omit acceleration and deceleration zones.
            buffer_frames: Frame margin to exclude from video ends. Overrides
                buffer_distance_meters if specified.
            samples_per_video: Target number of pairs to sample per video. If None,
                samples_per_frame is used for every frame.
            samples_per_frame: Number of target samples generated per base frame.
            max_videos_per_source: Optional limit on the number of videos indexed
                per modality.
            preload_frames: Whether to pre-extract and store all video frames
                in memory as uint8 arrays for faster iteration.
            recon_source: Modality supplying reconstruction frame ('forward',
                'rotation', or 'random').
            num_samples: Explicit dataset length override. If None, set to the
                maximum of forward and rotation pair counts.
            seed: Optional random seed for deterministic sampling.

        Raises:
            FileNotFoundError: If one_path_dir or rotation_dir cannot be found.
            ValueError: If no video files are discovered in either directory.
        """
        super().__init__()
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.video_fps = float(video_fps)
        self.straight_video_speed_mps = float(straight_video_speed_mps)
        self.preload_frames = bool(preload_frames)
        self.recon_source = str(recon_source).lower()

        if step_distance_meters is not None:
            self.step_distance_meters = float(step_distance_meters)
        else:
            self.step_distance_meters = self.straight_video_speed_mps / max(
                self.video_fps, 1.0
            )

        if buffer_frames is not None:
            self.buffer_frames = max(0, int(buffer_frames))
        elif buffer_distance_meters is not None and buffer_distance_meters > 0:
            self.buffer_frames = int(buffer_distance_meters / self.step_distance_meters)
        else:
            self.buffer_frames = 0

        self.max_forward_offset = max(1, int(max_forward_offset))
        self.min_forward_offset = max(1, int(min_forward_offset))
        self.max_rotation_offset = (
            max(1, int(max_rotation_offset))
            if max_rotation_offset is not None
            else None
        )

        resolved_one_path = self._resolve_dir(one_path_dir)
        resolved_rotation = self._resolve_dir(rotation_dir)

        if not resolved_one_path.exists():
            raise FileNotFoundError(
                f"Straight-path video directory not found: {one_path_dir} "
                f"(resolved: {resolved_one_path})"
            )
        if not resolved_rotation.exists():
            raise FileNotFoundError(
                f"Rotation video directory not found: {rotation_dir} "
                f"(resolved: {resolved_rotation})"
            )

        self.fwd_video_paths = self._find_video_files(
            resolved_one_path, max_videos_per_source
        )
        self.rot_video_paths = self._find_video_files(
            resolved_rotation, max_videos_per_source
        )

        if not self.fwd_video_paths:
            raise ValueError(
                f"No video files found in straight-path directory: {resolved_one_path}"
            )
        if not self.rot_video_paths:
            raise ValueError(
                f"No video files found in rotation directory: {resolved_rotation}"
            )

        rng = random.Random(seed) if seed is not None else random.Random()

        # Cache preloaded videos if requested
        self._cached_videos: Dict[Path, np.ndarray] = {}
        if self.preload_frames:
            all_paths = list(set(self.fwd_video_paths + self.rot_video_paths))
            self._cached_videos = self._preload_all_videos(all_paths)

        # Index forward pairs
        self.fwd_pairs: List[Tuple[Path, int, int, float]] = []
        for vp in self.fwd_video_paths:
            n_frames = (
                len(self._cached_videos[vp])
                if vp in self._cached_videos
                else self._probe_video_frame_count(vp)
            )
            if n_frames <= 1:
                continue

            s = self.buffer_frames
            e = n_frames - self.buffer_frames
            if e - s <= 1:
                s, e = 0, n_frames

            if samples_per_video is not None:
                for _ in range(samples_per_video):
                    if e - 1 <= s:
                        continue
                    t = rng.randint(s, e - 2)
                    max_diff = min(e - 1 - t, self.max_forward_offset)
                    if max_diff < self.min_forward_offset:
                        continue
                    offset = rng.randint(self.min_forward_offset, max_diff)
                    j = t + offset
                    dist = offset * self.step_distance_meters
                    self.fwd_pairs.append((vp, t, j, dist))
            else:
                for t in range(s, e - 1):
                    for _ in range(samples_per_frame):
                        max_diff = min(e - 1 - t, self.max_forward_offset)
                        if max_diff < self.min_forward_offset:
                            continue
                        offset = rng.randint(self.min_forward_offset, max_diff)
                        j = t + offset
                        dist = offset * self.step_distance_meters
                        self.fwd_pairs.append((vp, t, j, dist))

        # Index rotation pairs
        self.rot_pairs: List[Tuple[Path, int, int, Tuple[float, float]]] = []
        for vp in self.rot_video_paths:
            n_frames = (
                len(self._cached_videos[vp])
                if vp in self._cached_videos
                else self._probe_video_frame_count(vp)
            )
            if n_frames <= 1:
                continue

            half_turn = n_frames // 2
            eff_max_offset = (
                min(self.max_rotation_offset, half_turn)
                if self.max_rotation_offset is not None
                else half_turn
            )
            eff_max_offset = max(1, eff_max_offset)

            if samples_per_video is not None:
                for _ in range(samples_per_video):
                    t = rng.randint(0, n_frames - 1)
                    offset = rng.randint(-eff_max_offset, eff_max_offset)
                    if offset == 0:
                        offset = 1 if rng.random() < 0.5 else -1
                    j = (t + offset) % n_frames
                    angle_raw = float(offset) * (360.0 / max(float(n_frames), 1.0))
                    angle = ((angle_raw + 180.0) % 360.0) - 180.0
                    rad = angle * math.pi / 180.0
                    sin_cos = (math.sin(rad), math.cos(rad))
                    self.rot_pairs.append((vp, t, j, sin_cos))
            else:
                for t in range(n_frames):
                    for _ in range(samples_per_frame):
                        offset = rng.randint(-eff_max_offset, eff_max_offset)
                        if offset == 0:
                            offset = 1 if rng.random() < 0.5 else -1
                        j = (t + offset) % n_frames
                        angle_raw = float(offset) * (360.0 / max(float(n_frames), 1.0))
                        angle = ((angle_raw + 180.0) % 360.0) - 180.0
                        rad = angle * math.pi / 180.0
                        sin_cos = (math.sin(rad), math.cos(rad))
                        self.rot_pairs.append((vp, t, j, sin_cos))

        if not self.fwd_pairs:
            raise ValueError(
                "No valid forward frame pairs could be generated. "
                "Check video frame counts and buffer configurations."
            )
        if not self.rot_pairs:
            raise ValueError(
                "No valid rotation frame pairs could be generated. "
                "Check video frame counts."
            )

        if num_samples is not None:
            self.length = max(1, int(num_samples))
        else:
            self.length = max(len(self.fwd_pairs), len(self.rot_pairs))

    @staticmethod
    def _resolve_dir(path: Union[str, Path]) -> Path:
        """Resolve directory path against working directory and repo roots.

        Args:
            path: Path string or Path object to resolve.

        Returns:
            Resolved existing Path object if found, else original Path.
        """
        p = Path(path)
        if p.exists():
            return p.resolve()
        candidates = [
            Path.cwd() / p,
            Path("Visual_Navigation_System") / p,
            Path.cwd().parent.parent / p,
            Path.cwd().parent.parent / "Visual_Navigation_System" / p,
            Path.cwd().parent / p,
            Path.cwd().parent / "Visual_Navigation_System" / p,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return p

    @staticmethod
    def _find_video_files(
        directory: Path, max_videos: Optional[int] = None
    ) -> List[Path]:
        """Find all supported video files recursively in the specified directory.

        Args:
            directory: Root directory to search.
            max_videos: Optional cap on the number of discovered video files.

        Returns:
            List of Path objects pointing to discovered video files.
        """
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        videos = sorted(
            [p for p in directory.rglob("*") if p.suffix.lower() in video_exts]
        )
        if max_videos is not None and max_videos > 0:
            videos = videos[:max_videos]
        return videos

    @staticmethod
    def _probe_video_frame_count(video_path: Path) -> int:
        """Query total frame count from video metadata using OpenCV.

        Args:
            video_path: Path to target video file.

        Returns:
            Number of frames recorded in video metadata.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return cnt

    def _preload_all_videos(
        self,
        video_paths: Sequence[Path],
    ) -> Dict[Path, np.ndarray]:
        """Pre-extract and resize all video frames into in-memory uint8 arrays.

        Args:
            video_paths: Sequence of video file paths to preload.

        Returns:
            Mapping from video path to numpy array of shape (N, H, W, 3).
        """
        cached: Dict[Path, np.ndarray] = {}
        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue
            frames: List[np.ndarray] = []
            ret, frame = cap.read()
            while ret:
                resized = cv2.resize(
                    frame,
                    (self.img_width, self.img_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                ret, frame = cap.read()
            cap.release()
            if frames:
                cached[vp] = np.stack(frames, axis=0)
        return cached

    @staticmethod
    def _read_video_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        """Read a specific frame from an open OpenCV VideoCapture object.

        Args:
            cap: Active VideoCapture instance.
            frame_idx: Frame index to seek and decode.

        Returns:
            Decoded BGR image array.

        Raises:
            RuntimeError: If seeking or reading the specified frame fails.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}")
        return frame

    def _load_frame_pair(
        self,
        video_path: Path,
        idx_a: int,
        idx_b: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize a pair of frames from a video.

        Args:
            video_path: Path to target video file.
            idx_a: Index of first frame.
            idx_b: Index of second frame.

        Returns:
            Tuple of two normalized float32 tensors of shape (3, H, W).
        """
        if self.preload_frames and video_path in self._cached_videos:
            video_arr = self._cached_videos[video_path]
            frame_a = video_arr[idx_a]
            frame_b = video_arr[idx_b]
            tensor_a = torch.from_numpy(frame_a).float().permute(2, 0, 1) / 255.0
            tensor_b = torch.from_numpy(frame_b).float().permute(2, 0, 1) / 255.0
            return tensor_a, tensor_b

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        try:
            frame_a = self._read_video_frame_at(cap, idx_a)
            if idx_b == idx_a + 1:
                ret_b, frame_b = cap.read()
                if not ret_b or frame_b is None:
                    frame_b = self._read_video_frame_at(cap, idx_b)
            else:
                frame_b = self._read_video_frame_at(cap, idx_b)
        finally:
            cap.release()

        resized_a = cv2.resize(
            frame_a, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA
        )
        rgb_a = cv2.cvtColor(resized_a, cv2.COLOR_BGR2RGB)
        tensor_a = torch.from_numpy(rgb_a).float().permute(2, 0, 1) / 255.0

        resized_b = cv2.resize(
            frame_b, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA
        )
        rgb_b = cv2.cvtColor(resized_b, cv2.COLOR_BGR2RGB)
        tensor_b = torch.from_numpy(rgb_b).float().permute(2, 0, 1) / 255.0

        return tensor_a, tensor_b

    def __len__(self) -> int:
        """Return total number of paired samples.

        Returns:
            Integer count of paired dataset samples.
        """
        return self.length

    def __getitem__(self, idx: int) -> DualSourceBatch:
        """Retrieve paired straight-path and rotation frames and motion conditions.

        Args:
            idx: Sample index.

        Returns:
            DualSourceBatch containing aligned multi-branch tensors.
        """
        fwd_idx = idx % len(self.fwd_pairs)
        rot_idx = idx % len(self.rot_pairs)

        fwd_path, fwd_t, fwd_j, fwd_dist = self.fwd_pairs[fwd_idx]
        fwd_current, fwd_target = self._load_frame_pair(fwd_path, fwd_t, fwd_j)
        fwd_distance_tensor = torch.tensor([fwd_dist], dtype=torch.float32)

        rot_path, rot_t, rot_j, rot_sc = self.rot_pairs[rot_idx]
        rot_current, rot_target = self._load_frame_pair(rot_path, rot_t, rot_j)
        rot_sin_cos_tensor = torch.tensor(list(rot_sc), dtype=torch.float32)

        if self.recon_source == "forward":
            recon_frame = fwd_current.clone()
        elif self.recon_source == "rotation":
            recon_frame = rot_current.clone()
        else:
            recon_frame = (
                fwd_current.clone() if random.random() < 0.5 else rot_current.clone()
            )

        return DualSourceBatch(
            recon_frame=recon_frame,
            fwd_current=fwd_current,
            fwd_target=fwd_target,
            fwd_distance=fwd_distance_tensor,
            rot_current=rot_current,
            rot_target=rot_target,
            rot_sin_cos=rot_sin_cos_tensor,
        )

    def sample_batch(self, batch_size: int = 16) -> DualSourceBatch:
        """Sample an ad-hoc collated batch without creating a DataLoader.

        Args:
            batch_size: Number of paired samples in the extracted batch.

        Returns:
            Batched DualSourceBatch with shape (B, ...).
        """
        indices = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        samples = [self[i] for i in indices]
        return dual_source_collate_fn(samples)

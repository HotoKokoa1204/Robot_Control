"""
Module: latent_dataset
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Latent trajectory datasets for Residual Latent Transformer.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class InMemoryLatentOffsetDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset storing pre-encoded Latent Vectors and generating offset pairs.

    Stores Latent Vectors extracted from videos and samples index pairs
    with a bounded frame offset.
    """

    def __init__(
        self,
        video_root_dir: Union[str, Path] = "",
        max_frame_offset: int = 10,
        samples_per_base_frame: int = 1,
        autoencoder_model: Optional[nn.Module] = None,
        device_for_ae: str = "cpu",
        ae_encode_batch_size: int = 32,
    ) -> None:
        """Initialize the in-memory latent offset dataset.

        Args:
            video_root_dir: Directory containing video files to encode.
            max_frame_offset: Maximum frame distance between pair indices.
            samples_per_base_frame: Number of target samples per base frame.
            autoencoder_model: Optional Autoencoder model for encoding frames.
            device_for_ae: Compute device string for Autoencoder inference.
            ae_encode_batch_size: Batch size used when encoding video frames.
        """
        super().__init__()
        self.max_frame_offset: int = abs(int(max_frame_offset))
        self.samples_per_base_frame: int = samples_per_base_frame
        self.all_video_latents_cpu: List[torch.Tensor] = []
        self.data_pairs_indices: List[Tuple[int, int, int]] = []

        root_path = Path(video_root_dir) if video_root_dir else None
        if (
            autoencoder_model is not None
            and root_path is not None
            and root_path.exists()
        ):
            self._load_and_encode_videos(
                root_path, autoencoder_model, device_for_ae, ae_encode_batch_size
            )

    def _load_and_encode_videos(
        self,
        video_root_dir: Path,
        ae: nn.Module,
        device: str,
        batch_size: int,
    ) -> None:
        """Scan and encode video files from the root directory.

        Args:
            video_root_dir: Directory containing video files.
            ae: Autoencoder model used to extract Latent Vectors.
            device: Compute device string.
            batch_size: Frame batch size for encoding.
        """
        try:
            import cv2
        except ImportError:
            return

        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        video_paths = [
            p for p in video_root_dir.rglob("*") if p.suffix.lower() in video_exts
        ]

        ae = ae.to(device)
        ae.eval()

        for vid_idx, vp in enumerate(video_paths):
            cap = cv2.VideoCapture(str(vp))
            frames: List[torch.Tensor] = []
            ret, frame = cap.read()
            while ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = (
                    torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                )
                frames.append(frame_tensor)
                ret, frame = cap.read()
            cap.release()

            if not frames:
                continue

            all_frames = torch.stack(frames)
            latents_list: List[torch.Tensor] = []
            with torch.no_grad():
                for start_idx in range(0, len(all_frames), batch_size):
                    batch = all_frames[start_idx : start_idx + batch_size].to(device)
                    z = ae.encode(batch).cpu()
                    latents_list.append(z)

            cat_latents = torch.cat(latents_list, dim=0)
            self.all_video_latents_cpu.append(cat_latents)

            num_frames = len(cat_latents)
            for i in range(num_frames):
                for _ in range(self.samples_per_base_frame):
                    offset = random.randint(
                        -self.max_frame_offset, self.max_frame_offset
                    )
                    j = max(0, min(num_frames - 1, i + offset))
                    self.data_pairs_indices.append((vid_idx, i, j))

    def __len__(self) -> int:
        """Return the number of latent offset pairs in the dataset.

        Returns:
            Total count of index pairs.
        """
        return len(self.data_pairs_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Fetch a base Latent Vector, target Latent Vector, and offset.

        Args:
            idx: Index of the pair sample.

        Returns:
            Tuple of (z_i, z_j, k_diff) representing the start Latent Vector,
            target Latent Vector, and integer frame difference.
        """
        vid_idx, i, j = self.data_pairs_indices[idx]
        latents = self.all_video_latents_cpu[vid_idx]
        z_i = latents[i]
        z_j = latents[j]
        k_diff = j - i
        return z_i, z_j, k_diff


class DummyLatentPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Mock dataset generating pairs of Latent Vectors and target angles.

    Useful for rapid unit testing and pipeline smoke checks without video
    assets.
    """

    def __init__(
        self,
        num_samples: int = 32,
        latent_dim: int = 128,
        return_sin_cos: bool = False,
    ) -> None:
        """Initialize the dummy latent pair dataset.

        Args:
            num_samples: Number of synthetic sample pairs to generate.
            latent_dim: Dimension of each Latent Vector.
            return_sin_cos: Whether to return 2D [sin, cos] vector instead of 1D angle.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.latent_dim: int = latent_dim
        self.return_sin_cos: bool = return_sin_cos

    def __len__(self) -> int:
        """Return the number of samples in the mock dataset.

        Returns:
            Total synthetic sample count.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a random Latent Vector pair and rotation target.

        Args:
            idx: Index of sample (unused for random generation).

        Returns:
            Tuple of (z_i, z_j, target) where target is (2,) if return_sin_cos
            else (1,) in degrees.
        """
        z_i = torch.randn(self.latent_dim, dtype=torch.float32)
        z_j = torch.randn(self.latent_dim, dtype=torch.float32)
        angle_deg = random.uniform(-180.0, 180.0)
        if self.return_sin_cos:
            rad = angle_deg * math.pi / 180.0
            target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor([angle_deg], dtype=torch.float32)
        return z_i, z_j, target


class VideoLatentDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Encodes video frames into initial and horizon sequence Latent Vectors.

    Extracts (z_t, [z_{t+1}..z_{t+horizon}]) pairs using a VAE.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        vae_model: nn.Module,
        frameskip: int = 1,
        horizon: int = 10,
        device: str = "cpu",
        img_size: Tuple[int, int] = (192, 108),
    ) -> None:
        """Initialize the video latent dataset.

        Args:
            root_dir: Directory containing input video files.
            vae_model: VAE instance used for frame encoding.
            frameskip: Number of frames to skip between sequence steps.
            horizon: Prediction horizon steps into the future.
            device: Compute device string for VAE.
            img_size: Expected image resolution (width, height).
        """
        super().__init__()
        self.root_dir: Path = Path(root_dir)
        self.device: torch.device = torch.device(device)
        self.img_size: Tuple[int, int] = img_size
        self.horizon: int = horizon
        self.vae: nn.Module = vae_model.to(self.device)
        self.vae.eval()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if self.root_dir.exists():
            self._prepare_samples(frameskip)

    def _prepare_samples(self, frameskip: int) -> None:
        """Extract and encode frame sequences into latent horizon pairs.

        Args:
            frameskip: Frame skipping step size.
        """
        try:
            import cv2
        except ImportError:
            return

        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        video_paths = [
            p for p in self.root_dir.rglob("*") if p.suffix.lower() in video_exts
        ]

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            latents: List[torch.Tensor] = []
            ret, frame = cap.read()
            while ret:
                frame_resized = cv2.resize(
                    frame, self.img_size, interpolation=cv2.INTER_AREA
                )
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                x = frame_t.unsqueeze(0)

                with torch.no_grad():
                    if hasattr(self.vae, "get_latent"):
                        z = self.vae.get_latent(x.to(self.device)).cpu().squeeze(0)
                    elif hasattr(self.vae, "encode"):
                        enc = self.vae.encode(x.to(self.device))
                        z = (enc[0] if isinstance(enc, tuple) else enc).cpu().squeeze(0)
                    else:
                        z = self.vae(x.to(self.device)).cpu().squeeze(0)
                latents.append(z)

                for _ in range(frameskip):
                    ret, frame = cap.read()
                    if not ret:
                        break
            cap.release()

            for i in range(len(latents) - self.horizon):
                z_in = latents[i]
                z_future = torch.stack(latents[i + 1 : i + 1 + self.horizon])
                self.samples.append((z_in, z_future))

    def __len__(self) -> int:
        """Return the total number of horizon latent sequence pairs.

        Returns:
            Sample count.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch current Latent Vector and horizon future Latent Vectors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (z_in, z_future) with shapes (latent_dim,) and
            (horizon, latent_dim).
        """
        return self.samples[idx]


class DummyLatentHorizonDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Mock dataset generating synthetic initial and future latent horizons.

    Designed for testing trajectory prediction models over multi-step horizons.
    """

    def __init__(
        self,
        num_samples: int = 20,
        latent_dim: int = 128,
        horizon: int = 10,
    ) -> None:
        """Initialize the dummy latent horizon dataset.

        Args:
            num_samples: Number of synthetic sequences to provide.
            latent_dim: Dimension of each Latent Vector.
            horizon: Number of future time steps in the sequence.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.latent_dim: int = latent_dim
        self.horizon: int = horizon

    def __len__(self) -> int:
        """Return the sample count.

        Returns:
            Total synthetic sample count.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random initial Latent Vector and future horizon.

        Args:
            idx: Sample index (unused for random generation).

        Returns:
            Tuple of (z_in, z_future) where z_in is (latent_dim,) and
            z_future is (horizon, latent_dim).
        """
        z_in = torch.randn(self.latent_dim, dtype=torch.float32)
        z_future = torch.randn(self.horizon, self.latent_dim, dtype=torch.float32)
        return z_in, z_future


class AngleDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset providing Latent Vector pairs and corresponding rotation angles.

    Samples pairs from a pre-extracted array of Latent Vectors according to
    a defined frame-to-angle ratio.
    """

    def __init__(
        self,
        latent_array: Union[np.ndarray, torch.Tensor],
        total_frames: Optional[int] = None,
        samples_per_frame: int = 10,
        max_angle: float = 15.0,
        return_sin_cos: bool = False,
    ) -> None:
        """Initialize the angle dataset.

        Args:
            latent_array: Precomputed Latent Vector matrix of shape (N, D).
            total_frames: Optional total frame count limit.
            samples_per_frame: Number of random angle targets per frame.
            max_angle: Maximum angular offset in degrees to sample.
            return_sin_cos: Whether to return 2D [sin, cos] vector.
        """
        super().__init__()
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
        """Return the number of sampled pairs in the dataset.

        Returns:
            Total number of sampled evaluation pairs.
        """
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve start Latent Vector, angle Motion Command, and target Latent Vector.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (latent_i, target, latent_j) where target is (2,) if
            return_sin_cos else (1,) in degrees.
        """
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


class CachedLatentDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset loading pre-extracted Latent Vector records (.pt) for motion training.

    Supports 'rotation' and 'forward' motion modes with physical labels.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        mode: str = "rotation",
        max_frame_offset: int = 15,
        samples_per_frame: int = 5,
        step_distance_meters: float = 0.05,
        return_sin_cos: bool = False,
    ) -> None:
        """Initialize cached latent dataset.

        Args:
            cache_dir: Directory containing cached .pt files.
            mode: Motion mode, either 'rotation' or 'forward'.
            max_frame_offset: Maximum frame offset between paired latents.
            samples_per_frame: Number of target samples per base frame.
            step_distance_meters: Distance in meters per frame offset for forward mode.
            return_sin_cos: Whether to return 2D [sin, cos] vector for rotation mode.
        """
        super().__init__()
        self.mode: str = mode.lower()
        self.step_distance_meters: float = step_distance_meters
        self.return_sin_cos: bool = return_sin_cos
        self.records: List[Dict[str, Union[torch.Tensor, int, float, str]]] = []
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
                    for i in range(num_frames):
                        for _ in range(samples_per_frame):
                            offset = random.randint(-max_frame_offset, max_frame_offset)
                            j = max(0, min(num_frames - 1, i + offset))
                            self.pairs.append((r_idx, i, j))

    def __len__(self) -> int:
        """Return total sample pair count.

        Returns:
            Number of paired samples.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch start Latent Vector, target Latent Vector, and motion value tensor.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (z_i, z_j, motion_value) where motion_value is shape (2,)
            if rotation mode with return_sin_cos else (1,).
        """
        r_idx, i, j = self.pairs[idx]
        rec = self.records[r_idx]
        latents = rec["latents"]
        z_i = latents[i]
        z_j = latents[j]
        frame_diff = j - i

        if self.mode == "rotation":
            total_frames = float(rec.get("total_frames", len(latents)))
            angle = float(frame_diff) * (360.0 / max(total_frames, 1.0))
            if self.return_sin_cos:
                rad = angle * math.pi / 180.0
                sin_cos = [math.sin(rad), math.cos(rad)]
                return z_i, z_j, torch.tensor(sin_cos, dtype=torch.float32)
            return z_i, z_j, torch.tensor([angle], dtype=torch.float32)
        else:
            dist = float(frame_diff) * self.step_distance_meters
            return z_i, z_j, torch.tensor([dist], dtype=torch.float32)

"""
Module: sr_dataset
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Datasets for super-resolution and enhanced decoder training.
"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class VideoFrameDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads paired low-resolution and high-resolution video frame tensors."""

    def __init__(
        self,
        blur_root: str,
        sharp_root: str,
        num_videos_to_load: int = 100,
    ) -> None:
        """Initialize VideoFrameDataset.

        Args:
            blur_root: Path to low-resolution / blurry frames directory.
            sharp_root: Path to high-resolution / sharp frames directory.
            num_videos_to_load: Maximum number of video sequences to index.
        """
        self.blur_root = blur_root
        self.sharp_root = sharp_root
        self.num_videos_to_load = num_videos_to_load
        self.pairs: List[Tuple[str, str]] = []

        if os.path.exists(blur_root) and os.path.exists(sharp_root):
            self._index_frame_pairs()

    def _index_frame_pairs(self) -> None:
        """Indexes matching low-res and high-res image file paths."""
        blur_files = sorted(os.listdir(self.blur_root))
        sharp_files = sorted(os.listdir(self.sharp_root))
        common_files = set(blur_files).intersection(sharp_files)

        for filename in sorted(common_files):
            self.pairs.append(
                (
                    os.path.join(self.blur_root, filename),
                    os.path.join(self.sharp_root, filename),
                )
            )

    def __len__(self) -> int:
        """Returns total number of paired frames."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and returns a pair of (low_res, high_res) frame tensors.

        Args:
            idx: Index of the pair.

        Returns:
            Tuple of (lr_tensor, hr_tensor).
        """
        import cv2

        lr_path, hr_path = self.pairs[idx]
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)

        lr_tensor = (
            torch.from_numpy(lr_img.astype("float32") / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        hr_tensor = (
            torch.from_numpy(hr_img.astype("float32") / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        return lr_tensor, hr_tensor


class DummyPairedFrameDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic paired frame dataset for testing super-resolution pipelines."""

    def __init__(
        self,
        num_samples: int = 16,
        lr_shape: Tuple[int, int, int] = (3, 54, 96),
        hr_shape: Tuple[int, int, int] = (3, 108, 192),
    ) -> None:
        """Initialize DummyPairedFrameDataset.

        Args:
            num_samples: Number of synthetic pairs to produce.
            lr_shape: Shape tuple of low-resolution frame (C, H, W).
            hr_shape: Shape tuple of high-resolution frame (C, H, W).
        """
        self.num_samples = num_samples
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape

    def __len__(self) -> int:
        """Returns number of synthetic samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates random pair of (low_res, high_res) tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (lr_tensor, hr_tensor).
        """
        lr = torch.rand(*self.lr_shape, dtype=torch.float32)
        hr = torch.rand(*self.hr_shape, dtype=torch.float32)
        return lr, hr

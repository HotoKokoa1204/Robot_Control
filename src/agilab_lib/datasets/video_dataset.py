"""
Module: video_dataset
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Video dataset implementations for loading frames into tensors.
"""

import os
from typing import List

import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset[torch.Tensor]):
    """Loads all frames from a video file into normalized float32 tensors."""

    def __init__(
        self, video_path: str, img_height: int = 108, img_width: int = 192
    ) -> None:
        """Initialize VideoDataset.

        Args:
            video_path: Path to the target video file.
            img_height: Desired frame height.
            img_width: Desired frame width.
        """
        self.video_path = video_path
        self.img_height = img_height
        self.img_width = img_width
        self.frames: List[torch.Tensor] = []
        if os.path.exists(video_path):
            self._load_frames()

    def _load_frames(self) -> None:
        """Reads, resizes, and normalizes frames from video file into memory."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (self.img_width, self.img_height))
            frame_normalized = frame_resized.astype("float32") / 255.0
            # Convert [H, W, C] to [C, H, W]
            frame_tensor = (
                torch.from_numpy(frame_normalized).permute(2, 0, 1).contiguous()
            )
            self.frames.append(frame_tensor)
        cap.release()

    def __len__(self) -> int:
        """Returns total number of loaded frames."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns frame tensor at specified index.

        Args:
            idx: Index of the frame.

        Returns:
            Normalized frame tensor of shape (3, H, W).
        """
        return self.frames[idx]


class DummyVideoDataset(Dataset[torch.Tensor]):
    """Mock video dataset generating synthetic frame tensors for testing."""

    def __init__(
        self, num_frames: int = 20, img_height: int = 108, img_width: int = 192
    ) -> None:
        """Initialize DummyVideoDataset.

        Args:
            num_frames: Number of synthetic frames to generate.
            img_height: Height of synthetic frames.
            img_width: Width of synthetic frames.
        """
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self) -> int:
        """Returns total number of mock frames."""
        return self.num_frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generates random frame tensor.

        Args:
            idx: Frame index.

        Returns:
            Synthetic frame tensor of shape (3, H, W).
        """
        return torch.rand(3, self.img_height, self.img_width, dtype=torch.float32)

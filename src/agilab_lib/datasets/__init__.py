"""
Module: datasets
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Dataset loaders for video and latent trajectory representations.
"""

from agilab_lib.datasets.latent_dataset import (
    AngleDataset,
    DummyLatentHorizonDataset,
    DummyLatentPairDataset,
    InMemoryLatentOffsetDataset,
    VideoLatentDataset,
)
from agilab_lib.datasets.sr_dataset import (
    DummyPairedFrameDataset,
    VideoFrameDataset,
)
from agilab_lib.datasets.video_dataset import (
    DummyVideoDataset,
    VideoDataset,
)

__all__ = [
    "VideoDataset",
    "DummyVideoDataset",
    "VideoFrameDataset",
    "DummyPairedFrameDataset",
    "InMemoryLatentOffsetDataset",
    "DummyLatentPairDataset",
    "VideoLatentDataset",
    "DummyLatentHorizonDataset",
    "AngleDataset",
]

"""
Module: datasets
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Dataset loaders for video and latent trajectory representations.
"""

from agilab_lib.datasets.dual_source_dataset import (
    DualSourceBatch,
    DualSourceVideoDataset,
    DummyDualSourceVideoDataset,
    dual_source_collate_fn,
)
from agilab_lib.datasets.motion_indexing import (
    MotionPairSampler,
    RotationPairIndexer,
    StraightPairIndexer,
)

__all__ = [
    "DualSourceBatch",
    "DualSourceVideoDataset",
    "DummyDualSourceVideoDataset",
    "dual_source_collate_fn",
    "StraightPairIndexer",
    "RotationPairIndexer",
    "MotionPairSampler",
]

"""Module: interpolation
Stage: Library
Author: KafuuChino
Date: 2026-09-01
Description: Latent vector and keyframe interpolation utilities.
"""

from agilab_lib.utils.trajectory import (
    interpolate_all_frames_numpy,
    linear_interpolate_latent_sequence,
)

__all__ = [
    "interpolate_all_frames_numpy",
    "linear_interpolate_latent_sequence",
]

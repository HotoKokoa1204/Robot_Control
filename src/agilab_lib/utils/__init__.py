"""
Module: utils
Stage: Library
Author: AGILAB NTNU
Date: 2026-09-01
Description: Utility package with interpolation, evaluation, and PCA tools.
"""

from agilab_lib.utils.eval_metrics import (
    decode_latents_to_frames,
    evaluate_angle_prediction_mae,
    evaluate_latent_prediction_mse,
)
from agilab_lib.utils.interpolation import (
    interpolate_all_frames_numpy,
    linear_interpolate_latent_sequence,
)
from agilab_lib.utils.pca import BatchedPCA

__all__ = [
    "linear_interpolate_latent_sequence",
    "interpolate_all_frames_numpy",
    "evaluate_latent_prediction_mse",
    "evaluate_angle_prediction_mae",
    "decode_latents_to_frames",
    "BatchedPCA",
]

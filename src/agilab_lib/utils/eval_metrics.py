"""Module: eval_metrics
Stage: Library
Author: KafuuChino
Date: 2026-09-01
Description: Backward-compatibility shims re-exporting evaluation metrics
    from trajectory.
"""

from agilab_lib.utils.trajectory import (
    decode_latents_to_frames,
    evaluate_angle_prediction_mae,
    evaluate_latent_prediction_mse,
)

__all__ = [
    "decode_latents_to_frames",
    "evaluate_angle_prediction_mae",
    "evaluate_latent_prediction_mse",
]

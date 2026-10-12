"""Module: utils
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Utility package with trajectory operations, evaluation, and storage tools.
"""

from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    get_cache_dir,
    get_checkpoints_dir,
    get_data_dir,
    get_eval_dir,
    get_keyframes_dir,
    get_latents_dir,
    get_logs_dir,
    get_outputs_dir,
    get_plots_dir,
    get_project_root,
    get_samples_dir,
    resolve_project_path,
)
from agilab_lib.utils.trajectory import (
    BatchedPCA,
    decode_latents_to_frames,
    evaluate_angle_prediction_mae,
    evaluate_latent_prediction_mse,
    extract_keyframe_indices,
    interpolate_all_frames_numpy,
    linear_interpolate_latent_sequence,
)

__all__ = [
    "linear_interpolate_latent_sequence",
    "interpolate_all_frames_numpy",
    "evaluate_latent_prediction_mse",
    "evaluate_angle_prediction_mae",
    "decode_latents_to_frames",
    "extract_keyframe_indices",
    "BatchedPCA",
    "get_project_root",
    "resolve_project_path",
    "get_outputs_dir",
    "get_eval_dir",
    "get_samples_dir",
    "get_keyframes_dir",
    "get_latents_dir",
    "get_plots_dir",
    "get_logs_dir",
    "get_checkpoints_dir",
    "get_data_dir",
    "get_cache_dir",
    "ensure_writable_output_path",
]

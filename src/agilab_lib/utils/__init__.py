"""Module: utils
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Utility package with interpolation, evaluation, PCA, and storage tools.
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
from agilab_lib.utils.keyframes import extract_keyframe_indices
from agilab_lib.utils.pca import BatchedPCA
from agilab_lib.utils.storage import (
    get_cache_dir,
    get_checkpoints_dir,
    get_data_dir,
    get_outputs_dir,
    get_project_root,
    resolve_project_path,
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
    "get_checkpoints_dir",
    "get_data_dir",
    "get_cache_dir",
]

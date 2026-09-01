"""
Module: interpolation
Stage: Library
Author: KafuuChino
Date: 2026-09-01
Description: Latent vector and keyframe interpolation utilities.
"""

import numpy as np
import torch


def linear_interpolate_latent_sequence(
    latents: torch.Tensor, interp_frames: int = 0
) -> torch.Tensor:
    """Performs linear interpolation between consecutive latent vectors.

    Args:
        latents: Tensor of shape (N, latent_dim).
        interp_frames: Number of frames to insert between each adjacent pair.

    Returns:
        Interpolated tensor of shape ((N - 1) * (interp_frames + 1) + 1, latent_dim).
    """
    if interp_frames <= 0 or len(latents) < 2:
        return latents

    interpolated_list = []
    num_steps = len(latents)

    for i in range(num_steps - 1):
        start_z = latents[i]
        end_z = latents[i + 1]
        interpolated_list.append(start_z)

        for t in range(1, interp_frames + 1):
            alpha = t / (interp_frames + 1.0)
            interp_z = (1.0 - alpha) * start_z + alpha * end_z
            interpolated_list.append(interp_z)

    interpolated_list.append(latents[-1])
    return torch.stack(interpolated_list, dim=0)


def interpolate_all_frames_numpy(
    latents: np.ndarray, interpolation_steps: int = 10
) -> np.ndarray:
    """Numpy version of keyframe linear interpolation.

    Args:
        latents: Numpy array of shape (N, latent_dim).
        interpolation_steps: Number of frames to insert between each adjacent pair.

    Returns:
        Interpolated array of shape (N + (N - 1) * interpolation_steps, latent_dim).
    """
    num_original_frames = len(latents)
    if num_original_frames < 2 or interpolation_steps <= 0:
        return latents

    num_interpolated_frames_total = (num_original_frames - 1) * interpolation_steps
    total_output_frames = num_original_frames + num_interpolated_frames_total
    latent_dim = latents.shape[1]

    out_latents = np.zeros((total_output_frames, latent_dim), dtype=latents.dtype)
    current_out_idx = 0

    for i in range(num_original_frames - 1):
        la = latents[i]
        lb = latents[i + 1]
        steps_between = interpolation_steps + 1
        delta = (lb - la) / float(steps_between)

        for s in range(steps_between):
            out_latents[current_out_idx] = la + delta * s
            current_out_idx += 1

    out_latents[current_out_idx] = latents[-1]
    return out_latents

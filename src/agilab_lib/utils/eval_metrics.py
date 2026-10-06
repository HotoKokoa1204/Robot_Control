"""Module: eval_metrics
Stage: Library
Author: KafuuChino
Date: 2026-09-01
Description: Evaluation metrics for latent predictions, angles, and decoding.
"""

from typing import Any

import torch
import torch.nn.functional as F


def evaluate_latent_prediction_mse(
    predicted_latents: torch.Tensor, ground_truth_latents: torch.Tensor
) -> float:
    """Computes Mean Squared Error (MSE) between predicted and ground truth latents.

    Args:
        predicted_latents: Predicted latent tensor.
        ground_truth_latents: Ground truth latent tensor.

    Returns:
        Scalar MSE loss as a float.
    """
    return float(F.mse_loss(predicted_latents, ground_truth_latents).item())


def evaluate_angle_prediction_mae(
    predicted_angles: torch.Tensor, ground_truth_angles: torch.Tensor
) -> float:
    """Computes Mean Absolute Error (MAE) in degrees between angles or sin/cos vectors.

    Supports both 1D scalar angles in degrees (B, 1) and 2D normalized rotation
    vectors (B, 2) [sin(theta), cos(theta)]. For 2D vectors, computes the true
    geodesic angular difference on the circle.

    Args:
        predicted_angles: Predicted angle tensor of shape (B, 1) or (B, 2).
        ground_truth_angles: Ground truth angle tensor of shape (B, 1) or (B, 2).

    Returns:
        Scalar MAE in degrees as a float.
    """
    if predicted_angles.shape[-1] == 2 and ground_truth_angles.shape[-1] == 2:
        sin_p, cos_p = predicted_angles[..., 0], predicted_angles[..., 1]
        sin_g, cos_g = ground_truth_angles[..., 0], ground_truth_angles[..., 1]
        sin_diff = sin_p * cos_g - cos_p * sin_g
        cos_diff = cos_p * cos_g + sin_p * sin_g
        angle_diff_rad = torch.atan2(sin_diff, cos_diff)
        angle_diff_deg = torch.abs(angle_diff_rad * (180.0 / torch.pi))
        return float(torch.mean(angle_diff_deg).item())

    return float(F.l1_loss(predicted_angles, ground_truth_angles).item())


def decode_latents_to_frames(decoder_model: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decodes a batch or sequence of latents back to image tensors (B, 3, H, W).

    Args:
        decoder_model: VAE or decoder model instance with a .decode() method.
        latents: Latent vector tensor (B, latent_dim).

    Returns:
        Reconstructed frames tensor (B, 3, H, W).
    """
    decoder_model.eval()
    with torch.no_grad():
        frames = decoder_model.decode(latents)
    return frames

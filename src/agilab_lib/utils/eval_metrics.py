"""
Module: eval_metrics
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
    """Computes Mean Absolute Error (MAE) between predicted and ground truth angles.

    Args:
        predicted_angles: Predicted angle tensor.
        ground_truth_angles: Ground truth angle tensor.

    Returns:
        Scalar MAE loss as a float.
    """
    return float(F.l1_loss(predicted_angles, ground_truth_angles).item())


def decode_latents_to_frames(autoencoder: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decodes a batch or sequence of latents back to image tensors (B, 3, H, W).

    Args:
        autoencoder: Autoencoder model instance with a .decode() method.
        latents: Latent vector tensor (B, latent_dim).

    Returns:
        Reconstructed frames tensor (B, 3, H, W).
    """
    autoencoder.eval()
    with torch.no_grad():
        frames = autoencoder.decode(latents)
    return frames

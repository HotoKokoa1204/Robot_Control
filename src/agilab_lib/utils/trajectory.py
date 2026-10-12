"""Module: trajectory
Stage: Library
Author: KafuuChino
Date: 2026-10-12
Description: Unified trajectory, keyframe extraction, and PCA utilities.
"""

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agilab_lib.models.vae import VAE


class BatchedPCA:
    """Batched Incremental PCA using PyTorch for latent space reduction."""

    def __init__(self, n_components: int = 2) -> None:
        """Initializes BatchedPCA with number of target components.

        Args:
            n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "BatchedPCA":
        """Fits PCA model on latent matrix x: (N, D).

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            self
        """
        self.mean = torch.mean(x, dim=0, keepdim=True)
        x_centered = x - self.mean
        # SVD on centered data
        _u, _s, v = torch.pca_lowrank(x_centered, q=self.n_components, center=False)
        self.components_ = v[:, : self.n_components]  # (D, n_components)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Projects latent matrix x: (N, D) -> (N, n_components).

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Projected tensor of shape (N, n_components).

        Raises:
            RuntimeError: If called before fit.
        """
        if self.mean is None or self.components_ is None:
            raise RuntimeError("BatchedPCA must be fitted before calling transform.")
        x_centered = x - self.mean
        return torch.mm(x_centered, self.components_)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fits PCA model and transforms input tensor in one step.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Projected tensor of shape (N, n_components).
        """
        self.fit(x)
        return self.transform(x)


def extract_keyframe_indices(
    vae: VAE,
    frames: List[torch.Tensor],
    tau: float,
    device: torch.device,
) -> List[int]:
    """Extracts Keyframe indices where Latent Vector Euclidean distance exceeds Tau.

    Args:
        vae: Trained or initialized VAE model.
        frames: List of frame tensors of shape (3, H, W).
        tau: Euclidean distance threshold in Latent Vector space (τ).
        device: PyTorch device for tensor operations.

    Returns:
        List of selected Keyframe indices.
    """
    if len(frames) == 0:
        return []

    vae.eval()
    vae.to(device)

    keyframe_indices: List[int] = [0]
    with torch.no_grad():
        x0 = frames[0].unsqueeze(0).to(device)
        last_keyframe_latent: torch.Tensor = vae.get_latent(x0)

        for i in range(1, len(frames)):
            xi = frames[i].unsqueeze(0).to(device)
            current_latent: torch.Tensor = vae.get_latent(xi)
            distance = float(
                torch.norm(current_latent - last_keyframe_latent, p=2).item()
            )
            if distance >= tau:
                keyframe_indices.append(i)
                last_keyframe_latent = current_latent

    return keyframe_indices


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
        frames: torch.Tensor = decoder_model.decode(latents)
    return frames


__all__ = [
    "BatchedPCA",
    "decode_latents_to_frames",
    "evaluate_angle_prediction_mae",
    "evaluate_latent_prediction_mse",
    "extract_keyframe_indices",
    "interpolate_all_frames_numpy",
    "linear_interpolate_latent_sequence",
]

"""
Module: keyframes
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Keyframe extraction logic based on Latent Vector Euclidean distance.
"""

from typing import List

import torch

from agilab_lib.models.vae import VAE


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

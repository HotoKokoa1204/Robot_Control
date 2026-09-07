"""
Module: angle_predictor
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Neural network model for relative angle and Motion Command prediction.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class AnglePredictor(nn.Module):
    """Predicts relative rotation angle between pairs of Latent Vectors.

    Takes a current frame Latent Vector and a target Keyframe Latent Vector,
    concatenates them, and passes through feedforward projection layers to
    predict the relative angle Motion Command in degrees.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        use_batch_norm: bool = True,
    ) -> None:
        """Initialize the Angle Predictor network.

        Args:
            latent_dim: Dimension of each input Latent Vector.
            hidden_dims: List of internal hidden dimension sizes.
            use_batch_norm: Whether to insert 1D batch normalization layers.
        """
        super().__init__()
        self.latent_dim: int = latent_dim
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.hidden_dims: List[int] = list(hidden_dims)
        self.use_batch_norm: bool = use_batch_norm

        layers: List[nn.Module] = []
        input_dim = 2 * latent_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
    ) -> torch.Tensor:
        """Predict relative rotation angle from a pair of Latent Vectors.

        Args:
            latent1: Current frame Latent Vector of shape (B, latent_dim).
            latent2: Target Keyframe Latent Vector of shape (B, latent_dim).

        Returns:
            Predicted relative angle tensor of shape (B, 1) in degrees.
        """
        if latent1.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent1 dimension mismatch: expected {self.latent_dim}, "
                f"got {latent1.shape[1]}"
            )
        if latent2.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent2 dimension mismatch: expected {self.latent_dim}, "
                f"got {latent2.shape[1]}"
            )
        if latent1.shape[0] != latent2.shape[0]:
            raise ValueError(
                f"Batch sizes do not match: {latent1.shape[0]} vs {latent2.shape[0]}"
            )

        combined_latent = torch.cat((latent1, latent2), dim=1)
        predicted_angle: torch.Tensor = self.network(combined_latent)
        return predicted_angle

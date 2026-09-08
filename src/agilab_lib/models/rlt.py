"""
Module: rlt
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: 3D Motion Command conditioned Residual Latent Transformer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedResidualBlock(nn.Module):
    """Residual block conditioned on a 3D Motion Command vector.

    Takes a Latent Vector representation and concatenates a 3-element condition
    vector [sin(theta), cos(theta), distance_meters] before feedforward
    transformations with skip connection.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize the conditioned residual block.

        Args:
            dim: Dimension of the input and output Latent Vector.
            hidden_dim: Internal hidden dimension of the block.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(dim + 3, hidden_dim)
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc3: nn.Linear = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass applying conditioned residual transformation.

        Args:
            x: Input Latent Vector tensor of shape (B, dim).
            cond: 3D condition tensor of shape (B, 3).

        Returns:
            Output Latent Vector tensor of shape (B, dim) with residual added.
        """
        y = torch.cat([x, cond], dim=1)
        y = F.relu(self.ln1(self.fc1(y)))
        y = F.relu(self.ln2(self.fc2(y)))
        y = self.fc3(y)
        return x + y


class ResidualLatentTransformer(nn.Module):
    """Residual Latent Transformer for Motion Command conditioned transitions.

    Predicts future Latent Vectors given an initial Latent Vector and a
    3D Motion Command composed of rotation angle (degrees) and linear distance
    (meters).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 5,
        block_inner_dim: int = 128,
    ) -> None:
        """Initialize the Residual Latent Transformer.

        Args:
            latent_dim: Dimension of input and output Latent Vectors.
            hidden_dim: Hidden dimension for feature projection.
            num_blocks: Number of stacked ConditionedResidualBlock layers.
            block_inner_dim: Hidden dimension inside each residual block.
        """
        super().__init__()
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_blocks: int = num_blocks
        self.block_inner_dim: int = block_inner_dim

        self.fc_in: nn.Linear = nn.Linear(latent_dim, hidden_dim)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ConditionedResidualBlock(hidden_dim, block_inner_dim)
                for _ in range(num_blocks)
            ]
        )
        self.fc_out: nn.Linear = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        latent: torch.Tensor,
        angle_deg: Optional[torch.Tensor] = None,
        distance_meters: Optional[torch.Tensor] = None,
        sin_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting the transformed Latent Vector.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            angle_deg: Optional rotation angle in degrees of shape (B, 1) or (B,).
            distance_meters: Optional linear forward distance in meters of shape
                (B, 1) or (B,).
            sin_cos: Optional direct [sin, cos] unit vector tensor of shape (B, 2).

        Returns:
            Predicted future Latent Vector of shape (B, latent_dim).
        """
        if distance_meters is None:
            distance_meters = torch.zeros(latent.shape[0], 1, device=latent.device)
        elif distance_meters.dim() == 1:
            distance_meters = distance_meters.unsqueeze(1)

        if sin_cos is None:
            if angle_deg is None:
                angle_deg = torch.zeros(latent.shape[0], 1, device=latent.device)
            elif angle_deg.dim() == 1:
                angle_deg = angle_deg.unsqueeze(1)
            rad = angle_deg * math.pi / 180.0
            cond_sin = torch.sin(rad)
            cond_cos = torch.cos(rad)
            sin_cos = torch.cat([cond_sin, cond_cos], dim=-1)

        cond = torch.cat([sin_cos, distance_meters], dim=-1).to(latent.device)

        x = F.relu(self.fc_in(latent))
        for blk in self.blocks:
            x = blk(x, cond)
        return self.fc_out(x)

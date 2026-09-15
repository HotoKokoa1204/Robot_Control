"""
Module: rlt
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: 3D Motion Command conditioned Residual Latent Transformer.
"""

import math
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedResidualBlock(nn.Module):
    """Residual block conditioned on a motion condition vector.

    Takes a Latent Vector representation and concatenates a condition
    vector before feedforward transformations with skip connection.
    """

    def __init__(self, dim: int, hidden_dim: int, cond_dim: int = 3) -> None:
        """Initialize the conditioned residual block.

        Args:
            dim: Dimension of the input and output Latent Vector.
            hidden_dim: Internal hidden dimension of the block.
            cond_dim: Dimension of the conditioning vector (e.g. 2 for rotation,
                1 for forward distance, 3 for 3D Motion Command).
        """
        super().__init__()
        self.cond_dim: int = cond_dim
        self.fc1: nn.Linear = nn.Linear(dim + cond_dim, hidden_dim)
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc3: nn.Linear = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass applying conditioned residual transformation.

        Args:
            x: Input Latent Vector tensor of shape (B, dim).
            cond: Condition tensor of shape (B, cond_dim).

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
                ConditionedResidualBlock(hidden_dim, block_inner_dim, cond_dim=3)
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


class BaseLatentTransformer(nn.Module):
    """Base class for conditioned Residual Latent Transformers.

    Encapsulates linear input projection, stacked ConditionedResidualBlock layers,
    and output projection.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        block_inner_dim: int = 128,
        cond_dim: int = 1,
    ) -> None:
        """Initialize the Base Latent Transformer.

        Args:
            latent_dim: Dimension of input and output Latent Vectors.
            hidden_dim: Hidden dimension for feature projection.
            num_blocks: Number of stacked ConditionedResidualBlock layers.
            block_inner_dim: Hidden dimension inside each residual block.
            cond_dim: Dimension of the conditioning vector.
        """
        super().__init__()
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_blocks: int = num_blocks
        self.block_inner_dim: int = block_inner_dim
        self.cond_dim: int = cond_dim

        self.fc_in: nn.Linear = nn.Linear(latent_dim, hidden_dim)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ConditionedResidualBlock(hidden_dim, block_inner_dim, cond_dim=cond_dim)
                for _ in range(num_blocks)
            ]
        )
        self.fc_out: nn.Linear = nn.Linear(hidden_dim, latent_dim)

    def _forward_blocks(self, latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass through input projection, blocks, and output projection.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            cond: Conditioning tensor of shape (B, cond_dim).

        Returns:
            Transformed Latent Vector of shape (B, latent_dim).
        """
        x = F.relu(self.fc_in(latent))
        for blk in self.blocks:
            x = blk(x, cond)
        return self.fc_out(x)


class RotationLatentTransformer(BaseLatentTransformer):
    """Residual Latent Transformer conditioned exclusively on rotation.

    Predicts future Latent Vectors resulting from in-place angular rotation,
    conditioned on a 2D unit vector [sin(theta), cos(theta)] or rotation angle
    in degrees. Utilizes 5 ConditionedResidualBlock layers.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 5,
        block_inner_dim: int = 128,
    ) -> None:
        """Initialize the Rotation Latent Transformer.

        Args:
            latent_dim: Dimension of input and output Latent Vectors.
            hidden_dim: Hidden dimension for feature projection.
            num_blocks: Number of stacked ConditionedResidualBlock layers (default 5).
            block_inner_dim: Hidden dimension inside each residual block.
        """
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_inner_dim=block_inner_dim,
            cond_dim=2,
        )

    def forward(
        self,
        latent: torch.Tensor,
        angle_deg: Optional[Union[torch.Tensor, float]] = None,
        sin_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting future Latent Vector under rotation.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            angle_deg: Optional rotation angle in degrees of shape (B, 1), (B,),
                or scalar float.
            sin_cos: Optional direct 2D [sin, cos] unit vector tensor of shape (B, 2).

        Returns:
            Predicted future Latent Vector of shape (B, latent_dim).
        """
        if sin_cos is None:
            if angle_deg is None:
                angle_deg = torch.zeros(latent.shape[0], 1, device=latent.device)
            elif isinstance(angle_deg, (int, float)):
                angle_deg = torch.full(
                    (latent.shape[0], 1), float(angle_deg), device=latent.device
                )
            elif angle_deg.dim() == 1:
                angle_deg = angle_deg.unsqueeze(1)
            rad = angle_deg * (math.pi / 180.0)
            cond_sin = torch.sin(rad)
            cond_cos = torch.cos(rad)
            sin_cos = torch.cat([cond_sin, cond_cos], dim=-1)

        cond = sin_cos.to(latent.device)
        return self._forward_blocks(latent, cond)


class ForwardLatentTransformer(BaseLatentTransformer):
    """Lightweight Residual Latent Transformer conditioned on forward distance.

    Predicts future Latent Vectors resulting from linear translation along
    a path, conditioned on scalar forward distance in meters. Utilizes
    2 ConditionedResidualBlock layers.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        block_inner_dim: int = 128,
    ) -> None:
        """Initialize the Forward Latent Transformer.

        Args:
            latent_dim: Dimension of input and output Latent Vectors.
            hidden_dim: Hidden dimension for feature projection.
            num_blocks: Number of stacked ConditionedResidualBlock layers (default 2).
            block_inner_dim: Hidden dimension inside each residual block.
        """
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_inner_dim=block_inner_dim,
            cond_dim=1,
        )

    def forward(
        self,
        latent: torch.Tensor,
        distance_meters: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        """Forward pass predicting future Latent Vector under forward translation.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            distance_meters: Optional linear forward distance in meters of shape
                (B, 1), (B,), or scalar float.

        Returns:
            Predicted future Latent Vector of shape (B, latent_dim).
        """
        if distance_meters is None:
            distance_meters = torch.zeros(latent.shape[0], 1, device=latent.device)
        elif isinstance(distance_meters, (int, float)):
            distance_meters = torch.full(
                (latent.shape[0], 1), float(distance_meters), device=latent.device
            )
        elif distance_meters.dim() == 1:
            distance_meters = distance_meters.unsqueeze(1)

        cond = distance_meters.to(latent.device)
        return self._forward_blocks(latent, cond)


class ExecutionOrder(str, Enum):
    """Execution sequence order for chained latent transformation."""

    ROTATE_FIRST = "rotate_first"
    FORWARD_FIRST = "forward_first"


class ChainedLatentTransformer(nn.Module):
    """Composite transformer executing rotation and forward models sequentially.

    Applies RotationLatentTransformer and ForwardLatentTransformer in the
    sequence specified by execution_order ("rotate_first" or "forward_first").
    Preserves single-action invariance when motion components are zero.
    """

    def __init__(
        self,
        rotation_model: Optional[RotationLatentTransformer] = None,
        forward_model: Optional[ForwardLatentTransformer] = None,
        rotation_checkpoint: Optional[Union[str, Path]] = None,
        forward_checkpoint: Optional[Union[str, Path]] = None,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        rotation_num_blocks: int = 5,
        forward_num_blocks: int = 2,
        block_inner_dim: int = 128,
    ) -> None:
        """Initialize the Chained Latent Transformer.

        Args:
            rotation_model: Optional pre-constructed RotationLatentTransformer.
            forward_model: Optional pre-constructed ForwardLatentTransformer.
            rotation_checkpoint: Optional file path to rotation model weights.
            forward_checkpoint: Optional file path to forward model weights.
            latent_dim: Dimension of input and output Latent Vectors.
            hidden_dim: Hidden dimension for feature projection.
            rotation_num_blocks: Number of blocks for rotation model (default 5).
            forward_num_blocks: Number of blocks for forward model (default 2).
            block_inner_dim: Hidden dimension inside each residual block.
        """
        super().__init__()
        if rotation_model is not None:
            self.rotation_model: RotationLatentTransformer = rotation_model
        else:
            self.rotation_model = RotationLatentTransformer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_blocks=rotation_num_blocks,
                block_inner_dim=block_inner_dim,
            )
            self._load_checkpoint(self.rotation_model, rotation_checkpoint)

        if forward_model is not None:
            self.forward_model: ForwardLatentTransformer = forward_model
        else:
            self.forward_model = ForwardLatentTransformer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_blocks=forward_num_blocks,
                block_inner_dim=block_inner_dim,
            )
            self._load_checkpoint(self.forward_model, forward_checkpoint)

    @staticmethod
    def _load_checkpoint(
        model: nn.Module, checkpoint: Optional[Union[str, Path]]
    ) -> None:
        """Load pretrained state dictionary into a model if path exists.

        Args:
            model: PyTorch module into which weights will be loaded.
            checkpoint: Optional file path to the checkpoint tensor.
        """
        if checkpoint is not None and Path(checkpoint).exists():
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

    def forward(
        self,
        latent: torch.Tensor,
        angle_deg: Optional[Union[torch.Tensor, float]] = None,
        distance_meters: Optional[Union[torch.Tensor, float]] = None,
        sin_cos: Optional[torch.Tensor] = None,
        execution_order: Union[str, ExecutionOrder] = ExecutionOrder.ROTATE_FIRST,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass executing rotation and forward models in sequence.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            angle_deg: Optional rotation angle in degrees.
            distance_meters: Optional forward distance in meters.
            sin_cos: Optional 2D [sin, cos] unit vector tensor of shape (B, 2).
            execution_order: Execution sequence, either "rotate_first"
                (ExecutionOrder.ROTATE_FIRST) or "forward_first"
                (ExecutionOrder.FORWARD_FIRST).
            return_intermediate: If True, returns tuple of
                (final_latent, intermediate_latent).

        Returns:
            Predicted future Latent Vector of shape (B, latent_dim), or tuple
            (final_latent, intermediate_latent) if return_intermediate is True.

        Raises:
            ValueError: If execution_order is not 'rotate_first' or 'forward_first'.
        """
        order_str = (
            execution_order.value
            if isinstance(execution_order, ExecutionOrder)
            else str(execution_order)
        )
        if order_str not in (
            ExecutionOrder.ROTATE_FIRST.value,
            ExecutionOrder.FORWARD_FIRST.value,
        ):
            raise ValueError(
                f"Unsupported execution_order: '{execution_order}'. "
                "Must be 'rotate_first' or 'forward_first'."
            )

        is_zero_dist = (
            distance_meters is None
            or (isinstance(distance_meters, (int, float)) and distance_meters == 0)
            or (
                isinstance(distance_meters, torch.Tensor)
                and bool(torch.all(distance_meters == 0))
            )
        )

        is_zero_rot = sin_cos is None and (
            angle_deg is None
            or (isinstance(angle_deg, (int, float)) and angle_deg == 0)
            or (isinstance(angle_deg, torch.Tensor) and bool(torch.all(angle_deg == 0)))
        )

        if is_zero_dist and is_zero_rot:
            return (latent, latent) if return_intermediate else latent

        if is_zero_dist:
            z_rot = self.rotation_model(latent, angle_deg=angle_deg, sin_cos=sin_cos)
            return (z_rot, z_rot) if return_intermediate else z_rot

        if is_zero_rot:
            z_fwd = self.forward_model(latent, distance_meters=distance_meters)
            return (z_fwd, z_fwd) if return_intermediate else z_fwd

        if execution_order == "rotate_first":
            z_mid = self.rotation_model(latent, angle_deg=angle_deg, sin_cos=sin_cos)
            z_out = self.forward_model(z_mid, distance_meters=distance_meters)
        else:
            z_mid = self.forward_model(latent, distance_meters=distance_meters)
            z_out = self.rotation_model(z_mid, angle_deg=angle_deg, sin_cos=sin_cos)

        if return_intermediate:
            return z_out, z_mid
        return z_out

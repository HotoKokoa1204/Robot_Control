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
from typing import Any, Dict, List, Optional, Tuple, Union

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
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def reset_parameters(self) -> None:
        """Reset residual block parameters, zeroing fc3 for identity initialization."""
        self.fc1.reset_parameters()
        self.ln1.reset_parameters()
        self.fc2.reset_parameters()
        self.ln2.reset_parameters()
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

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


class BaseLatentTransformer(nn.Module):
    """Base class for conditioned Residual Latent Transformers.

    Routes latent vectors directly through stacked ConditionedResidualBlock
    layers without linear projection bottlenecks or ReLU clipping.
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
            hidden_dim: Legacy hidden dimension parameter (preserved for compatibility).
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

        self.fc_in: nn.Module = nn.Identity()
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ConditionedResidualBlock(latent_dim, block_inner_dim, cond_dim=cond_dim)
                for _ in range(num_blocks)
            ]
        )
        self.fc_out: nn.Module = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset transformer residual blocks to identity zero-initialization."""
        for blk in self.blocks:
            blk.reset_parameters()

    def _forward_blocks(self, latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass directly through stacked conditioned residual blocks.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            cond: Conditioning tensor of shape (B, cond_dim).

        Returns:
            Transformed Latent Vector of shape (B, latent_dim).
        """
        x = latent
        for blk in self.blocks:
            x = blk(x, cond)
        return x


class ResidualLatentTransformer(BaseLatentTransformer):
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
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_inner_dim=block_inner_dim,
            cond_dim=3,
        )

    def forward(
        self,
        latent: torch.Tensor,
        angle_deg: Optional[Union[torch.Tensor, float]] = None,
        distance_meters: Optional[Union[torch.Tensor, float]] = None,
        sin_cos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting the transformed Latent Vector.

        Args:
            latent: Current Latent Vector of shape (B, latent_dim).
            angle_deg: Optional rotation angle in degrees of shape (B, 1), (B,),
                or scalar float.
            distance_meters: Optional linear forward distance in meters of shape
                (B, 1), (B,), or scalar float.
            sin_cos: Optional direct [sin, cos] unit vector tensor of shape (B, 2).

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

        cond = torch.cat(
            [sin_cos.to(latent.device), distance_meters.to(latent.device)], dim=-1
        )
        return self._forward_blocks(latent, cond)


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
        rotation_checkpoint: Optional[Union[str, Path, Dict[str, Any]]] = None,
        forward_checkpoint: Optional[Union[str, Path, Dict[str, Any]]] = None,
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
            rotation_checkpoint: Optional file path or state dict for rotation weights.
            forward_checkpoint: Optional file path or state dict for forward weights.
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

    def reset_parameters(self) -> None:
        """Reset parameters of rotation and forward models to identity zero-init."""
        self.rotation_model.reset_parameters()
        self.forward_model.reset_parameters()

    @staticmethod
    def _load_checkpoint(
        model: nn.Module,
        checkpoint: Optional[Union[str, Path, Dict[str, Any]]],
    ) -> None:
        """Load pretrained state dictionary into a model if path or dict exists.

        Handles raw state dicts, nested dictionaries (e.g., 'state_dict',
        'model_state_dict', 'forward_state_dict', 'rotation_state_dict', 'model'),
        and key prefix stripping ('module.', 'forward_transformer.',
        'rotation_transformer.').

        Args:
            model: PyTorch module into which weights will be loaded.
            checkpoint: Optional file path or state dictionary tensor mapping.
        """
        if checkpoint is None:
            return

        if isinstance(checkpoint, (str, Path)):
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                return
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        elif isinstance(checkpoint, dict):
            ckpt = checkpoint
        else:
            return

        if isinstance(ckpt, dict):
            for candidate_key in (
                "state_dict",
                "model_state_dict",
                "forward_state_dict",
                "rotation_state_dict",
                "model",
            ):
                if candidate_key in ckpt and isinstance(ckpt[candidate_key], dict):
                    ckpt = ckpt[candidate_key]
                    break

            model_keys = set(model.state_dict().keys())
            if set(ckpt.keys()) == model_keys:
                state_dict = ckpt
            else:
                target_prefix = ""
                if isinstance(model, ForwardLatentTransformer):
                    target_prefix = "forward_transformer."
                elif isinstance(model, RotationLatentTransformer):
                    target_prefix = "rotation_transformer."

                cleaned_sd: Dict[str, torch.Tensor] = {}
                for k, v in ckpt.items():
                    key = k
                    if key.startswith("module."):
                        key = key[len("module.") :]
                    if target_prefix and key.startswith(target_prefix):
                        key = key[len(target_prefix) :]
                    if key in model_keys:
                        cleaned_sd[key] = v

                state_dict = cleaned_sd if cleaned_sd else ckpt
        else:
            state_dict = ckpt

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

        batch_size = latent.shape[0]

        # 1. Per-sample zero-distance mask of shape (B, 1)
        if distance_meters is None:
            zero_dist_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=latent.device
            )
        elif isinstance(distance_meters, (int, float)):
            val = distance_meters == 0
            zero_dist_mask = torch.full(
                (batch_size, 1), val, dtype=torch.bool, device=latent.device
            )
        elif isinstance(distance_meters, torch.Tensor):
            d_tensor = distance_meters.to(latent.device)
            if d_tensor.dim() == 0:
                zero_dist_mask = (d_tensor == 0).expand(batch_size, 1)
            elif d_tensor.dim() == 1:
                zero_dist_mask = (d_tensor == 0).unsqueeze(1)
            else:
                zero_dist_mask = d_tensor == 0
        else:
            zero_dist_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=latent.device
            )

        # 2. Per-sample zero-rotation mask of shape (B, 1)
        if sin_cos is not None:
            sc = sin_cos.to(latent.device)
            zero_rot_mask = (torch.abs(sc[..., 0:1]) < 1e-5) & (
                torch.abs(sc[..., 1:2] - 1.0) < 1e-5
            )
        elif angle_deg is None:
            zero_rot_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=latent.device
            )
        elif isinstance(angle_deg, (int, float)):
            val = angle_deg == 0
            zero_rot_mask = torch.full(
                (batch_size, 1), val, dtype=torch.bool, device=latent.device
            )
        elif isinstance(angle_deg, torch.Tensor):
            a_tensor = angle_deg.to(latent.device)
            if a_tensor.dim() == 0:
                zero_rot_mask = (a_tensor == 0).expand(batch_size, 1)
            elif a_tensor.dim() == 1:
                zero_rot_mask = (a_tensor == 0).unsqueeze(1)
            else:
                zero_rot_mask = a_tensor == 0
        else:
            zero_rot_mask = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=latent.device
            )

        # 3. Chained execution with per-sample identity invariance
        if order_str == ExecutionOrder.ROTATE_FIRST.value:
            if zero_rot_mask.all():
                z_mid = latent
            else:
                z_mid = self.rotation_model(
                    latent, angle_deg=angle_deg, sin_cos=sin_cos
                )
                z_mid = torch.where(zero_rot_mask, latent, z_mid)

            if zero_dist_mask.all():
                z_out = z_mid
            else:
                z_out = self.forward_model(z_mid, distance_meters=distance_meters)
                z_out = torch.where(zero_dist_mask, z_mid, z_out)
        else:
            if zero_dist_mask.all():
                z_mid = latent
            else:
                z_mid = self.forward_model(latent, distance_meters=distance_meters)
                z_mid = torch.where(zero_dist_mask, latent, z_mid)

            if zero_rot_mask.all():
                z_out = z_mid
            else:
                z_out = self.rotation_model(z_mid, angle_deg=angle_deg, sin_cos=sin_cos)
                z_out = torch.where(zero_rot_mask, z_mid, z_out)

        if return_intermediate:
            return z_out, z_mid
        return z_out

    def generate_progressive_keyframes(
        self,
        latent: torch.Tensor,
        angle_deg: float = 0.0,
        distance_meters: float = 0.0,
        execution_order: Union[str, ExecutionOrder] = ExecutionOrder.ROTATE_FIRST,
        substep_angle_deg: Optional[float] = None,
        substep_distance_meters: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate keyframe latents with optional progressive sub-stepping.

        Subdivides large angular or linear motions into intermediate sub-steps,
        producing a smooth trajectory of keyframe latents.

        Args:
            latent: Starting Latent Vector tensor of shape (1, latent_dim).
            angle_deg: Total target rotation angle in degrees.
            distance_meters: Total target forward distance in meters.
            execution_order: Motion execution order (rotate_first or forward_first).
            substep_angle_deg: Optional max angular step size in degrees.
            substep_distance_meters: Optional max linear step size in meters.

        Returns:
            Tensor of shape (num_keyframes, latent_dim) with keyframe latents.
        """
        order_str = (
            execution_order.value
            if isinstance(execution_order, ExecutionOrder)
            else str(execution_order)
        )
        key_latents: List[torch.Tensor] = [latent]
        curr_z = latent

        def _do_rotation(z: torch.Tensor) -> torch.Tensor:
            if abs(angle_deg) <= 1e-4:
                return z
            if substep_angle_deg is not None and substep_angle_deg > 0:
                n_steps = max(1, int(math.ceil(abs(angle_deg) / substep_angle_deg)))
                step_a = angle_deg / n_steps
                anchor_z = z
                cum_a = 0.0
                curr = z
                for _ in range(n_steps):
                    cum_a += step_a
                    if abs(cum_a) > 180.0:
                        anchor_z = curr
                        cum_a = step_a
                    curr = self.rotation_model(anchor_z, angle_deg=cum_a)
                    key_latents.append(curr)
                return curr
            z = self.rotation_model(z, angle_deg=angle_deg)
            key_latents.append(z)
            return z

        def _do_forward(z: torch.Tensor) -> torch.Tensor:
            if abs(distance_meters) <= 1e-4:
                return z
            if substep_distance_meters is not None and substep_distance_meters > 0:
                n_steps = max(
                    1, int(math.ceil(abs(distance_meters) / substep_distance_meters))
                )
                step_d = distance_meters / n_steps
                anchor_z = z
                cum_d = 0.0
                curr = z
                for _ in range(n_steps):
                    cum_d += step_d
                    if abs(cum_d) > 3.0:
                        anchor_z = curr
                        cum_d = step_d
                    curr = self.forward_model(anchor_z, distance_meters=cum_d)
                    key_latents.append(curr)
                return curr
            z = self.forward_model(z, distance_meters=distance_meters)
            key_latents.append(z)
            return z

        with torch.no_grad():
            if order_str == ExecutionOrder.ROTATE_FIRST.value:
                curr_z = _do_rotation(curr_z)
                curr_z = _do_forward(curr_z)
            else:
                curr_z = _do_forward(curr_z)
                curr_z = _do_rotation(curr_z)

        if len(key_latents) == 1:
            key_latents.append(curr_z)

        return torch.cat(key_latents, dim=0)

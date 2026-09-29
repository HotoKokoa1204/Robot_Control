"""
Module: joint_navigation
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Composite Multi-Branch Navigation Model integrating Shared VAE
    and Decoupled Latent Transformers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from agilab_lib.models.rlt import (
    ForwardLatentTransformer,
    RotationLatentTransformer,
)
from agilab_lib.models.vae import VAE


class JointNavigationModel(nn.Module):
    """Composite model integrating shared VAE and latent dynamics.

    Encapsulates:
    1. Shared VAE: 512-dim encoder mapping RGB frames (B, 3, 108, 192) to
       mean and logvar vectors, and visual dynamics decoder mapping 512-dim
       latents back to frames (B, 3, 108, 192).
    2. Forward Latent Transformer: 5-block conditioned residual network predicting
       latent translation transitions conditioned on linear distance.
    3. Rotation Latent Transformer: 5-block conditioned residual network predicting
       latent rotation transitions conditioned on [sin(theta), cos(theta)].
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_blocks: int = 5,
        block_inner_dim: int = 512,
        forward_num_blocks: Optional[int] = None,
        rotation_num_blocks: Optional[int] = None,
        vae: Optional[VAE] = None,
        forward_transformer: Optional[ForwardLatentTransformer] = None,
        rotation_transformer: Optional[RotationLatentTransformer] = None,
    ) -> None:
        """Initialize the Joint Navigation Model architecture.

        Args:
            latent_dim: Dimensionality of latent representation space (default 512).
            hidden_dim: Hidden dimension for feature projection in transformers (512).
            num_blocks: Default number of ConditionedResidualBlock layers (default 5).
            block_inner_dim: Hidden dimension inside each residual block (default 512).
            forward_num_blocks: Optional override for number of forward blocks.
            rotation_num_blocks: Optional override for rotation transformer blocks.
            vae: Optional pre-instantiated VAE module.
            forward_transformer: Optional pre-instantiated ForwardLatentTransformer.
            rotation_transformer: Optional pre-instantiated RotationLatentTransformer.
        """
        super().__init__()
        self.latent_dim: int = latent_dim
        self.hidden_dim: int = hidden_dim
        self.num_blocks: int = num_blocks
        self.block_inner_dim: int = block_inner_dim

        f_blocks = forward_num_blocks if forward_num_blocks is not None else num_blocks
        r_blocks = (
            rotation_num_blocks if rotation_num_blocks is not None else num_blocks
        )

        if vae is not None:
            self.vae: VAE = vae
        else:
            self.vae = VAE(latent_dim=latent_dim)

        if forward_transformer is not None:
            self.forward_transformer: ForwardLatentTransformer = forward_transformer
        else:
            self.forward_transformer = ForwardLatentTransformer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_blocks=f_blocks,
                block_inner_dim=block_inner_dim,
            )

        if rotation_transformer is not None:
            self.rotation_transformer: RotationLatentTransformer = rotation_transformer
        else:
            self.rotation_transformer = RotationLatentTransformer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_blocks=r_blocks,
                block_inner_dim=block_inner_dim,
            )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes frame into mean and log variance vectors via shared VAE.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (mu, logvar) tensors of shape (B, latent_dim).
        """
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector into reconstructed frame tensor via shared VAE.

        Args:
            z: Latent Vector tensor of shape (B, latent_dim).

        Returns:
            Reconstructed frame tensor of shape (B, 3, 108, 192).
        """
        return self.vae.decode(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deterministic Latent Vector (mu) for downstream navigation.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Latent mean vector mu of shape (B, latent_dim).
        """
        return self.vae.get_latent(x)

    def forward_reconstruction(
        self, x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Branch 1: Frame reconstruction and latent distribution inference.

        Args:
            x_t: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (recon_x, mu, logvar) where recon_x is (B, 3, 108, 192),
            and mu, logvar are (B, latent_dim).
        """
        return self.vae(x_t)

    def forward_translation(
        self,
        x_t: Optional[torch.Tensor] = None,
        distance_meters: Optional[Union[torch.Tensor, float]] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Branch 2: Forward translation dynamics prediction.

        Encodes x_t into mu_t (or uses provided latent), transforms mu_t by
        ForwardLatentTransformer conditioned on distance_meters to produce
        predicted future latent pred_latent, and decodes pred_latent to predicted image.

        Args:
            x_t: Input frame tensor (B, 3, 108, 192). Optional if latent is provided.
            distance_meters: Forward distance in meters as scalar, (B,), or (B, 1).
            latent: Optional pre-computed latent vector mu_t of shape (B, latent_dim).

        Returns:
            Tuple of (pred_img, pred_latent, mu_t) where pred_img is (B, 3, 108, 192),
            and pred_latent, mu_t are (B, latent_dim).

        Raises:
            ValueError: If neither x_t nor latent is provided.
        """
        if latent is not None:
            mu_t = latent
        elif x_t is not None:
            mu_t, _ = self.vae.encode(x_t)
        else:
            raise ValueError(
                "Either 'x_t' or 'latent' must be provided to forward_translation."
            )

        pred_latent = self.forward_transformer(mu_t, distance_meters=distance_meters)
        pred_img = self.vae.decode(pred_latent)
        return pred_img, pred_latent, mu_t

    def forward_rotation(
        self,
        x_t: Optional[torch.Tensor] = None,
        sin_cos: Optional[torch.Tensor] = None,
        angle_deg: Optional[Union[torch.Tensor, float]] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Branch 3: In-place rotation dynamics prediction.

        Encodes x_t into mu_t (or uses provided latent), transforms mu_t by
        RotationLatentTransformer conditioned on sin_cos (or angle_deg) to produce
        predicted future latent pred_latent, and decodes pred_latent to predicted image.

        Args:
            x_t: Input frame tensor (B, 3, 108, 192). Optional if latent is provided.
            sin_cos: Optional 2D unit vector tensor [sin, cos] of shape (B, 2).
            angle_deg: Optional rotation angle in degrees as scalar, (B,), or (B, 1).
            latent: Optional pre-computed latent vector mu_t of shape (B, latent_dim).

        Returns:
            Tuple of (pred_img, pred_latent, mu_t) where pred_img is (B, 3, 108, 192),
            and pred_latent, mu_t are (B, latent_dim).

        Raises:
            ValueError: If neither x_t nor latent is provided.
        """
        if latent is not None:
            mu_t = latent
        elif x_t is not None:
            mu_t, _ = self.vae.encode(x_t)
        else:
            raise ValueError(
                "Either 'x_t' or 'latent' must be provided to forward_rotation."
            )

        pred_latent = self.rotation_transformer(
            mu_t, angle_deg=angle_deg, sin_cos=sin_cos
        )
        pred_img = self.vae.decode(pred_latent)
        return pred_img, pred_latent, mu_t

    def forward(
        self,
        x_t: torch.Tensor,
        mode: str = "reconstruction",
        distance_meters: Optional[Union[torch.Tensor, float]] = None,
        sin_cos: Optional[torch.Tensor] = None,
        angle_deg: Optional[Union[torch.Tensor, float]] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Unified forward pass or branch dispatch.

        Args:
            x_t: Input frame tensor of shape (B, 3, 108, 192).
            mode: Execution mode, one of:
                - 'reconstruction' (or 'recon'): Autoencode input frame.
                - 'translation' (or 'forward'): Predict linear translation transition.
                - 'rotation' (or 'rot'): Predict angular rotation transition.
                - 'all': Compute forward passes across all active branches.
            distance_meters: Optional forward distance for translation mode.
            sin_cos: Optional [sin, cos] unit vector for rotation mode.
            angle_deg: Optional rotation angle in degrees for rotation mode.
            latent: Optional pre-computed latent tensor mu_t.

        Returns:
            Branch output tuple or dictionary of branch outputs if mode is 'all'.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        mode_lower = mode.lower()
        if mode_lower in ("reconstruction", "recon"):
            if distance_meters is not None:
                return self.forward_translation(
                    x_t=x_t, distance_meters=distance_meters, latent=latent
                )
            if sin_cos is not None or angle_deg is not None:
                return self.forward_rotation(
                    x_t=x_t, sin_cos=sin_cos, angle_deg=angle_deg, latent=latent
                )
            return self.forward_reconstruction(x_t)

        if mode_lower in ("translation", "forward", "fwd"):
            return self.forward_translation(
                x_t=x_t, distance_meters=distance_meters, latent=latent
            )

        if mode_lower in ("rotation", "rot"):
            return self.forward_rotation(
                x_t=x_t, sin_cos=sin_cos, angle_deg=angle_deg, latent=latent
            )

        if mode_lower == "all":
            recon_x, mu, logvar = self.forward_reconstruction(x_t)
            results: Dict[str, Any] = {
                "reconstruction": (recon_x, mu, logvar),
            }
            if distance_meters is not None:
                results["translation"] = self.forward_translation(
                    x_t=x_t, distance_meters=distance_meters, latent=mu
                )
            if sin_cos is not None or angle_deg is not None:
                results["rotation"] = self.forward_rotation(
                    x_t=x_t, sin_cos=sin_cos, angle_deg=angle_deg, latent=mu
                )
            return results

        raise ValueError(
            f"Unsupported mode: '{mode}'. Expected 'reconstruction', 'translation', "
            "'rotation', or 'all'."
        )

    def export_vae_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export state dictionary compatible with standalone VAE.

        Returns:
            State dictionary matching the keys and shapes of standalone VAE.
        """
        return {k: v.clone() for k, v in self.vae.state_dict().items()}

    def export_forward_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export state dictionary compatible with standalone ForwardLatentTransformer.

        Returns:
            State dictionary matching standalone ForwardLatentTransformer.
        """
        return {k: v.clone() for k, v in self.forward_transformer.state_dict().items()}

    def export_rotation_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export state dictionary compatible with standalone RotationLatentTransformer.

        Returns:
            State dictionary matching standalone RotationLatentTransformer.
        """
        return {k: v.clone() for k, v in self.rotation_transformer.state_dict().items()}

    def export_vae_checkpoint(
        self, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, torch.Tensor]:
        """Export VAE weights compatible with standalone VAE to dict and optional file.

        Args:
            checkpoint_path: Optional file path to save the weights to.

        Returns:
            State dictionary containing VAE weights.
        """
        sd = self.export_vae_state_dict()
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, p)
        return sd

    def export_forward_checkpoint(
        self, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, torch.Tensor]:
        """Export forward transformer weights to dictionary and optional file.

        Args:
            checkpoint_path: Optional file path to save the weights to.

        Returns:
            State dictionary containing forward transformer weights.
        """
        sd = self.export_forward_state_dict()
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, p)
        return sd

    def export_rotation_checkpoint(
        self, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, torch.Tensor]:
        """Export rotation transformer weights to dictionary and optional file.

        Args:
            checkpoint_path: Optional file path to save the weights to.

        Returns:
            State dictionary containing rotation transformer weights.
        """
        sd = self.export_rotation_state_dict()
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, p)
        return sd

    def load_vae_pretrained(
        self,
        checkpoint: Union[str, Path, Dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load pretrained VAE weights for warm-start initialization.

        Supports raw state dicts, checkpoint containers (e.g. 'state_dict',
        'model_state_dict', 'vae_state_dict', or 'vae'), and composite keys
        prefixed with 'vae.'.

        Args:
            checkpoint: Path to checkpoint file or pre-loaded state dictionary.
            strict: Whether to strictly enforce key matching.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
        """
        if isinstance(checkpoint, (str, Path)):
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {p}")
            state_dict = torch.load(p, map_location="cpu", weights_only=True)
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            for candidate_key in (
                "state_dict",
                "model_state_dict",
                "vae_state_dict",
                "vae",
            ):
                if candidate_key in state_dict and isinstance(
                    state_dict[candidate_key], dict
                ):
                    state_dict = state_dict[candidate_key]
                    break

            cleaned_sd: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("vae."):
                    cleaned_sd[k[len("vae.") :]] = v
                else:
                    cleaned_sd[k] = v
            state_dict = cleaned_sd

        self.vae.load_state_dict(state_dict, strict=strict)

    def load_forward_pretrained(
        self,
        checkpoint: Union[str, Path, Dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load pretrained ForwardLatentTransformer weights.

        Args:
            checkpoint: Path to checkpoint file or pre-loaded state dictionary.
            strict: Whether to strictly enforce key matching.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
        """
        if isinstance(checkpoint, (str, Path)):
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {p}")
            state_dict = torch.load(p, map_location="cpu", weights_only=True)
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            for candidate_key in (
                "state_dict",
                "model_state_dict",
                "forward_state_dict",
                "forward_transformer",
            ):
                if candidate_key in state_dict and isinstance(
                    state_dict[candidate_key], dict
                ):
                    state_dict = state_dict[candidate_key]
                    break

            cleaned_sd: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("forward_transformer."):
                    cleaned_sd[k[len("forward_transformer.") :]] = v
                else:
                    cleaned_sd[k] = v
            state_dict = cleaned_sd

        self.forward_transformer.load_state_dict(state_dict, strict=strict)

    def load_rotation_pretrained(
        self,
        checkpoint: Union[str, Path, Dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        """Load pretrained RotationLatentTransformer weights.

        Args:
            checkpoint: Path to checkpoint file or pre-loaded state dictionary.
            strict: Whether to strictly enforce key matching.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
        """
        if isinstance(checkpoint, (str, Path)):
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {p}")
            state_dict = torch.load(p, map_location="cpu", weights_only=True)
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            for candidate_key in (
                "state_dict",
                "model_state_dict",
                "rotation_state_dict",
                "rotation_transformer",
            ):
                if candidate_key in state_dict and isinstance(
                    state_dict[candidate_key], dict
                ):
                    state_dict = state_dict[candidate_key]
                    break

            cleaned_sd: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("rotation_transformer."):
                    cleaned_sd[k[len("rotation_transformer.") :]] = v
                else:
                    cleaned_sd[k] = v
            state_dict = cleaned_sd

        self.rotation_transformer.load_state_dict(state_dict, strict=strict)

    def get_parameter_groups(
        self,
        vae_lr: Optional[float] = None,
        forward_lr: Optional[float] = None,
        rotation_lr: Optional[float] = None,
        base_lr: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Constructs optimizer parameter groups for multi-branch learning rates.

        Args:
            vae_lr: Learning rate for shared VAE. Defaults to base_lr if None.
            forward_lr: Learning rate for Forward Transformer. Defaults to base_lr.
            rotation_lr: Learning rate for Rotation Transformer. Defaults to base_lr.
            base_lr: Default base learning rate.

        Returns:
            List of parameter group dictionaries suitable for PyTorch optimizers.
        """
        return [
            {
                "params": list(self.vae.parameters()),
                "lr": vae_lr if vae_lr is not None else base_lr,
                "name": "vae",
            },
            {
                "params": list(self.forward_transformer.parameters()),
                "lr": forward_lr if forward_lr is not None else base_lr,
                "name": "forward_transformer",
            },
            {
                "params": list(self.rotation_transformer.parameters()),
                "lr": rotation_lr if rotation_lr is not None else base_lr,
                "name": "rotation_transformer",
            },
        ]

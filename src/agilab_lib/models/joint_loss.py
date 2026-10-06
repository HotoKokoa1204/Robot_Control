"""
Module: joint_loss
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Joint multi-branch loss function coordinating reconstruction,
    forward translation dynamics, and in-place rotation dynamics with
    perceptual and latent KL regularization.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from PIL import Image  # isort: skip # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilab_lib.models.perceptual import VGGPerceptualLoss

if TYPE_CHECKING:
    from agilab_lib.models.joint_navigation import JointNavigationModel


def _extract_tensor(
    batch: Any,
    key: str,
    *alt_keys: str,
) -> Optional[torch.Tensor]:
    """Helper to extract a tensor from a dictionary, batch object, or container.

    Checks dictionary indexing (`batch[key]`), attribute access (`getattr(batch, key)`),
    and properties, converting numeric values to tensors if necessary.

    Args:
        batch: Dictionary, DualSourceBatch, or object holding tensors.
        key: Primary candidate key name to search for.
        *alt_keys: Additional fallback candidate key names.

    Returns:
        Found tensor or None.
    """
    if batch is None:
        return None

    keys = (key,) + alt_keys
    for k in keys:
        val = None

        # 1. Dictionary indexing
        if isinstance(batch, dict) or hasattr(batch, "__getitem__"):
            try:
                if k in batch and batch[k] is not None:
                    val = batch[k]
            except (TypeError, KeyError, IndexError):
                pass

        # 2. Attribute / property access
        if val is None and hasattr(batch, k):
            try:
                attr_val = getattr(batch, k)
                if attr_val is not None:
                    val = attr_val
            except Exception:
                pass

        if val is not None:
            if isinstance(val, torch.Tensor):
                return val
            if isinstance(val, (int, float, list, tuple, np.ndarray)):
                try:
                    return torch.as_tensor(val, dtype=torch.float32)
                except Exception:
                    pass

    return None


class JointLossOutput(Tuple[torch.Tensor, Dict[str, torch.Tensor]]):
    """Container holding total composite loss and detailed telemetry metrics.

    Allows standard tuple unpacking `(total_loss, metrics) = loss_fn(...)` as well
    as dictionary indexing `output['total_loss']`, `output['loss_recon']`, or
    attribute access `output.total_loss`, `output.metrics`.
    """

    def __new__(
        cls,
        total_loss: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
    ) -> "JointLossOutput":
        """Instantiate new JointLossOutput container.

        Args:
            total_loss: Total scalar loss tensor.
            metrics: Dictionary mapping loss component names to scalar tensors.

        Returns:
            JointLossOutput instance.
        """
        return super().__new__(cls, (total_loss, metrics))

    @property
    def total_loss(self) -> torch.Tensor:
        """Scalar total composite loss tensor."""
        return self[0]

    @property
    def loss(self) -> torch.Tensor:
        """Alias for total_loss."""
        return self[0]

    @property
    def metrics(self) -> Dict[str, torch.Tensor]:
        """Dictionary of individual loss components."""
        return self[1]

    def __getitem__(self, key: Any) -> Any:
        """Retrieve by tuple index or dictionary key.

        Args:
            key: Integer index or string metric name.

        Returns:
            Indexed tuple element or metric value.

        Raises:
            KeyError: If string metric name is not found.
        """
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        if isinstance(key, str):
            if key in self[1]:
                return self[1][key]
            if key in ("total_loss", "loss"):
                return self[0]
            raise KeyError(f"Metric '{key}' not found in JointLossOutput.")
        return super().__getitem__(key)

    def keys(self) -> Any:
        """Return metric keys view.

        Returns:
            Keys view for metrics dictionary.
        """
        return self[1].keys()

    def values(self) -> Any:
        """Return metric values view.

        Returns:
            Values view for metrics dictionary.
        """
        return self[1].values()

    def items(self) -> Any:
        """Return metric items view.

        Returns:
            Items view for metrics dictionary.
        """
        return self[1].items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get metric value by key with fallback default.

        Args:
            key: Metric name.
            default: Fallback return value if key is not found.

        Returns:
            Metric tensor or default value.
        """
        if key in self[1]:
            return self[1][key]
        if key in ("total_loss", "loss"):
            return self[0]
        return default


class JointNavigationLoss(nn.Module):
    """Joint multi-branch loss function for end-to-end navigation learning.

    Coordinates pixel-level reconstruction and latent dynamics supervision:
    - Branch 1 (Reconstruction):
        L_recon = (
            MSE(recon_x, x_t) + alpha_perc * VGG(recon_x, x_t)
            + beta_kl * KL(mu_t, logvar_t)
        )
    - Branch 2 (Forward Translation Dynamics):
        L_fwd = MSE(pred_fwd, target_fwd) + alpha_perc * VGG(pred_fwd, target_fwd)
    - Branch 3 (Rotation Dynamics):
        L_rot = MSE(pred_rot, target_rot) + alpha_perc * VGG(pred_rot, target_rot)
    - Total Objective:
        L_total = w_recon * L_recon + w_fwd * L_fwd + w_rot * L_rot

    Attributes:
        alpha_perc: Weight for VGG perceptual loss terms (default 0.5).
        beta_kl: Weight for latent KL divergence regularizer (default 0.0001).
        w_fwd: Weight for forward translation dynamics branch (default 1.0).
        w_rot: Weight for rotation dynamics branch (default 1.0).
        w_recon: Weight for reconstruction branch (default 1.0).
        w_latent: Weight for latent-space MSE alignment loss (default 1.0).
        perceptual_loss: VGGPerceptualLoss instance with frozen weights.
    """

    def __init__(
        self,
        alpha_perc: float = 0.5,
        beta_kl: float = 0.0001,
        w_fwd: float = 1.0,
        w_rot: float = 1.0,
        w_recon: float = 1.0,
        w_latent: float = 1.0,
        perceptual_loss: Optional[VGGPerceptualLoss] = None,
        pretrained_vgg: bool = True,
        perceptual_loss_type: str = "l1",
    ) -> None:
        """Initialize JointNavigationLoss architecture and hyperparameters.

        Args:
            alpha_perc: Perceptual loss weight across all branches. Defaults to 0.5.
            beta_kl: KL divergence regularization weight for Branch 1.
                Defaults to 0.0001.
            w_fwd: Forward dynamics branch loss weight. Defaults to 1.0.
            w_rot: Rotation dynamics branch loss weight. Defaults to 1.0.
            w_recon: Reconstruction branch loss weight. Defaults to 1.0.
            w_latent: Latent-space MSE alignment loss weight. Defaults to 1.0.
            perceptual_loss: Optional pre-instantiated VGGPerceptualLoss module.
            pretrained_vgg: Whether to load pretrained ImageNet weights if
                perceptual_loss is None.
            perceptual_loss_type: Distance metric for perceptual loss ('l1' or 'mse').
                Defaults to 'l1'.
        """
        super().__init__()
        self.alpha_perc: float = float(alpha_perc)
        self.beta_kl: float = float(beta_kl)
        self.w_fwd: float = float(w_fwd)
        self.w_rot: float = float(w_rot)
        self.w_recon: float = float(w_recon)
        self.w_latent: float = float(w_latent)

        if perceptual_loss is not None:
            self.perceptual_loss: VGGPerceptualLoss = perceptual_loss
        else:
            self.perceptual_loss = VGGPerceptualLoss(
                loss_type=perceptual_loss_type,
                pretrained=pretrained_vgg,
            )

    @staticmethod
    def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute standard Gaussian KL divergence for latent distributions.

        Args:
            mu: Mean tensor of shape (B, latent_dim).
            logvar: Log variance tensor of shape (B, latent_dim).

        Returns:
            Scalar KL divergence loss tensor.
        """
        return -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

    def forward_model(
        self,
        model: Union["JointNavigationModel", nn.Module, Any],
        batch: Union[Dict[str, Any], Any],
        stage: int = 1,
        stage1_recon_only: Optional[bool] = None,
    ) -> JointLossOutput:
        """Compute joint loss by executing stage-specific forward passes through model.

        Supports three curriculum training stages:
        - Stage 1: Autoencoder Reconstruction only. No transformer forward passes.
        - Stage 2: Latent dynamics training with detached base latents (or frozen VAE).
          Visual Dynamics Decoder decodes predictions. Reconstruction is inactive.
        - Stage 3: End-to-end multi-branch joint training. Base latents encoded with
          gradients; target latents detached; reconstruction active.

        Args:
            model: JointNavigationModel instance.
            batch: Paired batch (DualSourceBatch or dictionary) containing
                'recon_frame', 'fwd_current', 'fwd_target', 'fwd_distance',
                'rot_current', 'rot_target', and 'rot_sin_cos'.
            stage: Training stage (1, 2, or 3). Defaults to 1.
            stage1_recon_only: Legacy parameter for backwards compatibility.
                When True, forces stage=1. When False, defaults to stage=3.

        Returns:
            JointLossOutput containing total loss and detailed metrics dictionary.

        Raises:
            ValueError: If stage is not 1, 2, or 3.
            KeyError: If required keys are missing from batch for the selected stage.
        """
        # Retain backwards-compatibility if stage1_recon_only is passed
        if stage1_recon_only is not None:
            if stage1_recon_only:
                stage = 1
            else:
                if stage == 1:
                    stage = 3

        if stage not in (1, 2, 3):
            raise ValueError(f"Invalid stage={stage}. Must be 1, 2, or 3.")

        # Stage 1: Autoencoder Reconstruction Only
        if stage == 1:
            recon_frame = _extract_tensor(batch, "recon_frame", "x_t")
            if recon_frame is None:
                raise KeyError("Batch must contain 'recon_frame' or 'x_t'.")

            recon_x, mu_t, logvar_t = model.forward_reconstruction(recon_frame)
            return self.forward(
                recon_x=recon_x,
                x_t=recon_frame,
                mu_t=mu_t,
                logvar_t=logvar_t,
            )

        # Common extraction for dynamics branches (Stages 2 and 3)
        fwd_current = _extract_tensor(batch, "fwd_current")
        fwd_target = _extract_tensor(batch, "fwd_target")
        fwd_distance = _extract_tensor(batch, "fwd_distance")
        rot_current = _extract_tensor(batch, "rot_current")
        rot_target = _extract_tensor(batch, "rot_target")
        rot_sin_cos = _extract_tensor(batch, "rot_sin_cos")
        rot_angle_deg = _extract_tensor(batch, "rot_angle_deg")

        # Forward dynamics execution
        pred_fwd = None
        pred_fwd_latent = None
        mu_fwd_target = None
        if fwd_current is not None and fwd_distance is not None:
            if stage == 2:
                mu_fwd_base = model.vae.encode(fwd_current)[0].detach()
            else:
                mu_fwd_base = model.vae.encode(fwd_current)[0]

            pred_fwd_latent = model.forward_transformer(
                mu_fwd_base, distance_meters=fwd_distance
            )
            pred_fwd = model.vae.decode(pred_fwd_latent)

            if fwd_target is not None:
                mu_fwd_target = model.vae.encode(fwd_target)[0].detach()

        # Rotation dynamics execution
        pred_rot = None
        pred_rot_latent = None
        mu_rot_target = None
        if rot_current is not None and (
            rot_sin_cos is not None or rot_angle_deg is not None
        ):
            if stage == 2:
                mu_rot_base = model.vae.encode(rot_current)[0].detach()
            else:
                mu_rot_base = model.vae.encode(rot_current)[0]

            if rot_sin_cos is not None:
                pred_rot_latent = model.rotation_transformer(
                    mu_rot_base, sin_cos=rot_sin_cos
                )
            else:
                pred_rot_latent = model.rotation_transformer(
                    mu_rot_base, angle_deg=rot_angle_deg
                )
            pred_rot = model.vae.decode(pred_rot_latent)

            if rot_target is not None:
                mu_rot_target = model.vae.encode(rot_target)[0].detach()

        if stage == 2:
            return self.forward(
                pred_fwd=pred_fwd,
                target_fwd=fwd_target,
                pred_rot=pred_rot,
                target_rot=rot_target,
                pred_fwd_latent=pred_fwd_latent,
                mu_fwd_target=mu_fwd_target,
                pred_rot_latent=pred_rot_latent,
                mu_rot_target=mu_rot_target,
            )

        # Stage 3: End-to-end multi-branch with reconstruction
        recon_frame = _extract_tensor(batch, "recon_frame", "x_t")
        if recon_frame is None:
            raise KeyError("Batch must contain 'recon_frame' or 'x_t'.")

        recon_x, mu_t, logvar_t = model.forward_reconstruction(recon_frame)

        return self.forward(
            recon_x=recon_x,
            x_t=recon_frame,
            mu_t=mu_t,
            logvar_t=logvar_t,
            pred_fwd=pred_fwd,
            target_fwd=fwd_target,
            pred_rot=pred_rot,
            target_rot=rot_target,
            pred_fwd_latent=pred_fwd_latent,
            mu_fwd_target=mu_fwd_target,
            pred_rot_latent=pred_rot_latent,
            mu_rot_target=mu_rot_target,
        )

    def forward(
        self,
        predictions: Optional[Any] = None,
        targets: Optional[Any] = None,
        mu_t: Optional[torch.Tensor] = None,
        logvar_t: Optional[torch.Tensor] = None,
        pred_fwd: Optional[torch.Tensor] = None,
        target_fwd: Optional[torch.Tensor] = None,
        pred_rot: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
        pred_fwd_latent: Optional[torch.Tensor] = None,
        mu_fwd_target: Optional[torch.Tensor] = None,
        pred_rot_latent: Optional[torch.Tensor] = None,
        mu_rot_target: Optional[torch.Tensor] = None,
        *,
        stage: Optional[int] = None,
        stage1_recon_only: Optional[bool] = None,
        recon_x: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        target_recon: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        fwd_target: Optional[torch.Tensor] = None,
        rot_target: Optional[torch.Tensor] = None,
        fwd_latent_pred: Optional[torch.Tensor] = None,
        target_fwd_latent: Optional[torch.Tensor] = None,
        rot_latent_pred: Optional[torch.Tensor] = None,
        target_rot_latent: Optional[torch.Tensor] = None,
        model: Optional[Any] = None,
        batch: Optional[Any] = None,
    ) -> JointLossOutput:
        """Compute composite multi-branch loss and detailed telemetry metrics.

        Accepts explicit tensors, paired dictionary containers, DualSourceBatch
        instances, or direct model and batch execution.

        Args:
            predictions: Dictionary or container of model predictions, or first
                positional tensor.
            targets: Dictionary or DualSourceBatch of ground-truth targets, or
                second positional tensor.
            mu_t: Latent mean vector of shape (B, latent_dim).
            logvar_t: Latent log variance vector of shape (B, latent_dim).
            pred_fwd: Predicted forward translation frame (B, 3, H, W).
            target_fwd: Ground-truth forward target frame (B, 3, H, W).
            pred_rot: Predicted rotation frame (B, 3, H, W).
            target_rot: Ground-truth rotation target frame (B, 3, H, W).
            pred_fwd_latent: Predicted forward latent vector (B, latent_dim).
            mu_fwd_target: Ground-truth forward target latent vector (B, latent_dim).
            pred_rot_latent: Predicted rotation latent vector (B, latent_dim).
            mu_rot_target: Ground-truth rotation target latent vector (B, latent_dim).
            stage: Optional curriculum stage (1, 2, or 3) when dispatching
                model and batch.
            stage1_recon_only: Optional legacy stage 1 flag.
            recon_x: Keyword argument for reconstructed frame (B, 3, H, W).
            x_t: Keyword argument for reconstruction target frame (B, 3, H, W).
            target_recon: Alias for x_t.
            mu: Alias for mu_t.
            logvar: Alias for logvar_t.
            fwd_target: Alias for target_fwd.
            rot_target: Alias for target_rot.
            fwd_latent_pred: Alias for pred_fwd_latent.
            target_fwd_latent: Alias for mu_fwd_target.
            rot_latent_pred: Alias for pred_rot_latent.
            target_rot_latent: Alias for mu_rot_target.
            model: Optional JointNavigationModel instance for integrated forward
                execution.
            batch: Optional batch instance when model is supplied.

        Returns:
            JointLossOutput containing total composite loss scalar tensor and
            detailed metrics dictionary.

        Raises:
            ValueError: If input tensor shapes mismatch or no valid branch is provided.
        """
        # Handle model + batch dispatch
        if model is not None and batch is not None:
            stage_arg = 1 if stage is None else stage
            return self.forward_model(
                model, batch, stage=stage_arg, stage1_recon_only=stage1_recon_only
            )
        if isinstance(predictions, nn.Module) and targets is not None:
            stage_arg = 1 if stage is None else stage
            return self.forward_model(
                predictions,
                targets,
                stage=stage_arg,
                stage1_recon_only=stage1_recon_only,
            )
        if isinstance(targets, nn.Module) and predictions is not None:
            stage_arg = 1 if stage is None else stage
            return self.forward_model(
                targets,
                predictions,
                stage=stage_arg,
                stage1_recon_only=stage1_recon_only,
            )

        # Handle positional tensor inputs: (recon_x, x_t, mu_t, ...)
        if isinstance(predictions, torch.Tensor):
            if recon_x is None:
                recon_x = predictions
            if isinstance(targets, torch.Tensor) and x_t is None:
                x_t = targets
            predictions = None
            targets = None

        # Extract from predictions container if provided
        if predictions is not None:
            if recon_x is None:
                recon_x = _extract_tensor(
                    predictions, "recon_x", "pred_recon", "recon_frame", "recon"
                )
            if mu_t is None:
                mu_t = _extract_tensor(
                    predictions, "mu_t", "mu", "latent_mu", "mu_recon"
                )
            if logvar_t is None:
                logvar_t = _extract_tensor(
                    predictions,
                    "logvar_t",
                    "logvar",
                    "latent_logvar",
                    "logvar_recon",
                )
            if pred_fwd is None:
                pred_fwd = _extract_tensor(
                    predictions,
                    "pred_fwd",
                    "fwd_pred",
                    "pred_translation",
                    "translation_pred",
                )
            if pred_rot is None:
                pred_rot = _extract_tensor(
                    predictions,
                    "pred_rot",
                    "rot_pred",
                    "pred_rotation",
                    "rotation_pred",
                )
            if pred_fwd_latent is None:
                pred_fwd_latent = _extract_tensor(
                    predictions,
                    "pred_fwd_latent",
                    "fwd_pred_latent",
                    "fwd_latent_pred",
                    "fwd_latent",
                )
            if pred_rot_latent is None:
                pred_rot_latent = _extract_tensor(
                    predictions,
                    "pred_rot_latent",
                    "rot_pred_latent",
                    "rot_latent_pred",
                    "rot_latent",
                )

            # If predictions also includes targets (single container call)
            if targets is None:
                if x_t is None:
                    x_t = _extract_tensor(
                        predictions,
                        "recon_frame",
                        "x_t",
                        "target_recon",
                        "recon_target",
                    )
                if target_fwd is None:
                    target_fwd = _extract_tensor(
                        predictions,
                        "fwd_target",
                        "target_fwd",
                        "fwd_future",
                        "target_translation",
                    )
                if target_rot is None:
                    target_rot = _extract_tensor(
                        predictions,
                        "rot_target",
                        "target_rot",
                        "rot_future",
                        "target_rotation",
                    )
                if mu_fwd_target is None:
                    mu_fwd_target = _extract_tensor(
                        predictions,
                        "mu_fwd_target",
                        "fwd_target_latent",
                        "target_fwd_latent",
                        "mu_fwd",
                    )
                if mu_rot_target is None:
                    mu_rot_target = _extract_tensor(
                        predictions,
                        "mu_rot_target",
                        "rot_target_latent",
                        "target_rot_latent",
                        "mu_rot",
                    )

        # Extract from targets container if provided
        if targets is not None:
            if x_t is None:
                x_t = _extract_tensor(
                    targets,
                    "recon_frame",
                    "x_t",
                    "target_recon",
                    "recon_target",
                )
            if target_fwd is None:
                target_fwd = _extract_tensor(
                    targets,
                    "fwd_target",
                    "target_fwd",
                    "fwd_future",
                    "target_translation",
                )
            if target_rot is None:
                target_rot = _extract_tensor(
                    targets,
                    "rot_target",
                    "target_rot",
                    "rot_future",
                    "target_rotation",
                )
            if mu_fwd_target is None:
                mu_fwd_target = _extract_tensor(
                    targets,
                    "mu_fwd_target",
                    "fwd_target_latent",
                    "target_fwd_latent",
                    "mu_fwd",
                )
            if mu_rot_target is None:
                mu_rot_target = _extract_tensor(
                    targets,
                    "mu_rot_target",
                    "rot_target_latent",
                    "target_rot_latent",
                    "mu_rot",
                )

        # Apply keyword aliases
        if x_t is None and target_recon is not None:
            x_t = target_recon
        if mu_t is None and mu is not None:
            mu_t = mu
        if logvar_t is None and logvar is not None:
            logvar_t = logvar
        if target_fwd is None and fwd_target is not None:
            target_fwd = fwd_target
        if target_rot is None and rot_target is not None:
            target_rot = rot_target
        if pred_fwd_latent is None and fwd_latent_pred is not None:
            pred_fwd_latent = fwd_latent_pred
        if mu_fwd_target is None and target_fwd_latent is not None:
            mu_fwd_target = target_fwd_latent
        if pred_rot_latent is None and rot_latent_pred is not None:
            pred_rot_latent = rot_latent_pred
        if mu_rot_target is None and target_rot_latent is not None:
            mu_rot_target = target_rot_latent

        # Validate provided branch tensor pairs and shapes
        has_recon = recon_x is not None and x_t is not None
        has_fwd = (pred_fwd is not None and target_fwd is not None) or (
            pred_fwd_latent is not None and mu_fwd_target is not None
        )
        has_rot = (pred_rot is not None and target_rot is not None) or (
            pred_rot_latent is not None and mu_rot_target is not None
        )

        if not (has_recon or has_fwd or has_rot):
            raise ValueError(
                "JointNavigationLoss requires at least one active branch with "
                "both prediction and target tensors provided."
            )

        if has_recon:
            assert recon_x is not None and x_t is not None
            if recon_x.shape != x_t.shape:
                raise ValueError(
                    f"Reconstruction shape mismatch: recon_x {recon_x.shape} "
                    f"!= x_t {x_t.shape}."
                )

        if pred_fwd is not None and target_fwd is not None:
            if pred_fwd.shape != target_fwd.shape:
                raise ValueError(
                    f"Forward dynamics shape mismatch: pred_fwd {pred_fwd.shape} "
                    f"!= target_fwd {target_fwd.shape}."
                )

        if pred_rot is not None and target_rot is not None:
            if pred_rot.shape != target_rot.shape:
                raise ValueError(
                    f"Rotation dynamics shape mismatch: pred_rot {pred_rot.shape} "
                    f"!= target_rot {target_rot.shape}."
                )

        if pred_fwd_latent is not None and mu_fwd_target is not None:
            if pred_fwd_latent.shape != mu_fwd_target.shape:
                raise ValueError(
                    f"Forward latent shape mismatch: pred_fwd_latent "
                    f"{pred_fwd_latent.shape} != mu_fwd_target {mu_fwd_target.shape}."
                )

        if pred_rot_latent is not None and mu_rot_target is not None:
            if pred_rot_latent.shape != mu_rot_target.shape:
                raise ValueError(
                    f"Rotation latent shape mismatch: pred_rot_latent "
                    f"{pred_rot_latent.shape} != mu_rot_target {mu_rot_target.shape}."
                )

        if mu_t is not None and logvar_t is not None:
            if mu_t.shape != logvar_t.shape:
                raise ValueError(
                    f"Latent vector shape mismatch: mu_t {mu_t.shape} "
                    f"!= logvar_t {logvar_t.shape}."
                )

        # Determine reference device and floating point dtype
        ref_tensor = (
            recon_x
            if recon_x is not None
            else (
                pred_fwd
                if pred_fwd is not None
                else (
                    pred_rot
                    if pred_rot is not None
                    else (
                        pred_fwd_latent
                        if pred_fwd_latent is not None
                        else pred_rot_latent
                    )
                )
            )
        )
        assert ref_tensor is not None
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        # Branch 1: Reconstruction loss
        if has_recon:
            assert recon_x is not None and x_t is not None
            recon_mse = F.mse_loss(recon_x, x_t, reduction="mean")
            if self.alpha_perc > 0.0:
                recon_perc = self.perceptual_loss(recon_x, x_t)
            else:
                recon_perc = torch.zeros((), device=device, dtype=dtype)

            if mu_t is not None and logvar_t is not None and self.beta_kl > 0.0:
                recon_kl = self.compute_kl_divergence(mu_t, logvar_t)
            else:
                recon_kl = torch.zeros((), device=device, dtype=dtype)

            loss_recon = (
                recon_mse + self.alpha_perc * recon_perc + self.beta_kl * recon_kl
            )
        else:
            recon_mse = torch.zeros((), device=device, dtype=dtype)
            recon_perc = torch.zeros((), device=device, dtype=dtype)
            recon_kl = torch.zeros((), device=device, dtype=dtype)
            loss_recon = torch.zeros((), device=device, dtype=dtype)

        # Branch 2: Forward translation dynamics loss
        if has_fwd:
            if pred_fwd is not None and target_fwd is not None:
                fwd_mse = F.mse_loss(pred_fwd, target_fwd, reduction="mean")
                if self.alpha_perc > 0.0:
                    fwd_perc = self.perceptual_loss(pred_fwd, target_fwd)
                else:
                    fwd_perc = torch.zeros((), device=device, dtype=dtype)
            else:
                fwd_mse = torch.zeros((), device=device, dtype=dtype)
                fwd_perc = torch.zeros((), device=device, dtype=dtype)

            if pred_fwd_latent is not None and mu_fwd_target is not None:
                fwd_latent_mse = F.mse_loss(
                    pred_fwd_latent, mu_fwd_target, reduction="mean"
                )
            else:
                fwd_latent_mse = torch.zeros((), device=device, dtype=dtype)

            loss_fwd = (
                fwd_mse + self.alpha_perc * fwd_perc + self.w_latent * fwd_latent_mse
            )
        else:
            fwd_mse = torch.zeros((), device=device, dtype=dtype)
            fwd_perc = torch.zeros((), device=device, dtype=dtype)
            fwd_latent_mse = torch.zeros((), device=device, dtype=dtype)
            loss_fwd = torch.zeros((), device=device, dtype=dtype)

        # Branch 3: Rotation dynamics loss
        if has_rot:
            if pred_rot is not None and target_rot is not None:
                rot_mse = F.mse_loss(pred_rot, target_rot, reduction="mean")
                if self.alpha_perc > 0.0:
                    rot_perc = self.perceptual_loss(pred_rot, target_rot)
                else:
                    rot_perc = torch.zeros((), device=device, dtype=dtype)
            else:
                rot_mse = torch.zeros((), device=device, dtype=dtype)
                rot_perc = torch.zeros((), device=device, dtype=dtype)

            if pred_rot_latent is not None and mu_rot_target is not None:
                rot_latent_mse = F.mse_loss(
                    pred_rot_latent, mu_rot_target, reduction="mean"
                )
            else:
                rot_latent_mse = torch.zeros((), device=device, dtype=dtype)

            loss_rot = (
                rot_mse + self.alpha_perc * rot_perc + self.w_latent * rot_latent_mse
            )
        else:
            rot_mse = torch.zeros((), device=device, dtype=dtype)
            rot_perc = torch.zeros((), device=device, dtype=dtype)
            rot_latent_mse = torch.zeros((), device=device, dtype=dtype)
            loss_rot = torch.zeros((), device=device, dtype=dtype)

        # Total multi-branch objective
        total_loss = torch.zeros((), device=device, dtype=dtype)
        if has_recon and self.w_recon != 0.0:
            total_loss = total_loss + self.w_recon * loss_recon
        if has_fwd and self.w_fwd != 0.0:
            total_loss = total_loss + self.w_fwd * loss_fwd
        if has_rot and self.w_rot != 0.0:
            total_loss = total_loss + self.w_rot * loss_rot

        metrics: Dict[str, torch.Tensor] = {
            "total_loss": total_loss,
            "loss": total_loss,
            "loss_recon": loss_recon,
            "recon_mse": recon_mse,
            "recon_perc": recon_perc,
            "recon_kl": recon_kl,
            "loss_fwd": loss_fwd,
            "fwd_mse": fwd_mse,
            "fwd_perc": fwd_perc,
            "fwd_latent_mse": fwd_latent_mse,
            "loss_rot": loss_rot,
            "rot_mse": rot_mse,
            "rot_perc": rot_perc,
            "rot_latent_mse": rot_latent_mse,
        }

        return JointLossOutput(total_loss=total_loss, metrics=metrics)

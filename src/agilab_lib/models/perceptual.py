"""
Module: perceptual
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Frozen VGG-16 Perceptual Loss module with multi-scale features.
"""

import warnings
from typing import Dict, List, Optional, Sequence

from PIL import Image  # isort: skip # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """Frozen VGG-16 Perceptual Loss module with multi-scale feature representations.

    Extracts multi-scale representations from frozen VGG-16 layers (relu1_2,
    relu2_2, relu3_3, and relu4_3) to compute perceptual distance between
    predicted and target image frames.

    Attributes:
        slices: ModuleList containing the 4 feature extraction stages.
        weights: Weights applied to each slice's loss contribution.
        loss_type: Distance metric ('l1' or 'mse').
        normalize_inputs: Whether input frames are normalized using ImageNet stats.
        mean: Registered buffer for ImageNet channel means.
        std: Registered buffer for ImageNet channel standard deviations.
    """

    DEFAULT_WEIGHTS: List[float] = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0]
    LAYER_NAMES: List[str] = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        loss_type: str = "l1",
        pretrained: bool = True,
        normalize_inputs: bool = True,
    ) -> None:
        """Initialize VGGPerceptualLoss architecture.

        Args:
            weights: Weights for [relu1_2, relu2_2, relu3_3, relu4_3]. Defaults to
                [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4].
            loss_type: Feature distance metric, either 'l1' or 'mse'.
            pretrained: Whether to attempt loading pretrained ImageNet weights.
            normalize_inputs: Whether to normalize [0, 1] RGB frames with ImageNet
                statistics.

        Raises:
            ValueError: If weights length is not 4 or loss_type is unsupported.
        """
        super().__init__()

        if weights is None:
            self.weights: List[float] = list(self.DEFAULT_WEIGHTS)
        else:
            if len(weights) != 4:
                raise ValueError(
                    f"Expected 4 weights for 4 VGG slices, got {len(weights)}."
                )
            self.weights = [float(w) for w in weights]

        loss_type_lower = loss_type.lower()
        if loss_type_lower == "l1":
            self._criterion = F.l1_loss
        elif loss_type_lower in ("mse", "l2"):
            self._criterion = F.mse_loss
        else:
            raise ValueError(
                f"Unsupported loss_type: '{loss_type}'. Must be 'l1' or 'mse'."
            )
        self.loss_type = loss_type_lower
        self.normalize_inputs = normalize_inputs

        # Load VGG-16 backbone with fallback for offline environments
        vgg_features = self._load_vgg_features(pretrained=pretrained)

        # Build feature slices:
        # slice 1: layers 0..3 -> output of relu1_2
        # slice 2: layers 4..8 -> output of relu2_2
        # slice 3: layers 9..15 -> output of relu3_3
        # slice 4: layers 16..22 -> output of relu4_3
        self.slices = nn.ModuleList(
            [
                nn.Sequential(*[vgg_features[x] for x in range(0, 4)]),
                nn.Sequential(*[vgg_features[x] for x in range(4, 9)]),
                nn.Sequential(*[vgg_features[x] for x in range(9, 16)]),
                nn.Sequential(*[vgg_features[x] for x in range(16, 23)]),
            ]
        )

        # Expose individual slices as attributes for convenience
        self.slice1 = self.slices[0]
        self.slice2 = self.slices[1]
        self.slice3 = self.slices[2]
        self.slice4 = self.slices[3]

        # Prevent inplace modifications in ReLUs which interfere with autograd
        for module in self.slices.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # ImageNet normalization statistics:
        # mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _load_vgg_features(pretrained: bool) -> nn.Sequential:
        """Load VGG-16 features with graceful offline fallback.

        Args:
            pretrained: Whether to attempt loading pretrained weights.

        Returns:
            nn.Sequential of VGG-16 feature layers.
        """
        if pretrained:
            try:
                weights = getattr(models, "VGG16_Weights", None)
                if weights is not None:
                    vgg = models.vgg16(weights=weights.DEFAULT)
                else:
                    vgg = models.vgg16(pretrained=True)
                return vgg.features
            except Exception as err:
                warnings.warn(
                    f"Failed to load pretrained VGG-16 weights ({err}). "
                    "Falling back to uninitialized weights. "
                    "Perceptual loss may be degraded.",
                    UserWarning,
                    stacklevel=2,
                )
                vgg = models.vgg16(weights=None)
                return vgg.features
        vgg = models.vgg16(weights=None)
        return vgg.features

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        """Keep module permanently in evaluation mode.

        Args:
            mode: Ignored; module is always kept in eval mode.

        Returns:
            Self in evaluation mode.
        """
        return super().train(False)

    @property
    def normalize_input(self) -> bool:
        """Alias for normalize_inputs."""
        return self.normalize_inputs

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input frame tensor with ImageNet mean and std.

        Args:
            x: Input frame tensor of shape (B, 3, H, W) with values in [0, 1].

        Returns:
            Normalized tensor.
        """
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature maps from an input frame tensor.

        Args:
            x: Input frame tensor of shape (B, 3, H, W) with values in [0, 1].

        Returns:
            List of 4 feature tensors corresponding to relu1_2, relu2_2, relu3_3,
            and relu4_3.

        Raises:
            ValueError: If input tensor does not have shape (B, 3, H, W).
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(
                f"Expected input tensor of shape (B, 3, H, W), got {x.shape}."
            )

        h = self._normalize(x) if self.normalize_inputs else x
        features: List[torch.Tensor] = []
        for slice_module in self.slices:
            h = slice_module(h)
            features.append(h)
        return features

    def get_layer_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute unweighted per-layer feature distances.

        Args:
            pred: Predicted frame tensor of shape (B, 3, H, W).
            target: Target frame tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping layer names to their individual scalar loss tensors.

        Raises:
            ValueError: If pred and target shapes mismatch or are not
                (B, 3, H, W).
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.shape} "
                f"!= target shape {target.shape}."
            )
        if pred.dim() != 4 or pred.size(1) != 3:
            raise ValueError(
                f"Expected input tensors of shape (B, 3, H, W), got {pred.shape}."
            )

        h_pred = self._normalize(pred) if self.normalize_inputs else pred
        h_target = self._normalize(target) if self.normalize_inputs else target

        layer_losses: Dict[str, torch.Tensor] = {}
        for name, slice_module in zip(self.LAYER_NAMES, self.slices):
            h_pred = slice_module(h_pred)
            if not target.requires_grad:
                with torch.no_grad():
                    h_target = slice_module(h_target)
            else:
                h_target = slice_module(h_target)
            layer_losses[name] = self._criterion(h_pred, h_target)
        return layer_losses

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate weighted multi-scale perceptual loss between prediction and target.

        Args:
            pred: Predicted frame tensor of shape (B, 3, H, W) in range [0, 1].
            target: Target frame tensor of shape (B, 3, H, W) in range [0, 1].

        Returns:
            Scalar tensor representing the weighted perceptual loss.

        Raises:
            ValueError: If pred and target shapes mismatch or are not
                (B, 3, H, W).
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.shape} "
                f"!= target shape {target.shape}."
            )
        if pred.dim() != 4 or pred.size(1) != 3:
            raise ValueError(
                f"Expected input tensors of shape (B, 3, H, W), got {pred.shape}."
            )

        h_pred = self._normalize(pred) if self.normalize_inputs else pred
        h_target = self._normalize(target) if self.normalize_inputs else target

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for slice_module, weight in zip(self.slices, self.weights):
            h_pred = slice_module(h_pred)
            if not target.requires_grad:
                with torch.no_grad():
                    h_target = slice_module(h_target)
            else:
                h_target = slice_module(h_target)
            loss = loss + weight * self._criterion(h_pred, h_target)

        return loss

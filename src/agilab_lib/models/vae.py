"""Module: vae
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Modernized ResBlock Autoencoder (AE / VAE) for visual learning.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    """Creates a standard 8-group GroupNorm module.

    Args:
        c: Number of channels.

    Returns:
        GroupNorm module configured with min(8, c) groups.
    """
    num_groups = min(8, c)
    return nn.GroupNorm(num_groups, c)


class ResBlock(nn.Module):
    """Residual convolutional block with GroupNorm and SiLU activations."""

    def __init__(self, c: int) -> None:
        """Initializes ResBlock with specified channel capacity.

        Args:
            c: Number of input and output channels.
        """
        super().__init__()
        self.body = nn.Sequential(
            _gn(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _gn(c),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs residual forward step.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Residual output tensor of shape (B, C, H, W).
        """
        return x + self.body(x)


class VAE(nn.Module):
    """Modernized Autoencoder / VAE with ResBlocks, GroupNorm, and 1x1 Bottleneck."""

    def __init__(
        self,
        in_channels: int = 3,
        image_hw: Tuple[int, int] = (108, 192),
        pad_hw: Tuple[int, int] = (112, 192),
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_channels: int = 32,
        latent_dim: int = 512,
        variational: bool = False,
    ) -> None:
        """Initializes modernized Autoencoder architecture.

        Args:
            in_channels: Input image channel count (default 3 for RGB).
            image_hw: Target unpadded input/output resolution (H, W).
            pad_hw: Evenly divisible padded resolution for 4-stage downsampling.
            channels: Feature channel dimensions across downsampling stages.
            bottleneck_channels: 1x1 conv squeezed channels before flattening.
            latent_dim: Dimensionality of latent space representation (default 512).
            variational: Whether to instantiate variational logvar head.
                Defaults to False.
        """
        super().__init__()
        self.variational: bool = variational
        self.latent_dim: int = latent_dim
        self.image_hw: Tuple[int, int] = image_hw
        self.pad_hw: Tuple[int, int] = pad_hw

        n = len(channels)
        assert pad_hw[0] % (2**n) == 0 and pad_hw[1] % (2**n) == 0, (
            f"pad_hw {pad_hw} must be divisible by 2^{n} = {2**n}"
        )
        self.feat_hw: Tuple[int, int] = (pad_hw[0] // (2**n), pad_hw[1] // (2**n))
        self.top_c: int = channels[-1]
        self.bn_c: int = bottleneck_channels
        flat = bottleneck_channels * self.feat_hw[0] * self.feat_hw[1]

        # Encoder: stem + n x (stride-2 conv + ResBlock) + 1x1 bottleneck squeeze
        enc: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            ResBlock(channels[0]),
        ]
        c = channels[0]
        for c_next in channels:
            enc.extend(
                [
                    nn.Conv2d(c, c_next, kernel_size=3, stride=2, padding=1),
                    ResBlock(c_next),
                ]
            )
            c = c_next
        enc.extend(
            [
                _gn(c),
                nn.SiLU(),
                nn.Conv2d(c, bottleneck_channels, kernel_size=1),
            ]
        )
        self.encoder = nn.Sequential(*enc)

        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim) if variational else None

        # Decoder: FC + 1x1 channel expansion + n x (nearest upsample + conv + ResBlock)
        self.fc_decode = nn.Linear(latent_dim, flat)
        dec: list[nn.Module] = [
            nn.Conv2d(bottleneck_channels, self.top_c, kernel_size=1),
            ResBlock(self.top_c),
        ]
        c = self.top_c
        for c_next in list(reversed(channels[:-1])) + [channels[0]]:
            dec.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(c, c_next, kernel_size=3, padding=1),
                    ResBlock(c_next),
                ]
            )
            c = c_next
        dec.extend(
            [
                _gn(c),
                nn.SiLU(),
                nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
            ]
        )
        self.decoder = nn.Sequential(*dec)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Applies symmetric reflect padding to align input with pad_hw.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Padded tensor of shape (B, C, pad_H, pad_W).
        """
        dh = self.pad_hw[0] - self.image_hw[0]
        if dh > 0:
            return F.pad(x, (0, 0, dh // 2, dh - dh // 2), mode="reflect")
        return x

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """Crops padded output tensor back to target image_hw.

        Args:
            x: Output decoded tensor of shape (B, C, pad_H, pad_W).

        Returns:
            Cropped tensor of shape (B, C, H, W).
        """
        dh = self.pad_hw[0] - self.image_hw[0]
        top = dh // 2
        return x[:, :, top : top + self.image_hw[0], : self.image_hw[1]]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encodes frame into mean and optional log variance vectors.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (mu, logvar). logvar is None when variational is False.
        """
        h = self.encoder(self._pad(x)).flatten(1)
        mu: torch.Tensor = self.fc_mu(h)
        logvar: Optional[torch.Tensor] = (
            self.fc_logvar(h).clamp(-10.0, 10.0)
            if (self.variational and self.fc_logvar is not None)
            else None
        )
        return mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deterministic Latent Vector (mu) for downstream navigation.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Deterministic latent vector mu of shape (B, latent_dim).
        """
        return self.encode(x)[0]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies reparameterization trick to sample latent representation.

        Args:
            mu: Latent mean tensor.
            logvar: Latent log variance tensor.

        Returns:
            Sampled latent tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector into reconstructed frame tensor.

        Args:
            z: Latent Vector tensor of shape (B, latent_dim).

        Returns:
            Reconstructed frame tensor of shape (B, 3, 108, 192) in range [0, 1].
        """
        h = self.fc_decode(z).view(-1, self.bn_c, *self.feat_hw)
        return torch.sigmoid(self._crop(self.decoder(h)))

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Autoencoder or VAE.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).
            deterministic: If True, bypass reparameterization sampling and decode
                directly from mean vector mu. Defaults to False.

        Returns:
            Tuple of (reconstructed_frame, mu, logvar).
        """
        mu, logvar = self.encode(x)
        if self.variational and not deterministic and logvar is not None:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return self.decode(z), mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
    kl_weight: float = 0.0001,
    latent_l2: float = 0.0,
    beta: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates composite Autoencoder/VAE loss (L1 + MSE, optional KL, optional L2).

    Args:
        recon: Reconstructed frame tensor of shape (B, 3, H, W).
        x: Original input frame tensor of shape (B, 3, H, W).
        mu: Latent mean vector of shape (B, latent_dim).
        logvar: Optional latent log variance vector of shape (B, latent_dim).
        kl_weight: Weight factor for dimension-summed KL divergence. Defaults to 1e-4.
        latent_l2: Optional L2 weight penalty on latent vector magnitude.
            Defaults to 0.0.
        beta: Backward-compatible alias for kl_weight.

    Returns:
        Tuple of (total_loss_tensor, metrics_dict).
    """
    if beta is not None:
        kl_weight = beta

    l1 = F.l1_loss(recon, x)
    mse = F.mse_loss(recon, x)
    rec = l1 + mse

    loss = rec
    logs: Dict[str, float] = {
        "loss": rec.item(),
        "recon": rec.item(),
        "recon_l1": l1.item(),
        "recon_mse": mse.item(),
        "kl": 0.0,
        "latent_l2": 0.0,
    }

    if logvar is not None:
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        loss = loss + kl_weight * kl
        logs["kl"] = kl.item()

    if latent_l2 > 0.0:
        reg = mu.pow(2).mean()
        loss = loss + latent_l2 * reg
        logs["latent_l2"] = reg.item()

    logs["loss"] = loss.item()
    return loss, logs

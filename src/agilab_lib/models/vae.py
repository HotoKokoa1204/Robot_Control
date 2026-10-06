"""
Module: vae
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Variational Autoencoder (VAE) for visual representation learning.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates combined VAE loss (Reconstruction MSE + beta * KL divergence).

    Args:
        recon_x: Reconstructed frame tensor of shape (B, 3, H, W).
        x: Original input frame tensor of shape (B, 3, H, W).
        mu: Latent mean vector of shape (B, latent_dim).
        logvar: Latent log variance vector of shape (B, latent_dim).
        beta: Weight factor for KL divergence loss term.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


class VAE(nn.Module):
    """Variational Autoencoder (VAE) for learning structured latent spaces."""

    def __init__(self, latent_dim: int = 128) -> None:
        """Initialize VAE architecture.

        Args:
            latent_dim: Dimensionality of the latent representation.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder CNN: (B, 3, 108, 192) -> (B, 256, 13, 24)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        flat_dim = 256 * 13 * 24
        self.decoder_input_shape = (256, 13, 24)

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1, 0)
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes frame into mean and log variance vectors.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (mu, logvar) tensors of shape (B, latent_dim).
        """
        h = self.encoder_cnn(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu: torch.Tensor = self.fc_mu(h_flat)
        logvar: torch.Tensor = torch.clamp(self.fc_logvar(h_flat), min=-10.0, max=10.0)
        return mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deterministic Latent Vector (mu) for downstream navigation.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Mean vector mu of shape (B, latent_dim).
        """
        mu, _ = self.encode(x)
        return mu

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
            Reconstructed frame tensor of shape (B, 3, 108, 192).
        """
        h_flat = self.fc_decode(z)
        h = h_flat.view(-1, *self.decoder_input_shape)
        recon_x: torch.Tensor = self.decoder_cnn(h)
        return recon_x

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE or deterministic Autoencoder.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).
            deterministic: If True, bypass reparameterization sampling and decode
                directly from mean vector mu (standard Autoencoder mode). Defaults
                to False.

        Returns:
            Tuple of (reconstructed_frame, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

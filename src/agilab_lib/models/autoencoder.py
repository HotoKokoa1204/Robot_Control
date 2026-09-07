"""
Module: autoencoder
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Convolutional Autoencoder and VAE models for frame compression.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Convolutional Autoencoder for compressing video frames into Latent Vectors."""

    def __init__(self, latent_dim: int = 128) -> None:
        """Initialize Autoencoder network.

        Args:
            latent_dim: Dimensionality of the compressed Latent Vector.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B, 3, 108, 192) -> (B, 32, 27, 48)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # (B, 64, 54, 96)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # (B, 32, 27, 48)
        )
        self.flat_dim = 32 * 27 * 48
        self.fc1 = nn.Linear(self.flat_dim, latent_dim)

        # Decoder: (B, latent_dim) -> (B, 3, 108, 192)
        self.fc2 = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_start_shape = (32, 27, 48)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),  # (B, 32, 54, 96)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),  # (B, 64, 108, 192)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # (B, 3, 108, 192)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an input video frame tensor into a Latent Vector.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Latent Vector tensor of shape (B, latent_dim).
        """
        h = self.encoder(x)
        z: torch.Tensor = self.fc1(h.reshape(h.size(0), -1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a Latent Vector back into a reconstructed video frame tensor.

        Args:
            z: Latent Vector tensor of shape (B, latent_dim).

        Returns:
            Reconstructed frame tensor of shape (B, 3, 108, 192).
        """
        h = self.fc2(z).reshape(-1, *self.decoder_start_shape)
        recon: torch.Tensor = self.decoder(h)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Autoencoder.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (reconstructed_frame, latent_vector).
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


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
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input frame tensor of shape (B, 3, 108, 192).

        Returns:
            Tuple of (reconstructed_frame, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

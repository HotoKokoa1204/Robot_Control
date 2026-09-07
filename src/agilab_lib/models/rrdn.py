"""
Module: rrdn
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Residual-in-Residual Dense Network (RRDN) for enhanced decoding.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution followed by LeakyReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize ConvBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Convolution padding.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvBlock.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        return self.lrelu(self.conv(x))


class DenseBlock(nn.Module):
    """Dense block with densely connected convolutional layers."""

    def __init__(
        self,
        in_channels: int = 64,
        growth_rate: int = 32,
        num_layers: int = 4,
    ) -> None:
        """Initialize DenseBlock.

        Args:
            in_channels: Number of input channels.
            growth_rate: Number of channels added per layer.
            num_layers: Number of conv layers in the block.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvBlock(in_channels + i * growth_rate, growth_rate)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DenseBlock.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Concatenated feature tensor.
        """
        features: List[torch.Tensor] = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) combining dense connections with residual scaling."""

    def __init__(
        self,
        in_channels: int = 64,
        growth_rate: int = 32,
        num_layers: int = 4,
        res_scale: float = 0.2,
    ) -> None:
        """Initialize ResidualDenseBlock.

        Args:
            in_channels: Number of input channels.
            growth_rate: Growth rate of inner dense block.
            num_layers: Number of dense layers.
            res_scale: Residual scaling factor.
        """
        super().__init__()
        self.res_scale = res_scale
        self.dense = DenseBlock(in_channels, growth_rate, num_layers)
        dense_out_channels = in_channels + num_layers * growth_rate
        self.conv1x1 = nn.Conv2d(dense_out_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResidualDenseBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Residual scaled tensor of shape (B, C, H, W).
        """
        out = self.conv1x1(self.dense(x))
        return x + out * self.res_scale


class ResidualResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block (RRDB) stacking multiple RDB blocks."""

    def __init__(
        self,
        in_channels: int = 64,
        growth_rate: int = 32,
        num_rdb: int = 3,
        res_scale: float = 0.2,
    ) -> None:
        """Initialize ResidualResidualDenseBlock.

        Args:
            in_channels: Number of feature channels.
            growth_rate: Channel growth rate for inner dense blocks.
            num_rdb: Number of RDB modules.
            res_scale: Residual scaling factor.
        """
        super().__init__()
        self.res_scale = res_scale
        self.rdbs = nn.ModuleList(
            [
                ResidualDenseBlock(in_channels, growth_rate, res_scale=res_scale)
                for _ in range(num_rdb)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResidualResidualDenseBlock.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        out = x
        for rdb in self.rdbs:
            out = rdb(out)
        return x + out * self.res_scale


class RRDN(nn.Module):
    """Residual in Residual Dense Network (RRDN) for enhanced frame upscaling."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 4,
        growth_rate: int = 32,
        upscale_factor: int = 2,
    ) -> None:
        """Initialize RRDN architecture.

        Args:
            in_channels: Number of input frame channels.
            out_channels: Number of output frame channels.
            num_features: Number of intermediate feature channels.
            num_rrdb: Number of RRDB blocks in the trunk.
            growth_rate: Channel growth rate for dense blocks.
            upscale_factor: Upscaling factor (2 or 4).
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.ModuleList(
            [
                ResidualResidualDenseBlock(num_features, growth_rate=growth_rate)
                for _ in range(num_rrdb)
            ]
        )
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        if upscale_factor == 2:
            self.upsample: nn.Module = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif upscale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upsample = nn.Identity()

        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscales and refines input frame tensor.

        Args:
            x: Input frame tensor of shape (B, 3, H, W).

        Returns:
            Enhanced frame tensor of shape (B, 3, H * factor, W * factor).
        """
        fea = self.conv_first(x)
        res = fea
        for block in self.body:
            res = block(res)
        res = self.conv_body(res)
        fea = fea + res
        out = self.upsample(fea)
        out = self.conv_last(out)
        return torch.clamp(out, 0.0, 1.0)


class SimpleDiscriminator(nn.Module):
    """PatchGAN Discriminator for adversarial super-resolution training."""

    def __init__(self, in_channels: int = 3) -> None:
        """Initialize SimpleDiscriminator.

        Args:
            in_channels: Number of input frame channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PatchGAN discriminator.

        Args:
            x: Input frame tensor of shape (B, 3, H, W).

        Returns:
            Patch discrimination logits tensor.
        """
        logits: torch.Tensor = self.net(x)
        return logits


def sobel_edge_detection(img: torch.Tensor) -> torch.Tensor:
    """Calculates gradient edge map for edge consistency loss.

    Args:
        img: Frame tensor of shape (B, 3, H, W).

    Returns:
        Edge magnitude tensor of shape (B, 1, H, W).
    """
    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=img.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=img.device,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge: torch.Tensor = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return edge

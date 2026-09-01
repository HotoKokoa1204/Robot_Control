"""
Module: pca
Stage: Library
Author: AGILAB NTNU
Date: 2026-09-01
Description: Batched Incremental PCA using PyTorch for latent space reduction.
"""

from typing import Optional

import torch


class BatchedPCA:
    """Batched Incremental PCA using PyTorch for latent space reduction."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.mean: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "BatchedPCA":
        """Fits PCA model on latent matrix x: (N, D).

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            self
        """
        self.mean = torch.mean(x, dim=0, keepdim=True)
        x_centered = x - self.mean
        # SVD on centered data
        _u, _s, v = torch.pca_lowrank(x_centered, q=self.n_components, center=False)
        self.components_ = v[:, : self.n_components]  # (D, n_components)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Projects latent matrix x: (N, D) -> (N, n_components).

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Projected tensor of shape (N, n_components).
        """
        if self.mean is None or self.components_ is None:
            raise RuntimeError("BatchedPCA must be fitted before calling transform.")
        x_centered = x - self.mean
        return torch.mm(x_centered, self.components_)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fits PCA model and transforms input tensor in one step.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            Projected tensor of shape (N, n_components).
        """
        self.fit(x)
        return self.transform(x)

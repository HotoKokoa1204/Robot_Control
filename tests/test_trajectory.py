"""Unit tests for trajectory and latent operations in agilab_lib.utils.trajectory."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from agilab_lib.models.vae import VAE
from agilab_lib.utils.trajectory import (
    BatchedPCA,
    extract_keyframe_indices,
    interpolate_all_frames_numpy,
    linear_interpolate_latent_sequence,
)


def test_linear_interpolate_latent_sequence_tensor_shape() -> None:
    """Test interpolation sequence tensor output shape and boundary values."""
    latents = torch.randn(5, 128)
    interp_frames = 2
    out = linear_interpolate_latent_sequence(latents, interp_frames=interp_frames)
    expected_len = (5 - 1) * (interp_frames + 1) + 1
    assert out.shape == (expected_len, 128)
    assert torch.allclose(out[0], latents[0])
    assert torch.allclose(out[-1], latents[-1])


def test_linear_interpolate_latent_sequence_edge_cases() -> None:
    """Test edge cases with zero interpolation frames and single latent."""
    latents = torch.randn(5, 128)
    out_zero = linear_interpolate_latent_sequence(latents, interp_frames=0)
    assert out_zero.shape == (5, 128)

    single_latent = torch.randn(1, 128)
    out_single = linear_interpolate_latent_sequence(single_latent, interp_frames=3)
    assert out_single.shape == (1, 128)


def test_interpolate_all_frames_numpy_shape() -> None:
    """Test NumPy interpolation output shape and endpoints."""
    latents = np.random.randn(4, 64).astype(np.float32)
    interp_steps = 3
    out = interpolate_all_frames_numpy(latents, interpolation_steps=interp_steps)
    expected_len = 4 + (4 - 1) * interp_steps
    assert out.shape == (expected_len, 64)
    assert np.allclose(out[0], latents[0])
    assert np.allclose(out[-1], latents[-1])


def test_interpolate_all_frames_numpy_edge_cases() -> None:
    """Test NumPy interpolation with zero steps and single item."""
    latents = np.random.randn(4, 64).astype(np.float32)
    out_zero = interpolate_all_frames_numpy(latents, interpolation_steps=0)
    assert out_zero.shape == (4, 64)

    single_latent = np.random.randn(1, 64).astype(np.float32)
    out_single = interpolate_all_frames_numpy(single_latent, interpolation_steps=3)
    assert out_single.shape == (1, 64)


def test_batched_pca_fit_transform() -> None:
    """Test BatchedPCA fit_transform and subsequent transform."""
    x = torch.randn(50, 128)
    pca = BatchedPCA(n_components=2)
    out = pca.fit_transform(x)
    assert out.shape == (50, 2)
    assert pca.mean is not None
    assert pca.components_ is not None
    assert pca.components_.shape == (128, 2)

    new_x = torch.randn(10, 128)
    new_out = pca.transform(new_x)
    assert new_out.shape == (10, 2)


def test_batched_pca_unfitted_raises_error() -> None:
    """Test that BatchedPCA raises error when transform is called before fit."""
    pca = BatchedPCA(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        pca.transform(torch.randn(10, 128))


def test_extract_keyframe_indices_empty_frames() -> None:
    """Test keyframe extraction with an empty frames list returns empty list."""
    vae = VAE(latent_dim=128)
    device = torch.device("cpu")
    indices = extract_keyframe_indices(vae, [], tau=1.0, device=device)
    assert indices == []


def test_extract_keyframe_indices_threshold() -> None:
    """Test keyframe extraction triggers only when latent distance exceeds tau."""

    class MockVAE(nn.Module):
        """Mock VAE whose get_latent returns pre-programmed latents."""

        def __init__(self) -> None:
            super().__init__()
            self.step = 0

        def get_latent(self, x: torch.Tensor) -> torch.Tensor:
            # Step 0: [0.0]
            # Step 1: [0.75] (dist 0.75 < 1.0)
            # Step 2: [1.50] (dist 1.50 >= 1.0)
            val = float(self.step) * 0.75
            self.step += 1
            return torch.tensor([[val]], dtype=torch.float32)

    mock_vae = MockVAE()
    frames = [torch.zeros(3, 10, 10) for _ in range(3)]
    # Distances from last keyframe (idx 0, latent 0.0):
    # frame 1: latent 0.75, dist = 0.75 (< 1.0) -> not keyframe
    # frame 2: latent 1.50, dist = 1.50 (>= 1.0) -> keyframe, last becomes 1.50
    indices = extract_keyframe_indices(
        mock_vae,  # type: ignore[arg-type]
        frames,
        tau=1.0,
        device=torch.device("cpu"),
    )
    assert indices == [0, 2]

"""
Module: test_vae
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Variational Autoencoder (VAE) and video datasets.
"""

import torch
from agilab_lib.datasets.video_dataset import DummyVideoDataset
from agilab_lib.models.vae import VAE, vae_loss
from agilab_lib.utils.keyframes import extract_keyframe_indices


def test_vae_forward_and_reparameterize() -> None:
    """Test VAE forward pass and latent dimension shapes."""
    vae = VAE(latent_dim=128)
    x = torch.rand(2, 3, 108, 192)
    recon, mu, logvar = vae(x)
    assert recon.shape == (2, 3, 108, 192)
    assert mu.shape == (2, 128)
    assert logvar.shape == (2, 128)


def test_vae_encode_get_latent_and_decode() -> None:
    """Test VAE encoding, deterministic latent extraction, and decoding."""
    vae = VAE(latent_dim=64)
    x = torch.rand(4, 3, 108, 192)
    mu, logvar = vae.encode(x)
    z = vae.get_latent(x)
    assert torch.allclose(mu, z)

    recon = vae.decode(z)
    assert recon.shape == (4, 3, 108, 192)
    assert recon.min().item() >= 0.0
    assert recon.max().item() <= 1.0


def test_vae_loss_computation() -> None:
    """Test VAE combined loss calculation."""
    recon = torch.rand(2, 3, 108, 192)
    x = torch.rand(2, 3, 108, 192)
    mu = torch.randn(2, 128)
    logvar = torch.randn(2, 128)

    total_loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=0.001)
    assert total_loss.item() > 0.0
    assert recon_loss.item() > 0.0
    assert not torch.isnan(kl_loss)


def test_dummy_video_dataset_shapes() -> None:
    """Test DummyVideoDataset length and generated frame tensor shapes."""
    dataset = DummyVideoDataset(num_frames=12, img_height=108, img_width=192)
    assert len(dataset) == 12
    frame = dataset[0]
    assert frame.shape == (3, 108, 192)
    assert frame.dtype == torch.float32


def test_extract_keyframe_indices_logic() -> None:
    """Test Keyframe extraction logic using VAE representations."""
    vae = VAE(latent_dim=128)
    device = torch.device("cpu")
    frames = [torch.rand(3, 108, 192) for _ in range(10)]

    keyframes_dense = extract_keyframe_indices(
        vae=vae, frames=frames, tau=0.01, device=device
    )
    assert 0 in keyframes_dense
    assert len(keyframes_dense) >= 1
    assert keyframes_dense == sorted(keyframes_dense)

    assert extract_keyframe_indices(vae, [], tau=1.0, device=device) == []

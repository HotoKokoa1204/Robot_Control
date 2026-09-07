"""Unit tests for Autoencoder, VAE, and video datasets."""

import torch
from agilab_lib.datasets.video_dataset import DummyVideoDataset
from agilab_lib.models.autoencoder import VAE, Autoencoder
from agilab_lib.utils.keyframes import extract_keyframe_indices


def test_autoencoder_encode_decode_shapes() -> None:
    """Test Autoencoder encoding and decoding tensor dimensions."""
    ae = Autoencoder(latent_dim=128)
    x = torch.rand(4, 3, 108, 192)
    z = ae.encode(x)
    assert z.shape == (4, 128)

    recon = ae.decode(z)
    assert recon.shape == (4, 3, 108, 192)
    assert recon.min().item() >= 0.0
    assert recon.max().item() <= 1.0


def test_autoencoder_forward_pass() -> None:
    """Test full forward pass of Autoencoder."""
    ae = Autoencoder(latent_dim=64)
    x = torch.rand(2, 3, 108, 192)
    recon, z = ae(x)
    assert recon.shape == (2, 3, 108, 192)
    assert z.shape == (2, 64)


def test_vae_forward_and_reparameterize() -> None:
    """Test VAE forward pass and latent dimension shapes."""
    vae = VAE(latent_dim=128)
    x = torch.rand(2, 3, 108, 192)
    recon, mu, logvar = vae(x)
    assert recon.shape == (2, 3, 108, 192)
    assert mu.shape == (2, 128)
    assert logvar.shape == (2, 128)


def test_dummy_video_dataset_shapes() -> None:
    """Test DummyVideoDataset length and generated frame tensor shapes."""
    dataset = DummyVideoDataset(num_frames=12, img_height=108, img_width=192)
    assert len(dataset) == 12
    frame = dataset[0]
    assert frame.shape == (3, 108, 192)
    assert frame.dtype == torch.float32


def test_extract_keyframe_indices_logic() -> None:
    """Test Keyframe extraction logic on mock frames."""
    ae = Autoencoder(latent_dim=128)
    device = torch.device("cpu")
    frames = [torch.rand(3, 108, 192) for _ in range(10)]

    # Very small tau should select multiple keyframes
    keyframes_dense = extract_keyframe_indices(
        autoencoder=ae, frames=frames, tau=0.01, device=device
    )
    assert 0 in keyframes_dense
    assert len(keyframes_dense) >= 1
    assert keyframes_dense == sorted(keyframes_dense)

    # Empty frames should return empty list
    assert extract_keyframe_indices(ae, [], tau=1.0, device=device) == []

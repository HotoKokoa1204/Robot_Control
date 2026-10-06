"""Unit tests for RRDN, discriminator, and super-resolution datasets."""

import torch
from agilab_lib.datasets.sr_dataset import DummyPairedFrameDataset
from agilab_lib.models.rrdn import (
    RRDN,
    SimpleDiscriminator,
    sobel_edge_detection,
)


def test_rrdn_forward_upscale_shapes() -> None:
    """Test RRDN forward pass output tensor dimensions with x2 and x4 upscaling."""
    x = torch.rand(2, 3, 27, 48)

    # Upscale x2
    rrdn_x2 = RRDN(num_features=16, num_rrdb=2, growth_rate=16, upscale_factor=2)
    out_x2 = rrdn_x2(x)
    assert out_x2.shape == (2, 3, 54, 96)
    assert out_x2.min().item() >= 0.0
    assert out_x2.max().item() <= 1.0

    # Upscale x4
    rrdn_x4 = RRDN(num_features=16, num_rrdb=2, growth_rate=16, upscale_factor=4)
    out_x4 = rrdn_x4(x)
    assert out_x4.shape == (2, 3, 108, 192)


def test_simple_discriminator_shape() -> None:
    """Test PatchGAN discriminator output shape."""
    disc = SimpleDiscriminator(in_channels=3)
    x = torch.rand(2, 3, 64, 64)
    logits = disc(x)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1


def test_sobel_edge_detection_shape() -> None:
    """Test Sobel edge filter output shape."""
    x = torch.rand(2, 3, 54, 96)
    edge = sobel_edge_detection(x)
    assert edge.shape == (2, 1, 54, 96)


def test_dummy_paired_frame_dataset_shapes() -> None:
    """Test DummyPairedFrameDataset length and output tensor dimensions."""
    dataset = DummyPairedFrameDataset(
        num_samples=6, lr_shape=(3, 27, 48), hr_shape=(3, 54, 96)
    )
    assert len(dataset) == 6
    lr, hr = dataset[0]
    assert lr.shape == (3, 27, 48)
    assert hr.shape == (3, 54, 96)

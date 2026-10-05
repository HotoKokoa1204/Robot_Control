"""
Module: test_perceptual_loss
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Unit tests for frozen VGG-16 Perceptual Loss module.
"""

from PIL import Image  # isort: skip # noqa: F401
import pytest
import torch
from agilab_lib.models.perceptual import VGGPerceptualLoss


def test_perceptual_loss_identical_frames() -> None:
    """Test that identical input frames yield exactly 0.0 perceptual loss."""
    loss_fn_l1 = VGGPerceptualLoss(loss_type="l1")
    loss_fn_mse = VGGPerceptualLoss(loss_type="mse")

    x = torch.rand(2, 3, 108, 192)
    x_clone = x.clone()

    loss_l1 = loss_fn_l1(x, x_clone)
    loss_mse = loss_fn_mse(x, x_clone)

    assert torch.isclose(loss_l1, torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(loss_mse, torch.tensor(0.0), atol=1e-7)


def test_perceptual_loss_different_frames() -> None:
    """Test that different input frames yield strictly positive perceptual loss."""
    loss_fn = VGGPerceptualLoss(loss_type="l1")

    frame_a = torch.zeros(2, 3, 108, 192)
    frame_b = torch.ones(2, 3, 108, 192)

    loss = loss_fn(frame_a, frame_b)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_perceptual_loss_parameters_frozen() -> None:
    """Test that all model parameters have requires_grad == False and eval mode."""
    loss_fn = VGGPerceptualLoss()
    params = list(loss_fn.parameters())

    assert len(params) > 0
    assert all(not p.requires_grad for p in params)
    assert loss_fn.training is False

    # Calling train() should not switch module to training mode
    loss_fn.train()
    assert loss_fn.training is False


def test_perceptual_loss_device_movement() -> None:
    """Test module device movement across CPU and CUDA (when available)."""
    loss_fn = VGGPerceptualLoss()
    cpu_device = torch.device("cpu")

    loss_fn.to(cpu_device)
    x_cpu = torch.rand(2, 3, 108, 192, device=cpu_device)
    loss_cpu = loss_fn(x_cpu, x_cpu)
    assert loss_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        loss_fn.to(cuda_device)
        x_cuda_a = torch.rand(2, 3, 108, 192, device=cuda_device)
        x_cuda_b = torch.rand(2, 3, 108, 192, device=cuda_device)

        loss_cuda = loss_fn(x_cuda_a, x_cuda_b)
        assert loss_cuda.device.type == "cuda"
        assert loss_cuda.item() > 0.0


def test_perceptual_loss_gradient_flow() -> None:
    """Test gradient flow propagates to prediction tensor but not to backbone."""
    loss_fn = VGGPerceptualLoss()
    pred = torch.rand(2, 3, 108, 192, requires_grad=True)
    target = torch.rand(2, 3, 108, 192)

    loss = loss_fn(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()

    # Backbone parameters must not receive any gradients
    assert all(p.grad is None for p in loss_fn.parameters())


def test_perceptual_loss_custom_weights() -> None:
    """Test configurable layer weights and validation."""
    custom_weights = [0.1, 0.2, 0.3, 0.4]
    loss_fn = VGGPerceptualLoss(weights=custom_weights)
    assert loss_fn.weights == custom_weights

    # Invalid weights length must raise ValueError
    with pytest.raises(ValueError, match="Expected 4 weights"):
        VGGPerceptualLoss(weights=[0.5, 0.5])


def test_perceptual_loss_extract_features_and_layer_losses() -> None:
    """Test multi-scale feature extraction and per-layer loss dict."""
    loss_fn = VGGPerceptualLoss()
    x = torch.rand(2, 3, 108, 192)

    features = loss_fn.extract_features(x)
    assert len(features) == 4
    expected_channels = [64, 128, 256, 512]
    for feat, ch in zip(features, expected_channels):
        assert feat.size(1) == ch

    target = torch.rand(2, 3, 108, 192)
    layer_losses = loss_fn.get_layer_losses(x, target)
    assert set(layer_losses.keys()) == set(VGGPerceptualLoss.LAYER_NAMES)
    for l_loss in layer_losses.values():
        assert l_loss.item() > 0.0


def test_perceptual_loss_input_validation() -> None:
    """Test input tensor shape mismatch and dimension validation."""
    loss_fn = VGGPerceptualLoss()

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(torch.rand(2, 3, 108, 192), torch.rand(3, 3, 108, 192))

    # Incorrect dimension / channels
    with pytest.raises(ValueError, match="Expected input tensors of shape"):
        loss_fn(torch.rand(2, 1, 108, 192), torch.rand(2, 1, 108, 192))

    # Unsupported loss type
    with pytest.raises(ValueError, match="Unsupported loss_type"):
        VGGPerceptualLoss(loss_type="invalid_loss")


def test_perceptual_loss_uninitialized_fallback() -> None:
    """Test initialization with pretrained=False works gracefully."""
    loss_fn = VGGPerceptualLoss(pretrained=False)
    assert all(not p.requires_grad for p in loss_fn.parameters())

    x = torch.rand(1, 3, 64, 64)
    loss = loss_fn(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

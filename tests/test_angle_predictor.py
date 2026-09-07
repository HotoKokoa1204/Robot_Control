"""
Module: test_angle_predictor
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Angle Predictor model and training components.
"""

import pytest
import torch
import torch.nn.functional as F
from agilab_lib.models.angle_predictor import AnglePredictor
from agilab_lib.utils.eval_metrics import evaluate_angle_prediction_mae


def test_angle_predictor_forward_shapes() -> None:
    """Test AnglePredictor output shape with standard batch."""
    batch_size = 4
    latent_dim = 128
    predictor = AnglePredictor(latent_dim=latent_dim, hidden_dims=[64, 32])

    z1 = torch.randn(batch_size, latent_dim)
    z2 = torch.randn(batch_size, latent_dim)

    out = predictor(z1, z2)
    assert out.shape == (batch_size, 1)


def test_angle_predictor_eval_single_sample() -> None:
    """Test AnglePredictor in eval mode with a single sample batch."""
    latent_dim = 128
    predictor = AnglePredictor(
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        use_batch_norm=True,
    )
    predictor.eval()

    z1 = torch.randn(1, latent_dim)
    z2 = torch.randn(1, latent_dim)

    with torch.no_grad():
        out = predictor(z1, z2)
    assert out.shape == (1, 1)


def test_angle_predictor_without_batch_norm() -> None:
    """Test AnglePredictor without batch normalization."""
    latent_dim = 64
    predictor = AnglePredictor(
        latent_dim=latent_dim,
        hidden_dims=[32],
        use_batch_norm=False,
    )
    z1 = torch.randn(2, latent_dim)
    z2 = torch.randn(2, latent_dim)

    out = predictor(z1, z2)
    assert out.shape == (2, 1)


def test_angle_predictor_dimension_mismatch_raises_error() -> None:
    """Test that invalid input dimensions raise ValueError."""
    predictor = AnglePredictor(latent_dim=128)

    # Wrong latent_dim
    with pytest.raises(ValueError, match="latent1 dimension mismatch"):
        predictor(torch.randn(2, 64), torch.randn(2, 128))

    with pytest.raises(ValueError, match="latent2 dimension mismatch"):
        predictor(torch.randn(2, 128), torch.randn(2, 64))

    # Mismatched batch size
    with pytest.raises(ValueError, match="Batch sizes do not match"):
        predictor(torch.randn(2, 128), torch.randn(3, 128))


def test_angle_predictor_single_train_step() -> None:
    """Test single training step with gradient propagation."""
    latent_dim = 64
    predictor = AnglePredictor(
        latent_dim=latent_dim,
        hidden_dims=[32],
        use_batch_norm=False,
    )
    predictor.train()

    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)
    z1 = torch.randn(4, latent_dim)
    z2 = torch.randn(4, latent_dim)
    target_angle = torch.tensor([[10.0], [-20.0], [5.0], [0.0]])

    optimizer.zero_grad()
    pred_angle = predictor(z1, z2)
    loss = F.l1_loss(pred_angle, target_angle)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_evaluate_angle_prediction_mae_integration() -> None:
    """Test evaluating MAE using model predictions."""
    predictor = AnglePredictor(latent_dim=128, hidden_dims=[32])
    predictor.eval()

    z1 = torch.randn(5, 128)
    z2 = torch.randn(5, 128)
    target = torch.randn(5, 1)

    with torch.no_grad():
        preds = predictor(z1, z2)

    mae = evaluate_angle_prediction_mae(preds, target)
    assert isinstance(mae, float)
    assert mae >= 0.0

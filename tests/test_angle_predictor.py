"""Module: test_angle_predictor
Stage: Library
Author: KafuuChino
Date: 2026-09-07
Description: Unit tests for Angle Predictor model and training components.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from agilab_lib.models.angle_predictor import AnglePredictor
from agilab_lib.models.rlt import ResidualLatentTransformer
from agilab_lib.utils.eval_metrics import evaluate_angle_prediction_mae


def test_angle_predictor_forward_shapes_and_unit_norm() -> None:
    """Test AnglePredictor output shape (B, 2), unit norm, and [-1, 1] range."""
    batch_size = 8
    latent_dim = 128
    predictor = AnglePredictor(latent_dim=latent_dim, hidden_dims=[64, 32])

    z1 = torch.randn(batch_size, latent_dim)
    z2 = torch.randn(batch_size, latent_dim)

    out = predictor(z1, z2)
    assert out.shape == (batch_size, 2)

    # Values must be strictly bounded in [-1.0, 1.0]
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)

    # Norm along dim=-1 must be 1.0 (unit circle constraint)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


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
    assert out.shape == (1, 2)
    norm = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)


def test_angle_predictor_predict_angle_deg() -> None:
    """Test predict_angle_deg output shape (B, 1) and value bounds [-180, 180]."""
    latent_dim = 64
    predictor = AnglePredictor(latent_dim=latent_dim, hidden_dims=[32])
    predictor.eval()

    z1 = torch.randn(10, latent_dim)
    z2 = torch.randn(10, latent_dim)

    with torch.no_grad():
        deg = predictor.predict_angle_deg(z1, z2)

    assert deg.shape == (10, 1)
    assert torch.all(deg >= -180.0)
    assert torch.all(deg <= 180.0)


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
    assert out.shape == (2, 2)


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
    """Test single training step with gradient propagation on sin/cos targets."""
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

    target_deg = torch.tensor([10.0, -20.0, 90.0, -180.0])
    target_rad = target_deg * math.pi / 180.0
    target_sin_cos = torch.stack([torch.sin(target_rad), torch.cos(target_rad)], dim=-1)

    optimizer.zero_grad()
    pred_sin_cos = predictor(z1, z2)
    loss = F.mse_loss(pred_sin_cos, target_sin_cos)
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

    # Targets as [sin, cos]
    angles_deg = torch.tensor([0.0, 30.0, -45.0, 90.0, -120.0])
    rad = angles_deg * math.pi / 180.0
    target_sin_cos = torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)

    with torch.no_grad():
        preds = predictor(z1, z2)

    mae = evaluate_angle_prediction_mae(preds, target_sin_cos)
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_angle_predictor_self_supervised_step() -> None:
    """Test self-supervised training step with frozen RLT teacher."""
    latent_dim = 64
    teacher_rlt = ResidualLatentTransformer(
        latent_dim=latent_dim, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )
    for blk in teacher_rlt.blocks:
        torch.nn.init.normal_(blk.fc3.weight, mean=0.0, std=0.2)
        torch.nn.init.normal_(blk.fc3.bias, mean=0.0, std=0.1)
    teacher_rlt.eval()
    for p in teacher_rlt.parameters():
        p.requires_grad = False

    student_predictor = AnglePredictor(
        latent_dim=latent_dim, hidden_dims=[32], use_batch_norm=False
    )
    student_predictor.train()

    optimizer = torch.optim.Adam(student_predictor.parameters(), lr=0.01)

    z1 = torch.randn(4, latent_dim)
    z2 = torch.randn(4, latent_dim)

    optimizer.zero_grad()
    pred_sin_cos = student_predictor(z1, z2)
    pred_z2 = teacher_rlt(z1, sin_cos=pred_sin_cos)
    loss = F.mse_loss(pred_z2, z2)
    loss.backward()

    # Verify teacher parameters have no gradients
    for p in teacher_rlt.parameters():
        assert p.grad is None

    # Verify student parameters have computed gradients
    has_grad = False
    for p in student_predictor.parameters():
        if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            has_grad = True
            break
    assert has_grad, "Student AnglePredictor should receive backpropagated gradients"

    optimizer.step()
    assert torch.isfinite(loss)

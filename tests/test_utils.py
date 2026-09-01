import numpy as np
import pytest
import torch
import torch.nn as nn

from agilab_lib.utils.eval_metrics import (
    decode_latents_to_frames,
    evaluate_angle_prediction_mae,
    evaluate_latent_prediction_mse,
)
from agilab_lib.utils.interpolation import (
    interpolate_all_frames_numpy,
    linear_interpolate_latent_sequence,
)
from agilab_lib.utils.pca import BatchedPCA


def test_linear_interpolate_latent_sequence_tensor_shape():
    latents = torch.randn(5, 128)
    interp_frames = 2
    out = linear_interpolate_latent_sequence(latents, interp_frames=interp_frames)
    expected_len = (5 - 1) * (interp_frames + 1) + 1
    assert out.shape == (expected_len, 128)
    assert torch.allclose(out[0], latents[0])
    assert torch.allclose(out[-1], latents[-1])


def test_linear_interpolate_latent_sequence_edge_cases():
    latents = torch.randn(5, 128)
    out_zero = linear_interpolate_latent_sequence(latents, interp_frames=0)
    assert out_zero.shape == (5, 128)

    single_latent = torch.randn(1, 128)
    out_single = linear_interpolate_latent_sequence(single_latent, interp_frames=3)
    assert out_single.shape == (1, 128)


def test_interpolate_all_frames_numpy_shape():
    latents = np.random.randn(4, 64).astype(np.float32)
    interp_steps = 3
    out = interpolate_all_frames_numpy(latents, interpolation_steps=interp_steps)
    expected_len = 4 + (4 - 1) * interp_steps
    assert out.shape == (expected_len, 64)
    assert np.allclose(out[0], latents[0])
    assert np.allclose(out[-1], latents[-1])


def test_interpolate_all_frames_numpy_edge_cases():
    latents = np.random.randn(4, 64).astype(np.float32)
    out_zero = interpolate_all_frames_numpy(latents, interpolation_steps=0)
    assert out_zero.shape == (4, 64)

    single_latent = np.random.randn(1, 64).astype(np.float32)
    out_single = interpolate_all_frames_numpy(single_latent, interpolation_steps=3)
    assert out_single.shape == (1, 64)


def test_evaluate_latent_prediction_mse():
    pred = torch.randn(8, 128)
    gt = torch.randn(8, 128)
    loss = evaluate_latent_prediction_mse(pred, gt)
    assert isinstance(loss, float)
    assert loss >= 0.0

    perfect_loss = evaluate_latent_prediction_mse(pred, pred)
    assert pytest.approx(perfect_loss, abs=1e-6) == 0.0


def test_evaluate_angle_prediction_mae():
    pred = torch.tensor([[10.0], [-15.0], [30.0]])
    gt = torch.tensor([[12.0], [-10.0], [25.0]])
    loss = evaluate_angle_prediction_mae(pred, gt)
    assert isinstance(loss, float)
    expected = (2.0 + 5.0 + 5.0) / 3.0
    assert pytest.approx(loss, abs=1e-5) == expected


def test_decode_latents_to_frames():
    class DummyAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(128, 3 * 108 * 192)

        def decode(self, z):
            return self.decoder(z).view(-1, 3, 108, 192)

    ae = DummyAutoencoder()
    ae.train()
    latents = torch.randn(4, 128)
    frames = decode_latents_to_frames(ae, latents)

    assert not ae.training
    assert frames.shape == (4, 3, 108, 192)


def test_batched_pca_fit_transform():
    x = torch.randn(50, 128)
    pca = BatchedPCA(n_components=2)
    out = pca.fit_transform(x)
    assert out.shape == (50, 2)
    assert pca.mean is not None
    assert pca.components_ is not None
    assert pca.components_.shape == (128, 2)

    # Transform new data
    new_x = torch.randn(10, 128)
    new_out = pca.transform(new_x)
    assert new_out.shape == (10, 2)


def test_batched_pca_unfitted_raises_error():
    pca = BatchedPCA(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        pca.transform(torch.randn(10, 128))

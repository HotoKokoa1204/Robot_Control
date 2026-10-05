"""
Module: test_joint_navigation_model
Stage: Library
Author: KafuuChino
Date: 2026-09-29
Description: Unit tests for JointNavigationModel composite multi-branch architecture.
"""

from pathlib import Path
from typing import Dict

import pytest
import torch

from agilab_lib.models import (
    VAE,
    ForwardLatentTransformer,
    JointNavigationModel,
    RotationLatentTransformer,
)


def test_joint_model_initialization() -> None:
    """Test default architecture parameters and sub-network configurations."""
    model = JointNavigationModel()
    assert model.latent_dim == 512
    assert model.hidden_dim == 512
    assert model.num_blocks == 5
    assert model.block_inner_dim == 512

    # Check VAE dimensions
    assert isinstance(model.vae, VAE)
    assert model.vae.latent_dim == 512

    # Check Forward transformer dimensions and block count (5 blocks per spec)
    assert isinstance(model.forward_transformer, ForwardLatentTransformer)
    assert model.forward_transformer.latent_dim == 512
    assert model.forward_transformer.hidden_dim == 512
    assert len(model.forward_transformer.blocks) == 5
    assert model.forward_transformer.cond_dim == 1

    # Check Rotation transformer dimensions and block count (5 blocks per spec)
    assert isinstance(model.rotation_transformer, RotationLatentTransformer)
    assert model.rotation_transformer.latent_dim == 512
    assert model.rotation_transformer.hidden_dim == 512
    assert len(model.rotation_transformer.blocks) == 5
    assert model.rotation_transformer.cond_dim == 2


def test_joint_model_custom_initialization() -> None:
    """Test custom dimension configurations and module injection."""
    custom_vae = VAE(latent_dim=256)
    custom_fwd = ForwardLatentTransformer(
        latent_dim=256, hidden_dim=128, num_blocks=3, block_inner_dim=128
    )
    custom_rot = RotationLatentTransformer(
        latent_dim=256, hidden_dim=128, num_blocks=3, block_inner_dim=128
    )

    model = JointNavigationModel(
        latent_dim=256,
        hidden_dim=128,
        num_blocks=3,
        block_inner_dim=128,
        vae=custom_vae,
        forward_transformer=custom_fwd,
        rotation_transformer=custom_rot,
    )

    assert model.latent_dim == 256
    assert model.vae is custom_vae
    assert model.forward_transformer is custom_fwd
    assert model.rotation_transformer is custom_rot
    assert len(model.forward_transformer.blocks) == 3
    assert len(model.rotation_transformer.blocks) == 3


def test_forward_reconstruction_shapes_and_values() -> None:
    """Test Branch 1: frame reconstruction and latent distribution outputs."""
    batch_size = 2
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )
    x = torch.rand(batch_size, 3, 108, 192)

    recon_x, mu, logvar = model.forward_reconstruction(x)

    assert recon_x.shape == (batch_size, 3, 108, 192)
    assert mu.shape == (batch_size, 128)
    assert logvar.shape == (batch_size, 128)

    # Sigmoid output in [0, 1]
    assert torch.all(recon_x >= 0.0)
    assert torch.all(recon_x <= 1.0)
    assert torch.all(torch.isfinite(recon_x))
    assert torch.all(torch.isfinite(mu))
    assert torch.all(torch.isfinite(logvar))


def test_forward_translation_shapes_and_modes() -> None:
    """Test Branch 2: forward translation dynamics with varying distance formats."""
    batch_size = 4
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )
    x = torch.rand(batch_size, 3, 108, 192)

    # 1. 2D distance tensor (B, 1)
    dist_2d = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
    pred_img, pred_latent, mu_t = model.forward_translation(x, distance_meters=dist_2d)
    assert pred_img.shape == (batch_size, 3, 108, 192)
    assert pred_latent.shape == (batch_size, 128)
    assert mu_t.shape == (batch_size, 128)
    assert torch.all(torch.isfinite(pred_img))

    # 2. 1D distance tensor (B,)
    dist_1d = torch.tensor([0.2, 0.4, 0.6, 0.8])
    pred_img_1d, pred_latent_1d, _ = model.forward_translation(
        x, distance_meters=dist_1d
    )
    assert pred_img_1d.shape == (batch_size, 3, 108, 192)
    assert pred_latent_1d.shape == (batch_size, 128)

    # 3. Scalar float distance
    pred_img_sc, pred_latent_sc, _ = model.forward_translation(x, distance_meters=1.5)
    assert pred_img_sc.shape == (batch_size, 3, 108, 192)
    assert pred_latent_sc.shape == (batch_size, 128)

    # 4. Using pre-computed latent
    mu = model.get_latent(x)
    pred_img_lat, pred_lat_out, mu_out = model.forward_translation(
        distance_meters=1.0, latent=mu
    )
    assert pred_img_lat.shape == (batch_size, 3, 108, 192)
    assert torch.equal(mu, mu_out)

    # 5. Missing both x_t and latent raises ValueError
    with pytest.raises(ValueError, match="Either 'x_t' or 'latent'"):
        model.forward_translation(distance_meters=1.0)


def test_forward_rotation_shapes_and_modes() -> None:
    """Test Branch 3: rotation dynamics with sin_cos unit vectors and angle formats."""
    batch_size = 4
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )
    x = torch.rand(batch_size, 3, 108, 192)

    # 1. 2D sin_cos unit vector tensor (B, 2)
    sin_cos = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071], [-1.0, 0.0]])
    pred_img, pred_latent, mu_t = model.forward_rotation(x, sin_cos=sin_cos)
    assert pred_img.shape == (batch_size, 3, 108, 192)
    assert pred_latent.shape == (batch_size, 128)
    assert mu_t.shape == (batch_size, 128)
    assert torch.all(torch.isfinite(pred_img))

    # 2. Scalar float angle in degrees
    pred_img_deg, pred_latent_deg, _ = model.forward_rotation(x, angle_deg=45.0)
    assert pred_img_deg.shape == (batch_size, 3, 108, 192)
    assert pred_latent_deg.shape == (batch_size, 128)

    # 3. 1D tensor angle in degrees (B,)
    angle_1d = torch.tensor([30.0, -45.0, 90.0, -90.0])
    pred_img_1d, pred_latent_1d, _ = model.forward_rotation(x, angle_deg=angle_1d)
    assert pred_img_1d.shape == (batch_size, 3, 108, 192)
    assert pred_latent_1d.shape == (batch_size, 128)

    # 4. Using pre-computed latent
    mu = model.get_latent(x)
    pred_img_lat, pred_lat_out, mu_out = model.forward_rotation(
        sin_cos=sin_cos, latent=mu
    )
    assert pred_img_lat.shape == (batch_size, 3, 108, 192)
    assert torch.equal(mu, mu_out)

    # 5. Missing both x_t and latent raises ValueError
    with pytest.raises(ValueError, match="Either 'x_t' or 'latent'"):
        model.forward_rotation(sin_cos=sin_cos)


def test_unified_forward_dispatch() -> None:
    """Test unified forward method with explicit modes and automatic dispatch."""
    batch_size = 2
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )
    x = torch.rand(batch_size, 3, 108, 192)
    dist = torch.tensor([[1.0], [2.0]])
    sin_cos = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    # 1. Default / explicit reconstruction
    res_default = model(x)
    assert len(res_default) == 3
    assert res_default[0].shape == (batch_size, 3, 108, 192)

    res_recon = model(x, mode="reconstruction")
    assert len(res_recon) == 3

    # 2. Explicit translation mode
    res_trans = model(x, mode="translation", distance_meters=dist)
    assert len(res_trans) == 3
    assert res_trans[0].shape == (batch_size, 3, 108, 192)

    # 3. Explicit rotation mode
    res_rot = model(x, mode="rotation", sin_cos=sin_cos)
    assert len(res_rot) == 3
    assert res_rot[0].shape == (batch_size, 3, 108, 192)

    # 4. Auto-dispatch based on arguments
    res_auto_fwd = model(x, distance_meters=dist)
    assert res_auto_fwd[0].shape == (batch_size, 3, 108, 192)
    assert res_auto_fwd[1].shape == (batch_size, 128)

    res_auto_rot = model(x, sin_cos=sin_cos)
    assert res_auto_rot[0].shape == (batch_size, 3, 108, 192)
    assert res_auto_rot[1].shape == (batch_size, 128)

    # 5. Mode "all"
    res_all = model(x, mode="all", distance_meters=dist, sin_cos=sin_cos)
    assert isinstance(res_all, dict)
    assert "reconstruction" in res_all
    assert "translation" in res_all
    assert "rotation" in res_all

    # 6. Unsupported mode raises ValueError
    with pytest.raises(ValueError, match="Unsupported mode"):
        model(x, mode="unknown_mode")


def test_convenience_methods() -> None:
    """Test encode, decode, and get_latent convenience delegation."""
    batch_size = 2
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )
    x = torch.rand(batch_size, 3, 108, 192)

    mu, logvar = model.encode(x)
    assert mu.shape == (batch_size, 128)
    assert logvar.shape == (batch_size, 128)

    mu_direct = model.get_latent(x)
    assert torch.equal(mu, mu_direct)

    recon = model.decode(mu)
    assert recon.shape == (batch_size, 3, 108, 192)


def test_parameter_groups_and_registration() -> None:
    """Test submodule registrations and parameter group generation."""
    model = JointNavigationModel(
        latent_dim=128, hidden_dim=64, num_blocks=2, block_inner_dim=64
    )

    # Verify child modules
    child_names = [name for name, _ in model.named_children()]
    assert "vae" in child_names
    assert "forward_transformer" in child_names
    assert "rotation_transformer" in child_names

    # Total parameters equals sum of submodules
    total_params = sum(p.numel() for p in model.parameters())
    vae_params = sum(p.numel() for p in model.vae.parameters())
    fwd_params = sum(p.numel() for p in model.forward_transformer.parameters())
    rot_params = sum(p.numel() for p in model.rotation_transformer.parameters())
    assert total_params == vae_params + fwd_params + rot_params

    # Parameter groups
    groups = model.get_parameter_groups(
        vae_lr=1e-4, forward_lr=2e-4, rotation_lr=3e-4, base_lr=5e-5
    )
    assert len(groups) == 3
    assert groups[0]["name"] == "vae"
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["name"] == "forward_transformer"
    assert groups[1]["lr"] == 2e-4
    assert groups[2]["name"] == "rotation_transformer"
    assert groups[2]["lr"] == 3e-4


def test_checkpoint_export_and_standalone_loading(tmp_path: Path) -> None:
    """Test exported state dicts load into standalone models with 512-dim latents."""
    model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )

    # 1. VAE export & load into standalone VAE
    vae_sd = model.export_vae_state_dict()
    standalone_vae = VAE(latent_dim=512)
    # This must load without missing or unexpected keys
    standalone_vae.load_state_dict(vae_sd, strict=True)
    for k, v in standalone_vae.state_dict().items():
        assert torch.equal(v, model.vae.state_dict()[k])

    # 2. Forward transformer export & load into standalone ForwardLatentTransformer
    fwd_sd = model.export_forward_state_dict()
    standalone_fwd = ForwardLatentTransformer(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    standalone_fwd.load_state_dict(fwd_sd, strict=True)
    for k, v in standalone_fwd.state_dict().items():
        assert torch.equal(v, model.forward_transformer.state_dict()[k])

    # 3. Rotation transformer export & load into standalone RotationLatentTransformer
    rot_sd = model.export_rotation_state_dict()
    standalone_rot = RotationLatentTransformer(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    standalone_rot.load_state_dict(rot_sd, strict=True)
    for k, v in standalone_rot.state_dict().items():
        assert torch.equal(v, model.rotation_transformer.state_dict()[k])

    # 4. Export to filesystem checkpoints
    vae_path = tmp_path / "checkpoints" / "vae_512.pt"
    fwd_path = tmp_path / "checkpoints" / "forward_512.pt"
    rot_path = tmp_path / "checkpoints" / "rotation_512.pt"

    saved_vae_sd = model.export_vae_checkpoint(vae_path)
    saved_fwd_sd = model.export_forward_checkpoint(fwd_path)
    saved_rot_sd = model.export_rotation_checkpoint(rot_path)

    assert vae_path.exists()
    assert fwd_path.exists()
    assert rot_path.exists()

    loaded_vae_sd: Dict[str, torch.Tensor] = torch.load(
        vae_path, map_location="cpu", weights_only=True
    )
    for k, v in loaded_vae_sd.items():
        assert torch.equal(v, saved_vae_sd[k])

    loaded_fwd_sd: Dict[str, torch.Tensor] = torch.load(
        fwd_path, map_location="cpu", weights_only=True
    )
    for k, v in loaded_fwd_sd.items():
        assert torch.equal(v, saved_fwd_sd[k])

    loaded_rot_sd: Dict[str, torch.Tensor] = torch.load(
        rot_path, map_location="cpu", weights_only=True
    )
    for k, v in loaded_rot_sd.items():
        assert torch.equal(v, saved_rot_sd[k])


def test_load_pretrained_warm_start(tmp_path: Path) -> None:
    """Test warm-start loading of standalone VAE, forward, and rotation weights."""
    model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )

    # 1. Warm-start VAE from standalone VAE
    src_vae = VAE(latent_dim=512)
    # Mutate a parameter to verify it actually transfers
    with torch.no_grad():
        src_vae.fc_mu.weight.fill_(0.42)

    vae_path = tmp_path / "pretrained_vae.pt"
    torch.save(src_vae.state_dict(), vae_path)

    model.load_vae_pretrained(vae_path, strict=True)
    assert torch.allclose(
        model.vae.fc_mu.weight, torch.full_like(model.vae.fc_mu.weight, 0.42)
    )

    # 2. Test wrapped container format (e.g. {"state_dict": ...})
    wrapped_vae_path = tmp_path / "wrapped_vae.pt"
    torch.save({"model_state_dict": src_vae.state_dict()}, wrapped_vae_path)
    model.load_vae_pretrained(wrapped_vae_path, strict=True)
    assert torch.allclose(
        model.vae.fc_mu.weight, torch.full_like(model.vae.fc_mu.weight, 0.42)
    )

    # 3. Test composite key prefix format (e.g. "vae.fc_mu.weight")
    composite_sd = {f"vae.{k}": v for k, v in src_vae.state_dict().items()}
    model.load_vae_pretrained(composite_sd, strict=True)
    assert torch.allclose(
        model.vae.fc_mu.weight, torch.full_like(model.vae.fc_mu.weight, 0.42)
    )

    # 4. Test forward transformer warm-start
    src_fwd = ForwardLatentTransformer(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    with torch.no_grad():
        src_fwd.blocks[0].fc1.weight.fill_(0.84)
    model.load_forward_pretrained(src_fwd.state_dict(), strict=True)
    assert torch.allclose(
        model.forward_transformer.blocks[0].fc1.weight,
        torch.full_like(model.forward_transformer.blocks[0].fc1.weight, 0.84),
    )

    # 5. Test rotation transformer warm-start
    src_rot = RotationLatentTransformer(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    with torch.no_grad():
        src_rot.blocks[0].fc1.weight.fill_(0.96)
    model.load_rotation_pretrained(src_rot.state_dict(), strict=True)
    assert torch.allclose(
        model.rotation_transformer.blocks[0].fc1.weight,
        torch.full_like(model.rotation_transformer.blocks[0].fc1.weight, 0.96),
    )

    # 6. File not found error
    with pytest.raises(FileNotFoundError):
        model.load_vae_pretrained(tmp_path / "nonexistent.pt")


def test_gradient_flow_through_all_branches() -> None:
    """Test that backwards pass produces valid gradients across all parameter groups."""
    batch_size = 2
    model = JointNavigationModel(
        latent_dim=64, hidden_dim=32, num_blocks=2, block_inner_dim=32
    )

    x = torch.rand(batch_size, 3, 108, 192, requires_grad=True)
    dist = torch.tensor([[1.0], [2.0]])
    sin_cos = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    recon_x, mu, logvar = model.forward_reconstruction(x)
    pred_fwd, fwd_lat, _ = model.forward_translation(x, distance_meters=dist)
    pred_rot, rot_lat, _ = model.forward_rotation(x, sin_cos=sin_cos)

    loss = (
        recon_x.sum()
        + mu.sum()
        + logvar.sum()
        + pred_fwd.sum()
        + fwd_lat.sum()
        + pred_rot.sum()
        + rot_lat.sum()
    )
    loss.backward()

    # VAE gradients
    for name, p in model.vae.named_parameters():
        assert p.grad is not None, f"VAE parameter {name} has no gradient."
        assert not torch.isnan(p.grad).any(), f"VAE parameter {name} has NaN gradient."

    # Forward transformer gradients
    for name, p in model.forward_transformer.named_parameters():
        assert p.grad is not None, f"Forward parameter {name} has no gradient."
        assert not torch.isnan(p.grad).any(), (
            f"Forward parameter {name} has NaN gradient."
        )

    # Rotation transformer gradients
    for name, p in model.rotation_transformer.named_parameters():
        assert p.grad is not None, f"Rotation parameter {name} has no gradient."
        assert not torch.isnan(p.grad).any(), (
            f"Rotation parameter {name} has NaN gradient."
        )

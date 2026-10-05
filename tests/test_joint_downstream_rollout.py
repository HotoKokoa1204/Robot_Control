"""
Module: test_joint_downstream_rollout
Stage: Test
Author: KafuuChino
Date: 2026-09-29
Description: Comprehensive integration and regression tests verifying downstream
    pipeline compatibility for checkpoints exported from JointNavigationModel
    and three-stage curricular training (512-dim latent space, 5-block forward
    and rotation transformers). Tests ChainedLatentTransformer rollouts
    (ROTATE_FIRST, FORWARD_FIRST), three-stage checkpoint loading, single-action
    invariance, VAE frame decoding, progressive keyframing, nested checkpoint loading,
    and end-to-end generate_video.py execution.
"""

import json
from pathlib import Path
from typing import Dict

# Windows DLL initialization guard for PIL/torchvision
from PIL import Image  # isort: skip # noqa: F401

import torch
from agilab_lib.datasets.dual_source_dataset import DummyDualSourceVideoDataset
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.models.rlt import (
    ChainedLatentTransformer,
    ExecutionOrder,
)
from agilab_lib.models.vae import VAE
from omegaconf import OmegaConf

from scripts.generate_video import generate_video
from scripts.train_multibranch import train_multibranch


def export_test_checkpoints(
    model: JointNavigationModel, out_dir: Path
) -> Dict[str, Path]:
    """Helper to export standalone checkpoints from a JointNavigationModel.

    Args:
        model: JointNavigationModel instance.
        out_dir: Target directory to save checkpoints.

    Returns:
        Dictionary mapping model names ('vae', 'forward', 'rotation') to file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "vae": out_dir / "vae_512.pt",
        "forward": out_dir / "forward_transformer.pt",
        "rotation": out_dir / "rotation_transformer.pt",
    }
    model.export_vae_checkpoint(paths["vae"])
    model.export_forward_checkpoint(paths["forward"])
    model.export_rotation_checkpoint(paths["rotation"])
    return paths


def test_export_and_load_standalone_checkpoints(tmp_path: Path) -> None:
    """Verify exporting 512-dim standalone checkpoints and loading into models."""
    joint_model = JointNavigationModel(
        latent_dim=512,
        hidden_dim=512,
        num_blocks=5,
        block_inner_dim=512,
    )

    ckpt_paths = export_test_checkpoints(joint_model, tmp_path / "ckpts")

    for name, p in ckpt_paths.items():
        assert p.exists(), f"Missing exported checkpoint: {p}"
        assert p.stat().st_size > 0, f"Empty checkpoint file: {p}"

    # 1. Load exported vae_512.pt into standalone VAE(latent_dim=512)
    standalone_vae = VAE(latent_dim=512)
    loaded_vae_sd = torch.load(ckpt_paths["vae"], map_location="cpu", weights_only=True)
    standalone_vae.load_state_dict(loaded_vae_sd)

    for k, v in standalone_vae.state_dict().items():
        assert torch.equal(v, joint_model.vae.state_dict()[k])

    # 2. Load forward and rotation checkpoints into standalone ChainedLatentTransformer
    chained_model = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpt_paths["rotation"],
        forward_checkpoint=ckpt_paths["forward"],
    )

    for k, v in chained_model.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])

    for k, v in chained_model.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])


def test_nested_checkpoint_loading(tmp_path: Path) -> None:
    """Verify ChainedLatentTransformer safely unrolls nested state dicts."""
    joint_model = JointNavigationModel(
        latent_dim=512,
        hidden_dim=512,
        num_blocks=5,
        block_inner_dim=512,
    )

    # Test 'state_dict' wrapper
    nested_rot_path = tmp_path / "nested_rot.pt"
    torch.save(
        {"state_dict": joint_model.export_rotation_state_dict()}, nested_rot_path
    )

    # Test 'model_state_dict' wrapper
    nested_fwd_path = tmp_path / "nested_fwd.pt"
    torch.save(
        {"model_state_dict": joint_model.export_forward_state_dict()},
        nested_fwd_path,
    )

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=nested_rot_path,
        forward_checkpoint=nested_fwd_path,
    )

    for k, v in chained.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])

    for k, v in chained.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])

    # Test loading from a composite full JointNavigationModel state_dict directly
    composite_path = tmp_path / "composite_joint.pt"
    torch.save(joint_model.state_dict(), composite_path)

    chained_from_composite = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=composite_path,
        forward_checkpoint=composite_path,
    )

    for k, v in chained_from_composite.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])

    for k, v in chained_from_composite.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])


def test_compound_rollout_rotate_first(tmp_path: Path) -> None:
    """Verify compound rollout under ROTATE_FIRST execution sequence."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_rf")

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    batch_size = 4
    z_init = torch.randn(batch_size, 512)
    angle_deg = torch.tensor([15.0, -30.0, 45.0, 90.0])
    distance_m = torch.tensor([0.5, 1.0, 1.5, 2.0])

    final_z, mid_z = chained(
        z_init,
        angle_deg=angle_deg,
        distance_meters=distance_m,
        execution_order=ExecutionOrder.ROTATE_FIRST,
        return_intermediate=True,
    )

    # Verify tensor shapes
    assert mid_z.shape == (batch_size, 512)
    assert final_z.shape == (batch_size, 512)

    # Verify values are finite
    assert torch.all(torch.isfinite(mid_z))
    assert torch.all(torch.isfinite(final_z))

    # Verify numerical consistency with underlying decoupled modules
    with torch.no_grad():
        expected_mid = chained.rotation_model(z_init, angle_deg=angle_deg)
        expected_final = chained.forward_model(mid_z, distance_meters=distance_m)

    assert torch.allclose(mid_z, expected_mid, atol=1e-6)
    assert torch.allclose(final_z, expected_final, atol=1e-6)


def test_compound_rollout_forward_first(tmp_path: Path) -> None:
    """Verify compound rollout under FORWARD_FIRST execution sequence."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_ff")

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    batch_size = 4
    z_init = torch.randn(batch_size, 512)
    angle_deg = torch.tensor([-45.0, 30.0, -90.0, 60.0])
    distance_m = torch.tensor([1.2, 0.4, 2.5, 0.8])

    final_z, mid_z = chained(
        z_init,
        angle_deg=angle_deg,
        distance_meters=distance_m,
        execution_order=ExecutionOrder.FORWARD_FIRST,
        return_intermediate=True,
    )

    # Verify tensor shapes
    assert mid_z.shape == (batch_size, 512)
    assert final_z.shape == (batch_size, 512)

    # Verify values are finite
    assert torch.all(torch.isfinite(mid_z))
    assert torch.all(torch.isfinite(final_z))

    # Verify numerical consistency with underlying decoupled modules
    with torch.no_grad():
        expected_mid = chained.forward_model(z_init, distance_meters=distance_m)
        expected_final = chained.rotation_model(mid_z, angle_deg=angle_deg)

    assert torch.allclose(mid_z, expected_mid, atol=1e-6)
    assert torch.allclose(final_z, expected_final, atol=1e-6)


def test_single_action_invariance(tmp_path: Path) -> None:
    """Verify single-action invariance for zero angle and zero distance."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_inv")

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    z_init = torch.randn(3, 512)

    with torch.no_grad():
        # Case 1: Zero angle, positive distance in ROTATE_FIRST
        # Rotation step must produce unchanged latent (mid_z == z_init)
        final_z, mid_z = chained(
            z_init,
            angle_deg=0.0,
            distance_meters=1.5,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )
        assert torch.equal(mid_z, z_init)
        expected_fwd = chained.forward_model(z_init, distance_meters=1.5)
        assert torch.allclose(final_z, expected_fwd, atol=1e-6)

        # Case 2: Positive angle, zero distance in ROTATE_FIRST
        # Forward step must produce unchanged latent (final_z == mid_z)
        final_z2, mid_z2 = chained(
            z_init,
            angle_deg=45.0,
            distance_meters=0.0,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )
        expected_rot = chained.rotation_model(z_init, angle_deg=45.0)
        assert torch.allclose(mid_z2, expected_rot, atol=1e-6)
        assert torch.equal(final_z2, mid_z2)

        # Case 3: Zero angle, positive distance in FORWARD_FIRST
        # Rotation step must produce unchanged latent (final_z == mid_z)
        final_z3, mid_z3 = chained(
            z_init,
            angle_deg=0.0,
            distance_meters=1.5,
            execution_order=ExecutionOrder.FORWARD_FIRST,
            return_intermediate=True,
        )
        assert torch.allclose(mid_z3, expected_fwd, atol=1e-6)
        assert torch.equal(final_z3, mid_z3)

        # Case 4: Positive angle, zero distance in FORWARD_FIRST
        # Forward step must produce unchanged latent (mid_z == z_init)
        final_z4, mid_z4 = chained(
            z_init,
            angle_deg=45.0,
            distance_meters=0.0,
            execution_order=ExecutionOrder.FORWARD_FIRST,
            return_intermediate=True,
        )
        assert torch.equal(mid_z4, z_init)
        assert torch.allclose(final_z4, expected_rot, atol=1e-6)

        # Case 5: Zero angle, zero distance -> completely unchanged
        final_zero, mid_zero = chained(
            z_init,
            angle_deg=0.0,
            distance_meters=0.0,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )
        assert torch.equal(mid_zero, z_init)
        assert torch.equal(final_zero, z_init)


def test_decoding_rollouts_with_standalone_vae(tmp_path: Path) -> None:
    """Verify decoding intermediate and final latents using standalone VAE(512)."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_dec")

    standalone_vae = VAE(latent_dim=512)
    standalone_vae.load_state_dict(
        torch.load(ckpts["vae"], map_location="cpu", weights_only=True)
    )
    standalone_vae.eval()

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    # Synthetic RGB frame batch (B, 3, 108, 192)
    batch_size = 2
    x = torch.rand(batch_size, 3, 108, 192)

    with torch.no_grad():
        z_0 = standalone_vae.get_latent(x)
        assert z_0.shape == (batch_size, 512)

        final_z, mid_z = chained(
            z_0,
            angle_deg=30.0,
            distance_meters=1.0,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )

        mid_recon = standalone_vae.decode(mid_z)
        final_recon = standalone_vae.decode(final_z)

    # Verify frame shapes
    assert mid_recon.shape == (batch_size, 3, 108, 192)
    assert final_recon.shape == (batch_size, 3, 108, 192)

    # Verify pixel ranges and finite values
    assert torch.all(mid_recon >= 0.0)
    assert torch.all(mid_recon <= 1.0)
    assert torch.all(torch.isfinite(mid_recon))

    assert torch.all(final_recon >= 0.0)
    assert torch.all(final_recon <= 1.0)
    assert torch.all(torch.isfinite(final_recon))


def test_generate_progressive_keyframes_512(tmp_path: Path) -> None:
    """Verify progressive keyframing with 512-dim latents and VAE decoding."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_prog")

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    standalone_vae = VAE(latent_dim=512)
    standalone_vae.load_state_dict(
        torch.load(ckpts["vae"], map_location="cpu", weights_only=True)
    )
    standalone_vae.eval()

    z_start = torch.randn(1, 512)
    keyframes = chained.generate_progressive_keyframes(
        latent=z_start,
        angle_deg=60.0,
        distance_meters=1.0,
        execution_order=ExecutionOrder.ROTATE_FIRST,
        substep_angle_deg=20.0,
        substep_distance_meters=0.5,
    )

    # Substepping: ceil(60/20) = 3 angle steps, ceil(1.0/0.5) = 2 distance steps
    # Total keyframes = 1 (start) + 3 + 2 = 6
    assert keyframes.shape == (6, 512)
    assert torch.all(torch.isfinite(keyframes))

    with torch.no_grad():
        recon_frames = standalone_vae.decode(keyframes)

    assert recon_frames.shape == (6, 3, 108, 192)
    assert torch.all(recon_frames >= 0.0)
    assert torch.all(recon_frames <= 1.0)
    assert torch.all(torch.isfinite(recon_frames))


def test_generate_video_end_to_end_512(tmp_path: Path) -> None:
    """Verify end-to-end generate_video.py execution with 512-dim checkpoints."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_vid")

    # Prepare dummy keyframes JSON
    kf_json_path = tmp_path / "keyframes.json"
    with open(kf_json_path, "w", encoding="utf-8") as f:
        json.dump({"keyframe_indices": [0, 5, 10]}, f)

    out_video_path = tmp_path / "generated_video_512.mp4"

    cfg = OmegaConf.create(
        {
            "video_path": str(tmp_path / "nonexistent.mp4"),
            "keyframes_json": str(kf_json_path),
            "latent_dim": 512,
            "vae_checkpoint": str(ckpts["vae"]),
            "interp_steps": 2,
            "use_rrdn": False,
            "rrdn_checkpoint": "",
            "rrdn_upscale_factor": 2,
            "output_video": str(out_video_path),
            "fps: ": 10,
            "fps": 10,
            "use_dummy_if_missing": True,
            "use_chained_transformer": True,
            "rotation_checkpoint": str(ckpts["rotation"]),
            "forward_checkpoint": str(ckpts["forward"]),
            "execution_order": "rotate_first",
            "motion_angle_deg": 30.0,
            "motion_distance_meters": 0.5,
            "substep_angle_deg": 15.0,
            "substep_distance_meters": 0.25,
            # Let hidden_dim, forward_num_blocks, block_inner_dim default automatically
            "hidden_dim": None,
            "forward_num_blocks": None,
            "rotation_num_blocks": None,
            "block_inner_dim": None,
        }
    )

    result_path = generate_video(cfg)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0
    assert result_path == str(out_video_path)


def test_generate_video_forward_first_512(tmp_path: Path) -> None:
    """Verify end-to-end generate_video.py with FORWARD_FIRST execution order."""
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_ff_vid")

    kf_json_path = tmp_path / "keyframes_ff.json"
    with open(kf_json_path, "w", encoding="utf-8") as f:
        json.dump({"keyframe_indices": [0, 2]}, f)

    out_video_path = tmp_path / "video_ff_512.mp4"

    cfg = OmegaConf.create(
        {
            "video_path": str(tmp_path / "missing.mp4"),
            "keyframes_json": str(kf_json_path),
            "latent_dim": 512,
            "vae_checkpoint": str(ckpts["vae"]),
            "interp_steps": 2,
            "use_rrdn": False,
            "rrdn_checkpoint": "",
            "rrdn_upscale_factor": 2,
            "output_video": str(out_video_path),
            "fps": 10,
            "use_dummy_if_missing": True,
            "use_chained_transformer": True,
            "rotation_checkpoint": str(ckpts["rotation"]),
            "forward_checkpoint": str(ckpts["forward"]),
            "execution_order": "forward_first",
            "motion_angle_deg": 45.0,
            "motion_distance_meters": 1.0,
            "substep_angle_deg": None,
            "substep_distance_meters": None,
            "hidden_dim": 512,
            "forward_num_blocks": 5,
            "rotation_num_blocks": 5,
            "block_inner_dim": 512,
        }
    )

    result_path = generate_video(cfg)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0


def test_multistep_compound_rollout_preserves_latent_norm_and_luminance(
    tmp_path: Path,
) -> None:
    """Verify multi-step compound rollout preserves latent norm and avoids collapse.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_test_checkpoints(joint_model, tmp_path / "ckpts_multistep")

    standalone_vae = VAE(latent_dim=512)
    standalone_vae.load_state_dict(
        torch.load(ckpts["vae"], map_location="cpu", weights_only=True)
    )
    standalone_vae.eval()

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    chained.eval()

    # Synthetic realistic-ish image tensor in [0.2, 0.8] range
    torch.manual_seed(42)
    x0 = torch.rand(1, 3, 108, 192) * 0.6 + 0.2

    with torch.no_grad():
        z = standalone_vae.get_latent(x0)
        initial_norm = torch.norm(z, dim=-1).item()
        initial_frame = standalone_vae.decode(z)
        initial_mean_luminance = initial_frame.mean().item()

        # Step through 5 compound rollouts with zero-init direct-stream model
        current_z = z
        for _ in range(5):
            current_z = chained(
                current_z,
                angle_deg=15.0,
                distance_meters=0.5,
                execution_order=ExecutionOrder.ROTATE_FIRST,
            )
            current_norm = torch.norm(current_z, dim=-1).item()
            # Norm must stay exactly identical at initialization
            assert abs(current_norm - initial_norm) < 1e-5

        final_frame = standalone_vae.decode(current_z)
        final_mean_luminance = final_frame.mean().item()

        # Decoded frames must be identical - zero illumination decay or collapse
        assert abs(final_mean_luminance - initial_mean_luminance) < 1e-5
        assert torch.allclose(final_frame, initial_frame, atol=1e-5)


def export_three_stage_checkpoints(
    model: JointNavigationModel, out_dir: Path
) -> Dict[str, Path]:
    """Helper to export checkpoints mimicking the three-stage training milestones.

    Exports:
    - Stage 1: stage1_vae.pt
    - Stage 2: stage2_transformers.pt, forward_transformer.pt, rotation_transformer.pt
    - Stage 3: joint_navigation_model.pt, final_joint_model.pt, vae_512.pt

    Args:
        model: JointNavigationModel instance.
        out_dir: Target directory to save checkpoints.

    Returns:
        Dictionary mapping checkpoint names to file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage1_vae": out_dir / "stage1_vae.pt",
        "stage2_transformers": out_dir / "stage2_transformers.pt",
        "forward": out_dir / "forward_transformer.pt",
        "rotation": out_dir / "rotation_transformer.pt",
        "joint_navigation_model": out_dir / "joint_navigation_model.pt",
        "final_joint_model": out_dir / "final_joint_model.pt",
        "vae_512": out_dir / "vae_512.pt",
    }
    # Stage 1 milestone
    model.export_vae_checkpoint(paths["stage1_vae"])

    # Stage 2 milestone
    torch.save(
        {
            "forward_transformer": model.forward_transformer.state_dict(),
            "rotation_transformer": model.rotation_transformer.state_dict(),
        },
        paths["stage2_transformers"],
    )
    model.export_forward_checkpoint(paths["forward"])
    model.export_rotation_checkpoint(paths["rotation"])

    # Stage 3 milestones
    torch.save(model.state_dict(), paths["joint_navigation_model"])
    torch.save(model.state_dict(), paths["final_joint_model"])
    model.export_vae_checkpoint(paths["vae_512"])

    return paths


def test_three_stage_checkpoints_loading(tmp_path: Path) -> None:
    """Verify Stage 1, Stage 2, and Stage 3 checkpoints load into downstream models.

    Validates that:
    - stage1_vae.pt loads cleanly into standalone VAE and JointNavigationModel.
    - stage2_transformers.pt (and standalone forward/rotation checkpoints) load cleanly
      into ChainedLatentTransformer and JointNavigationModel.
    - joint_navigation_model.pt loads cleanly into JointNavigationModel and
      ChainedLatentTransformer.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    joint_model = JointNavigationModel(
        latent_dim=512,
        hidden_dim=512,
        num_blocks=5,
        block_inner_dim=512,
    )
    ckpts = export_three_stage_checkpoints(joint_model, tmp_path / "three_stage_ckpts")

    # 1. Verify stage1_vae.pt loading
    standalone_vae = VAE(latent_dim=512)
    stage1_sd = torch.load(ckpts["stage1_vae"], map_location="cpu", weights_only=True)
    standalone_vae.load_state_dict(stage1_sd)
    for k, v in standalone_vae.state_dict().items():
        assert torch.equal(v, joint_model.vae.state_dict()[k])

    downstream_joint = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    downstream_joint.load_vae_pretrained(ckpts["stage1_vae"])
    for k, v in downstream_joint.vae.state_dict().items():
        assert torch.equal(v, joint_model.vae.state_dict()[k])

    # 2. Verify stage2_transformers.pt loading into ChainedLatentTransformer
    chained_from_stage2 = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["stage2_transformers"],
        forward_checkpoint=ckpts["stage2_transformers"],
    )
    for k, v in chained_from_stage2.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])
    for k, v in chained_from_stage2.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])

    # Verify stage2_transformers.pt loading into JointNavigationModel
    downstream_joint.load_forward_pretrained(ckpts["stage2_transformers"])
    downstream_joint.load_rotation_pretrained(ckpts["stage2_transformers"])
    for k, v in downstream_joint.forward_transformer.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])
    for k, v in downstream_joint.rotation_transformer.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])

    # 3. Verify standalone forward_transformer.pt and rotation_transformer.pt loading
    chained_from_standalone = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["rotation"],
        forward_checkpoint=ckpts["forward"],
    )
    for k, v in chained_from_standalone.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])
    for k, v in chained_from_standalone.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])

    # 4. Verify Stage 3 composite joint_navigation_model.pt loading
    joint_model_reloaded = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    joint_model_reloaded.load_state_dict(
        torch.load(
            ckpts["joint_navigation_model"], map_location="cpu", weights_only=True
        )
    )
    for k, v in joint_model_reloaded.state_dict().items():
        assert torch.equal(v, joint_model.state_dict()[k])

    chained_from_composite = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["joint_navigation_model"],
        forward_checkpoint=ckpts["joint_navigation_model"],
    )
    for k, v in chained_from_composite.forward_model.state_dict().items():
        assert torch.equal(v, joint_model.forward_transformer.state_dict()[k])
    for k, v in chained_from_composite.rotation_model.state_dict().items():
        assert torch.equal(v, joint_model.rotation_transformer.state_dict()[k])


def test_three_stage_sequential_compound_rollouts(tmp_path: Path) -> None:
    """Verify sequential compound rollouts on latents from three-stage checkpoints.

    Tests Rotate-First and Forward-First compound rollouts without dimension errors,
    numerical degeneration, or shape mismatch.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_three_stage_checkpoints(joint_model, tmp_path / "rollout_ckpts")

    vae = VAE(latent_dim=512)
    vae.load_state_dict(
        torch.load(ckpts["stage1_vae"], map_location="cpu", weights_only=True)
    )
    vae.eval()

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["stage2_transformers"],
        forward_checkpoint=ckpts["stage2_transformers"],
    )
    chained.eval()

    batch_size = 4
    torch.manual_seed(101)
    x = torch.rand(batch_size, 3, 108, 192)

    with torch.no_grad():
        z_init = vae.get_latent(x)
        assert z_init.shape == (batch_size, 512)
        assert torch.all(torch.isfinite(z_init))

        # Rotate-First compound rollout
        angle_rf = torch.tensor([25.0, -45.0, 60.0, -90.0])
        dist_rf = torch.tensor([0.8, 1.5, 0.3, 2.0])
        final_rf, mid_rf = chained(
            z_init,
            angle_deg=angle_rf,
            distance_meters=dist_rf,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )

        assert mid_rf.shape == (batch_size, 512)
        assert final_rf.shape == (batch_size, 512)
        assert torch.all(torch.isfinite(mid_rf))
        assert torch.all(torch.isfinite(final_rf))

        expected_mid_rf = chained.rotation_model(z_init, angle_deg=angle_rf)
        expected_final_rf = chained.forward_model(mid_rf, distance_meters=dist_rf)
        assert torch.allclose(mid_rf, expected_mid_rf, atol=1e-6)
        assert torch.allclose(final_rf, expected_final_rf, atol=1e-6)

        # Forward-First compound rollout
        angle_ff = torch.tensor([-30.0, 45.0, -15.0, 75.0])
        dist_ff = torch.tensor([1.0, 0.5, 2.2, 1.4])
        final_ff, mid_ff = chained(
            z_init,
            angle_deg=angle_ff,
            distance_meters=dist_ff,
            execution_order=ExecutionOrder.FORWARD_FIRST,
            return_intermediate=True,
        )

        assert mid_ff.shape == (batch_size, 512)
        assert final_ff.shape == (batch_size, 512)
        assert torch.all(torch.isfinite(mid_ff))
        assert torch.all(torch.isfinite(final_ff))

        expected_mid_ff = chained.forward_model(z_init, distance_meters=dist_ff)
        expected_final_ff = chained.rotation_model(mid_ff, angle_deg=angle_ff)
        assert torch.allclose(mid_ff, expected_mid_ff, atol=1e-6)
        assert torch.allclose(final_ff, expected_final_ff, atol=1e-6)

        # Verify equivalence with ChainedLatentTransformer loaded from
        # joint_navigation_model.pt
        chained_joint = ChainedLatentTransformer(
            latent_dim=512,
            hidden_dim=512,
            forward_num_blocks=5,
            rotation_num_blocks=5,
            block_inner_dim=512,
            rotation_checkpoint=ckpts["joint_navigation_model"],
            forward_checkpoint=ckpts["joint_navigation_model"],
        )
        chained_joint.eval()

        final_joint_rf, mid_joint_rf = chained_joint(
            z_init,
            angle_deg=angle_rf,
            distance_meters=dist_rf,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )
        assert torch.allclose(final_rf, final_joint_rf, atol=1e-6)
        assert torch.allclose(mid_rf, mid_joint_rf, atol=1e-6)


def test_three_stage_decoder_image_reconstruction(tmp_path: Path) -> None:
    """Verify decoder decodes chained rollout latents to valid images in [0, 1].

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    joint_model = JointNavigationModel(
        latent_dim=512, hidden_dim=512, num_blocks=5, block_inner_dim=512
    )
    ckpts = export_three_stage_checkpoints(joint_model, tmp_path / "decode_ckpts")

    vae = VAE(latent_dim=512)
    vae.load_state_dict(
        torch.load(ckpts["stage1_vae"], map_location="cpu", weights_only=True)
    )
    vae.eval()

    chained = ChainedLatentTransformer(
        latent_dim=512,
        hidden_dim=512,
        forward_num_blocks=5,
        rotation_num_blocks=5,
        block_inner_dim=512,
        rotation_checkpoint=ckpts["stage2_transformers"],
        forward_checkpoint=ckpts["stage2_transformers"],
    )
    chained.eval()

    batch_size = 3
    torch.manual_seed(202)
    x = torch.rand(batch_size, 3, 108, 192)

    with torch.no_grad():
        z_init = vae.get_latent(x)

        # 1. Rotate-First compound rollout and frame decoding
        final_rf, mid_rf = chained(
            z_init,
            angle_deg=torch.tensor([30.0, -45.0, 60.0]),
            distance_meters=torch.tensor([1.0, 0.5, 1.8]),
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )

        mid_recon_rf = vae.decode(mid_rf)
        final_recon_rf = vae.decode(final_rf)

        assert mid_recon_rf.shape == (batch_size, 3, 108, 192)
        assert final_recon_rf.shape == (batch_size, 3, 108, 192)
        assert torch.all(mid_recon_rf >= 0.0)
        assert torch.all(mid_recon_rf <= 1.0)
        assert torch.all(torch.isfinite(mid_recon_rf))
        assert torch.all(final_recon_rf >= 0.0)
        assert torch.all(final_recon_rf <= 1.0)
        assert torch.all(torch.isfinite(final_recon_rf))

        # 2. Forward-First compound rollout and frame decoding
        final_ff, mid_ff = chained(
            z_init,
            angle_deg=torch.tensor([-15.0, 50.0, -90.0]),
            distance_meters=torch.tensor([0.4, 2.1, 1.2]),
            execution_order=ExecutionOrder.FORWARD_FIRST,
            return_intermediate=True,
        )

        mid_recon_ff = vae.decode(mid_ff)
        final_recon_ff = vae.decode(final_ff)

        assert mid_recon_ff.shape == (batch_size, 3, 108, 192)
        assert final_recon_ff.shape == (batch_size, 3, 108, 192)
        assert torch.all(mid_recon_ff >= 0.0)
        assert torch.all(mid_recon_ff <= 1.0)
        assert torch.all(torch.isfinite(mid_recon_ff))
        assert torch.all(final_recon_ff >= 0.0)
        assert torch.all(final_recon_ff <= 1.0)
        assert torch.all(torch.isfinite(final_recon_ff))

        # 3. Progressive keyframing and frame decoding
        keyframes = chained.generate_progressive_keyframes(
            latent=z_init[0:1],
            angle_deg=40.0,
            distance_meters=1.0,
            execution_order=ExecutionOrder.ROTATE_FIRST,
            substep_angle_deg=20.0,
            substep_distance_meters=0.5,
        )
        assert keyframes.shape == (5, 512)
        decoded_keyframes = vae.decode(keyframes)
        assert decoded_keyframes.shape == (5, 3, 108, 192)
        assert torch.all(decoded_keyframes >= 0.0)
        assert torch.all(decoded_keyframes <= 1.0)
        assert torch.all(torch.isfinite(decoded_keyframes))


def test_end_to_end_three_stage_training_downstream_rollout(
    tmp_path: Path,
) -> None:
    """Verify downstream rollout and decoding from a real three-stage training run.

    Executes train_multibranch with 1 epoch per stage, produces real milestone
    checkpoints (stage1_vae.pt, stage2_transformers.pt, joint_navigation_model.pt),
    and validates clean downstream loading, compound rollout, and frame decoding.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
    """
    dataset = DummyDualSourceVideoDataset(
        num_samples=4,
        img_height=108,
        img_width=192,
        seed=123,
    )
    init_model = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    out_dir = tmp_path / "trained_ckpts"
    cfg = OmegaConf.create(
        {
            "latent_dim": 64,
            "hidden_dim": 32,
            "num_blocks": 2,
            "block_inner_dim": 32,
            "lr": 1e-4,
            "lr_vae_stage1": 1e-4,
            "lr_vae_stage3": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "use_amp": False,
            "max_epochs": 3,
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "stage3_epochs": 1,
            "w_latent": 1.0,
            "sanity_check_min_std": 0.1,
            "alpha_perc": 0.5,
            "beta_kl": 0.0001,
            "w_fwd": 1.0,
            "w_rot": 1.0,
            "w_recon": 1.0,
            "output_dir": str(out_dir),
            "save_interval_epochs": 1,
            "vae_checkpoint": None,
            "device": "cpu",
        }
    )

    _, history = train_multibranch(cfg, dataset=dataset, model=init_model)
    assert len(history["loss"]) == 3

    # Check that milestone checkpoints exist
    stage1_ckpt = out_dir / "stage1_vae.pt"
    stage2_ckpt = out_dir / "stage2_transformers.pt"
    joint_ckpt = out_dir / "joint_navigation_model.pt"
    assert stage1_ckpt.exists()
    assert stage2_ckpt.exists()
    assert joint_ckpt.exists()

    # Load into downstream VAE
    downstream_vae = VAE(latent_dim=64)
    downstream_vae.load_state_dict(
        torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    )
    downstream_vae.eval()

    # Load into downstream ChainedLatentTransformer using stage2_transformers.pt
    chained = ChainedLatentTransformer(
        latent_dim=64,
        hidden_dim=32,
        forward_num_blocks=2,
        rotation_num_blocks=2,
        block_inner_dim=32,
        rotation_checkpoint=stage2_ckpt,
        forward_checkpoint=stage2_ckpt,
    )
    chained.eval()

    # Load into downstream JointNavigationModel using joint_navigation_model.pt
    downstream_joint = JointNavigationModel(
        latent_dim=64,
        hidden_dim=32,
        num_blocks=2,
        block_inner_dim=32,
    )
    downstream_joint.load_state_dict(
        torch.load(joint_ckpt, map_location="cpu", weights_only=True)
    )
    downstream_joint.eval()

    # Test sequential compound rollouts from encoded frame
    test_frame = torch.rand(2, 3, 108, 192)
    with torch.no_grad():
        z_start = downstream_vae.get_latent(test_frame)
        assert z_start.shape == (2, 64)

        # Rotate-First rollout
        final_rf, mid_rf = chained(
            z_start,
            angle_deg=torch.tensor([30.0, -45.0]),
            distance_meters=torch.tensor([1.0, 0.5]),
            execution_order=ExecutionOrder.ROTATE_FIRST,
            return_intermediate=True,
        )
        assert final_rf.shape == (2, 64)
        assert mid_rf.shape == (2, 64)
        assert torch.all(torch.isfinite(final_rf))

        # Decode using downstream VAE
        decoded_rf = downstream_vae.decode(final_rf)
        assert decoded_rf.shape == (2, 3, 108, 192)
        assert torch.all(decoded_rf >= 0.0)
        assert torch.all(decoded_rf <= 1.0)
        assert torch.all(torch.isfinite(decoded_rf))

        # Forward-First rollout
        final_ff, mid_ff = chained(
            z_start,
            angle_deg=torch.tensor([-60.0, 20.0]),
            distance_meters=torch.tensor([1.5, 0.8]),
            execution_order=ExecutionOrder.FORWARD_FIRST,
            return_intermediate=True,
        )
        assert final_ff.shape == (2, 64)
        assert mid_ff.shape == (2, 64)
        assert torch.all(torch.isfinite(final_ff))

        # Decode using downstream JointNavigationModel decoder
        decoded_ff = downstream_joint.decode(final_ff)
        assert decoded_ff.shape == (2, 3, 108, 192)
        assert torch.all(decoded_ff >= 0.0)
        assert torch.all(decoded_ff <= 1.0)
        assert torch.all(torch.isfinite(decoded_ff))

"""
Module: generate_video
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Reconstructs and interpolates video from Keyframes and Latent Vectors.
"""

import json
import sys
from pathlib import Path
from typing import List, Union

# Windows DLL initialization guard for PIL/torchvision
from PIL import Image  # isort: skip # noqa: F401

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure 'src' is in sys.path when invoked directly
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agilab_lib.datasets.video_dataset import (  # noqa: E402
    DummyVideoDataset,
    VideoDataset,
)
from agilab_lib.models.rlt import (  # noqa: E402
    ChainedLatentTransformer,
    ExecutionOrder,
)
from agilab_lib.models.rrdn import RRDN  # noqa: E402
from agilab_lib.models.vae import VAE  # noqa: E402
from agilab_lib.utils.interpolation import (  # noqa: E402
    linear_interpolate_latent_sequence,
)
from agilab_lib.utils.storage import resolve_project_path  # noqa: E402


def render_frames_to_video(
    frames: List[np.ndarray], output_video_path: Union[str, Path], fps: int = 30
) -> None:
    """Writes a sequence of RGB/BGR numpy frames to an MP4 video file.

    Args:
        frames: List of images of shape (H, W, 3) with uint8 dtype.
        output_video_path: Destination path for .mp4 file.
        fps: Playback frame rate.
    """
    if not frames:
        return

    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)
    writer.release()


def generate_video(cfg: DictConfig) -> str:
    """Executes the video generation and interpolation pipeline.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        Path to the generated video file.
    """
    print("Executing Video Generation pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize VAE
    latent_dim = int(cfg.latent_dim)
    vae = VAE(latent_dim=latent_dim).to(device)
    if cfg.get("vae_checkpoint") and str(cfg.vae_checkpoint).strip():
        vae_ckpt_path = resolve_project_path(str(cfg.vae_checkpoint))
        if vae_ckpt_path.exists():
            ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict):
                for candidate_key in (
                    "state_dict",
                    "model_state_dict",
                    "vae_state_dict",
                    "vae",
                ):
                    if candidate_key in ckpt and isinstance(ckpt[candidate_key], dict):
                        ckpt = ckpt[candidate_key]
                        break
                cleaned_sd = {}
                vae_keys = set(vae.state_dict().keys())
                for k, v in ckpt.items():
                    key = k
                    if key.startswith("module."):
                        key = key[len("module.") :]
                    if key.startswith("vae."):
                        key = key[len("vae.") :]
                    if key in vae_keys:
                        cleaned_sd[key] = v
                vae.load_state_dict(cleaned_sd if cleaned_sd else ckpt)
            else:
                vae.load_state_dict(ckpt)
            print(f"Loaded VAE checkpoint: {vae_ckpt_path}")
    vae.eval()

    # 2. Load dataset
    video_path = resolve_project_path(str(cfg.video_path))
    if video_path.exists():
        dataset = VideoDataset(video_path)
        print(f"Loaded real video with {len(dataset)} frames.")
    elif cfg.use_dummy_if_missing:
        dataset = DummyVideoDataset(num_frames=30)
        print(f"Video file missing. Using DummyVideoDataset ({len(dataset)} frames).")
    else:
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # 3. Load Keyframe indices
    keyframe_indices: List[int] = []
    kf_path = resolve_project_path(str(cfg.keyframes_json))
    if kf_path.exists():
        with open(kf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            keyframe_indices = data.get("keyframe_indices", [])
        print(f"Loaded {len(keyframe_indices)} keyframes from {kf_path}")
    elif cfg.use_dummy_if_missing:
        keyframe_indices = [0, min(10, len(dataset) - 1), len(dataset) - 1]
        print(f"Keyframe JSON missing. Using default indices: {keyframe_indices}")
    else:
        raise FileNotFoundError(f"Keyframe JSON not found at: {kf_path}")

    # Filter valid keyframe indices
    valid_indices = [idx for idx in keyframe_indices if idx < len(dataset)]
    if not valid_indices:
        valid_indices = [0]

    # 4. Extract and encode Keyframes into Latent Vectors
    keyframe_tensors = [dataset[idx] for idx in valid_indices]
    with torch.no_grad():
        batch_kf = torch.stack(keyframe_tensors, dim=0).to(device)
        keyframe_latents = vae.get_latent(batch_kf)

    # 4b. Optional compound motion projection via ChainedLatentTransformer
    if cfg.get("use_chained_transformer", False):
        rot_ckpt_cfg = cfg.get("rotation_checkpoint", "")
        fwd_ckpt_cfg = cfg.get("forward_checkpoint", "")
        rot_ckpt = (
            resolve_project_path(str(rot_ckpt_cfg))
            if rot_ckpt_cfg and str(rot_ckpt_cfg).strip()
            else None
        )
        fwd_ckpt = (
            resolve_project_path(str(fwd_ckpt_cfg))
            if fwd_ckpt_cfg and str(fwd_ckpt_cfg).strip()
            else None
        )
        execution_order = ExecutionOrder(
            str(cfg.get("execution_order", "rotate_first"))
        )
        angle_deg = float(cfg.get("motion_angle_deg", 0.0))
        dist_m = float(cfg.get("motion_distance_meters", 0.0))
        substep_angle = cfg.get("substep_angle_deg", None)
        substep_dist = cfg.get("substep_distance_meters", None)

        default_hidden = 512 if latent_dim == 512 else 128
        hidden_dim = int(cfg.get("hidden_dim") or default_hidden)

        default_fwd_blocks = 5 if latent_dim == 512 else 2
        fwd_blocks = int(cfg.get("forward_num_blocks") or default_fwd_blocks)

        rot_blocks = int(cfg.get("rotation_num_blocks") or 5)

        default_inner_dim = hidden_dim
        block_inner_dim = int(cfg.get("block_inner_dim") or default_inner_dim)

        chained_model = ChainedLatentTransformer(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            forward_num_blocks=fwd_blocks,
            rotation_num_blocks=rot_blocks,
            block_inner_dim=block_inner_dim,
            rotation_checkpoint=(rot_ckpt if rot_ckpt and rot_ckpt.exists() else None),
            forward_checkpoint=(fwd_ckpt if fwd_ckpt and fwd_ckpt.exists() else None),
        ).to(device)
        chained_model.eval()

        z_start = keyframe_latents[:1]
        keyframe_latents = chained_model.generate_progressive_keyframes(
            latent=z_start,
            angle_deg=angle_deg,
            distance_meters=dist_m,
            execution_order=execution_order,
            substep_angle_deg=(
                float(substep_angle) if substep_angle is not None else None
            ),
            substep_distance_meters=(
                float(substep_dist) if substep_dist is not None else None
            ),
        )
        print(
            f"ChainedLatentTransformer generated key sequence "
            f"with {len(keyframe_latents)} keyframes (order [{execution_order}], "
            f"angle={angle_deg}deg, distance={dist_m}m)."
        )

    # 5. Linear interpolation in Latent Vector space
    interp_latents = linear_interpolate_latent_sequence(
        keyframe_latents, interp_frames=int(cfg.interp_steps)
    )
    print(f"Interpolated into {len(interp_latents)} latent vectors.")

    # 6. Decode Latent Vectors to image frames
    with torch.no_grad():
        decoded_frames = vae.decode(interp_latents)

    # 7. Optional RRDN Enhanced Decoder upscaling
    if cfg.use_rrdn:
        factor = int(cfg.rrdn_upscale_factor)
        rrdn = RRDN(upscale_factor=factor).to(device)
        if cfg.get("rrdn_checkpoint") and str(cfg.rrdn_checkpoint).strip():
            rrdn_ckpt_path = resolve_project_path(str(cfg.rrdn_checkpoint))
            if rrdn_ckpt_path.exists():
                rrdn.load_state_dict(
                    torch.load(rrdn_ckpt_path, map_location=device, weights_only=True)
                )
                print(f"Loaded RRDN checkpoint: {rrdn_ckpt_path}")
        rrdn.eval()

        with torch.no_grad():
            decoded_frames = rrdn(decoded_frames)
        print(f"Upscaled frames with RRDN (factor x{factor}).")

    # 8. Convert to uint8 BGR frames for OpenCV VideoWriter
    output_frames: List[np.ndarray] = []
    for i in range(len(decoded_frames)):
        frame_tensor = decoded_frames[i].cpu().clamp(0.0, 1.0)
        # (3, H, W) -> (H, W, 3)
        frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        # RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        output_frames.append(frame_bgr)

    # 9. Write MP4 video
    out_video_path = resolve_project_path(str(cfg.output_video))
    render_frames_to_video(output_frames, out_video_path, fps=int(cfg.fps))
    print(f"Successfully generated video saved to: {out_video_path}")
    return str(out_video_path)


@hydra.main(version_base=None, config_path="../configs", config_name="generate_video")
def main(cfg: DictConfig) -> None:
    """CLI entry point for video generation pipeline.

    Args:
        cfg: Hydra configuration dictionary.
    """
    generate_video(cfg)


if __name__ == "__main__":
    main()

"""
Module: generate_video
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Reconstructs and interpolates video from Keyframes and Latent Vectors.
"""

import json
import os
from typing import List

import cv2
import hydra
import numpy as np
import torch
from agilab_lib.datasets.video_dataset import (
    DummyVideoDataset,
    VideoDataset,
)
from agilab_lib.models.rlt import ChainedLatentTransformer
from agilab_lib.models.rrdn import RRDN
from agilab_lib.models.vae import VAE
from agilab_lib.utils.interpolation import linear_interpolate_latent_sequence
from omegaconf import DictConfig, OmegaConf


def render_frames_to_video(
    frames: List[np.ndarray], output_video_path: str, fps: int = 30
) -> None:
    """Writes a sequence of RGB/BGR numpy frames to an MP4 video file.

    Args:
        frames: List of images of shape (H, W, 3) with uint8 dtype.
        output_video_path: Destination path for .mp4 file.
        fps: Playback frame rate.
    """
    if not frames:
        return

    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)
    writer.release()


@hydra.main(version_base=None, config_path="../configs", config_name="generate_video")
def main(cfg: DictConfig) -> None:
    """Entry point for video generation pipeline.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("Executing Video Generation pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize VAE
    vae = VAE(latent_dim=int(cfg.latent_dim)).to(device)
    if cfg.vae_checkpoint and os.path.exists(cfg.vae_checkpoint):
        vae.load_state_dict(
            torch.load(cfg.vae_checkpoint, map_location=device, weights_only=True)
        )
        print(f"Loaded VAE checkpoint: {cfg.vae_checkpoint}")
    vae.eval()

    # 2. Load dataset
    if os.path.exists(cfg.video_path):
        dataset = VideoDataset(cfg.video_path)
        print(f"Loaded real video with {len(dataset)} frames.")
    elif cfg.use_dummy_if_missing:
        dataset = DummyVideoDataset(num_frames=30)
        print(f"Video file missing. Using DummyVideoDataset ({len(dataset)} frames).")
    else:
        raise FileNotFoundError(f"Video file not found at {cfg.video_path}")

    # 3. Load Keyframe indices
    keyframe_indices: List[int] = []
    if os.path.exists(cfg.keyframes_json):
        with open(cfg.keyframes_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            keyframe_indices = data.get("keyframe_indices", [])
        print(f"Loaded {len(keyframe_indices)} keyframes from {cfg.keyframes_json}")
    elif cfg.use_dummy_if_missing:
        keyframe_indices = [0, min(10, len(dataset) - 1), len(dataset) - 1]
        print(f"Keyframe JSON missing. Using default indices: {keyframe_indices}")
    else:
        raise FileNotFoundError(f"Keyframe JSON not found at: {cfg.keyframes_json}")

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
        rot_ckpt = cfg.get("rotation_checkpoint", "") or None
        fwd_ckpt = cfg.get("forward_checkpoint", "") or None
        execution_order = str(cfg.get("execution_order", "rotate_first"))
        angle_deg = float(cfg.get("motion_angle_deg", 0.0))
        dist_m = float(cfg.get("motion_distance_meters", 0.0))

        chained_model = ChainedLatentTransformer(
            latent_dim=int(cfg.latent_dim),
            rotation_checkpoint=(
                rot_ckpt if rot_ckpt and os.path.exists(str(rot_ckpt)) else None
            ),
            forward_checkpoint=(
                fwd_ckpt if fwd_ckpt and os.path.exists(str(fwd_ckpt)) else None
            ),
        ).to(device)
        chained_model.eval()

        z_start = keyframe_latents[:1]
        with torch.no_grad():
            res = chained_model(
                latent=z_start,
                angle_deg=angle_deg,
                distance_meters=dist_m,
                execution_order=execution_order,
                return_intermediate=True,
            )
            z_final, z_intermediate = res
        keyframe_latents = torch.cat([z_start, z_intermediate, z_final], dim=0)
        print(
            f"ChainedLatentTransformer generated key sequence "
            f"with order [{execution_order}] "
            f"(angle={angle_deg}deg, distance={dist_m}m)."
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
        if cfg.rrdn_checkpoint and os.path.exists(cfg.rrdn_checkpoint):
            rrdn.load_state_dict(
                torch.load(cfg.rrdn_checkpoint, map_location=device, weights_only=True)
            )
            print(f"Loaded RRDN checkpoint: {cfg.rrdn_checkpoint}")
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
    render_frames_to_video(output_frames, str(cfg.output_video), fps=int(cfg.fps))
    print(f"Successfully generated video saved to: {cfg.output_video}")


if __name__ == "__main__":
    main()

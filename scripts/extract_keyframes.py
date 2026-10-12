"""
Module: extract_keyframes
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Keyframe extraction from video via Latent Vector distances.
"""

# ruff: noqa: E402

import json
import sys
from pathlib import Path
from typing import List

# Ensure 'src' is in sys.path when invoked directly
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    resolve_project_path,
)
from agilab_lib.utils.trajectory import extract_keyframe_indices


def load_video_frames(
    video_path: Path,
    img_height: int = 108,
    img_width: int = 192,
) -> List[torch.Tensor]:
    """Read, resize, and normalize frames from a video file into tensors.

    Args:
        video_path: Path to target video file.
        img_height: Desired frame height.
        img_width: Desired frame width.

    Returns:
        List of normalized frame tensors of shape (3, H, W).

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    frames: List[torch.Tensor] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (img_width, img_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype("float32") / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).contiguous()
        frames.append(frame_tensor)
    cap.release()
    return frames


def extract_keyframes(cfg: DictConfig) -> Path:
    """Extract keyframe indices from video and save JSON output.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        Path to the saved keyframe JSON file.

    Raises:
        FileNotFoundError: If video file is missing and dummy fallback is disabled.
    """
    print("Executing Keyframe extraction pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=int(cfg.latent_dim))

    if cfg.get("vae_checkpoint") and str(cfg.vae_checkpoint).strip():
        vae_ckpt_path = resolve_project_path(str(cfg.vae_checkpoint))
        if vae_ckpt_path.exists():
            state_dict = torch.load(
                vae_ckpt_path, map_location=device, weights_only=True
            )
            vae.load_state_dict(state_dict)
            print(f"Loaded VAE checkpoint from: {vae_ckpt_path}")

    video_path = resolve_project_path(str(cfg.video_path))
    if video_path.exists():
        frames = load_video_frames(video_path)
        print(f"Loaded real video from {video_path} with {len(frames)} frames.")
    elif cfg.use_dummy_if_missing:
        frames = [torch.rand(3, 108, 192, dtype=torch.float32) for _ in range(30)]
        print(f"Video file not found. Using dummy dataset with {len(frames)} frames.")
    else:
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    keyframe_indices = extract_keyframe_indices(
        vae=vae,
        frames=frames,
        tau=float(cfg.tau),
        device=device,
    )

    output_data = {
        "video_path": str(video_path),
        "tau": float(cfg.tau),
        "total_frames": len(frames),
        "keyframe_count": len(keyframe_indices),
        "keyframe_indices": keyframe_indices,
    }

    output_path = ensure_writable_output_path(str(cfg.output_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully extracted {len(keyframe_indices)} keyframes.")
    print(f"Keyframe indices saved to: {output_path}")
    return output_path


@hydra.main(
    version_base=None, config_path="../configs", config_name="extract_keyframes"
)
def main(cfg: DictConfig) -> None:
    """Entry point for Keyframe extraction script.

    Args:
        cfg: Hydra configuration dictionary.
    """
    extract_keyframes(cfg)


if __name__ == "__main__":
    main()

"""
Module: extract_keyframes
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Keyframe extraction from video via Latent Vector distances.
"""

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.datasets.video_dataset import (
    DummyVideoDataset,
    VideoDataset,
)
from agilab_lib.models.vae import VAE
from agilab_lib.utils.keyframes import extract_keyframe_indices


@hydra.main(
    version_base=None, config_path="../configs", config_name="extract_keyframes"
)
def main(cfg: DictConfig) -> None:
    """Entry point for Keyframe extraction script.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("Executing Keyframe extraction pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim=int(cfg.latent_dim))

    if cfg.vae_checkpoint and os.path.exists(cfg.vae_checkpoint):
        state_dict = torch.load(cfg.vae_checkpoint, map_location=device)
        vae.load_state_dict(state_dict)
        print(f"Loaded VAE checkpoint from: {cfg.vae_checkpoint}")

    if os.path.exists(cfg.video_path):
        dataset = VideoDataset(cfg.video_path)
        print(f"Loaded real video from {cfg.video_path} with {len(dataset)} frames.")
    elif cfg.use_dummy_if_missing:
        dataset = DummyVideoDataset(num_frames=30)
        print(f"Video file not found. Using dummy dataset with {len(dataset)} frames.")
    else:
        raise FileNotFoundError(f"Video file not found at: {cfg.video_path}")

    frames = [dataset[i] for i in range(len(dataset))]
    keyframe_indices = extract_keyframe_indices(
        vae=vae,
        frames=frames,
        tau=float(cfg.tau),
        device=device,
    )

    output_data = {
        "video_path": str(cfg.video_path),
        "tau": float(cfg.tau),
        "total_frames": len(frames),
        "keyframe_count": len(keyframe_indices),
        "keyframe_indices": keyframe_indices,
    }

    output_path = str(cfg.output_json)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully extracted {len(keyframe_indices)} keyframes.")
    print(f"Keyframe indices saved to: {output_path}")


if __name__ == "__main__":
    main()

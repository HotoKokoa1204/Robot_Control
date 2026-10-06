"""
Module: extract_keyframes
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Keyframe extraction from video via Latent Vector distances.
"""

import json
from pathlib import Path

import hydra
import torch
from agilab_lib.datasets.video_dataset import (
    DummyVideoDataset,
    VideoDataset,
)
from agilab_lib.models.vae import VAE
from agilab_lib.utils.keyframes import extract_keyframe_indices
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    resolve_project_path,
)
from omegaconf import DictConfig, OmegaConf


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
        dataset = VideoDataset(video_path)
        print(f"Loaded real video from {video_path} with {len(dataset)} frames.")
    elif cfg.use_dummy_if_missing:
        dataset = DummyVideoDataset(num_frames=30)
        print(f"Video file not found. Using dummy dataset with {len(dataset)} frames.")
    else:
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    frames = [dataset[i] for i in range(len(dataset))]
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

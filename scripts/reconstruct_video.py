"""Module: reconstruct_video
Stage: Script
Author: KafuuChino
Date: 2026-09-08
Description: Evaluates VAE by encoding and decoding a real video frame-by-frame.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import resolve_project_path


def build_side_by_side_frame(
    original_bgr: np.ndarray,
    reconstructed_bgr: np.ndarray,
    scale: int = 2,
) -> np.ndarray:
    """Builds a labeled side-by-side composite frame.

    Args:
        original_bgr: Original frame in BGR uint8 format (H, W, 3).
        reconstructed_bgr: Reconstructed frame in BGR uint8 format (H, W, 3).
        scale: Upscale factor for visualization clarity.

    Returns:
        Horizontal composite frame of shape (H * scale, W * scale * 2, 3).
    """
    h, w, _ = original_bgr.shape
    orig_scaled = cv2.resize(
        original_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
    )
    recon_scaled = cv2.resize(
        reconstructed_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
    )

    # Add text labels
    cv2.putText(
        orig_scaled,
        "Original",
        (10, 25 * scale // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * (scale / 2),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        recon_scaled,
        "VAE Reconstructed",
        (10, 25 * scale // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * (scale / 2),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.hstack([orig_scaled, recon_scaled])


def reconstruct_video(cfg: DictConfig) -> Tuple[Path, Path]:
    """Evaluates VAE reconstruction quality on a target video file.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        Tuple of (output_video_path, sample_frames_dir).

    Raises:
        FileNotFoundError: If checkpoint or video file does not exist.
        ValueError: If unable to open video stream.
    """
    print("Executing video reconstruction evaluation with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load VAE model
    vae = VAE(latent_dim=int(cfg.latent_dim)).to(device)
    ckpt_path = resolve_project_path(str(cfg.vae_checkpoint))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    vae.load_state_dict(state_dict)
    vae.eval()
    print(f"Loaded VAE checkpoint from: {ckpt_path}")

    # 2. Open target video
    video_path = resolve_project_path(str(cfg.video_path))
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source video: {video_path.name} ({total_frames} frames)")

    width = int(cfg.img_width)
    height = int(cfg.img_height)
    scale = 2
    out_w = width * scale * 2
    out_h = height * scale

    output_video_path = resolve_project_path(str(cfg.output_video))
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path), fourcc, int(cfg.fps), (out_w, out_h)
    )

    sample_dir = resolve_project_path(str(cfg.sample_frames_dir))
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_frame_indices = {
        0,
        total_frames // 4,
        total_frames // 2,
        (3 * total_frames) // 4,
    }

    frame_idx = 0
    saved_samples: List[Tuple[int, Path]] = []

    print("Reconstructing frames through VAE (encode -> decode)...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize to model input resolution
        orig_small_bgr = cv2.resize(
            frame_bgr, (width, height), interpolation=cv2.INTER_AREA
        )
        orig_rgb = cv2.cvtColor(orig_small_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to float tensor (1, 3, H, W)
        x = (
            torch.from_numpy(orig_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
            / 255.0
        )

        with torch.no_grad():
            latent = vae.get_latent(x)
            recon = vae.decode(latent)

        recon_tensor = recon[0].cpu().clamp(0.0, 1.0)
        recon_np = (recon_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

        composite = build_side_by_side_frame(orig_small_bgr, recon_bgr, scale=scale)
        writer.write(composite)

        if frame_idx in sample_frame_indices:
            sample_path = sample_dir / f"sample_frame_{frame_idx:04d}.png"
            cv2.imwrite(str(sample_path), composite)
            saved_samples.append((frame_idx, sample_path))

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"  Processed frame {frame_idx}/{total_frames}...", flush=True)

    cap.release()
    writer.release()
    print(f"Reconstructed video saved to: {output_video_path}")
    print(f"Sample images saved: {len(saved_samples)} images in {sample_dir}")
    return output_video_path, sample_dir


@hydra.main(
    version_base=None, config_path="../configs", config_name="reconstruct_video"
)
def main(cfg: DictConfig) -> None:
    """Evaluates VAE reconstruction quality on a target video file.

    Args:
        cfg: Hydra configuration dictionary.
    """
    reconstruct_video(cfg)


if __name__ == "__main__":
    main()

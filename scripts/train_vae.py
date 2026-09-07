"""
Module: train_vae
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train VAE for Latent Vector representation learning on real videos.
"""

from pathlib import Path
from typing import List

import cv2
import hydra
import torch
from agilab_lib.models.vae import VAE, vae_loss
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class RealVideoFramesDataset(Dataset[torch.Tensor]):
    """Extracts and normalizes frames from a collection of video files."""

    def __init__(
        self,
        video_paths: List[Path],
        frameskip: int = 1,
        img_width: int = 192,
        img_height: int = 108,
    ) -> None:
        """Initialize video frames dataset.

        Args:
            video_paths: List of file paths to source videos.
            frameskip: Frame skipping step size.
            img_width: Target frame width in pixels.
            img_height: Target frame height in pixels.
        """
        super().__init__()
        self.frames: List[torch.Tensor] = []

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                continue

            frame_idx = 0
            ret, frame = cap.read()
            while ret:
                if frame_idx % frameskip == 0:
                    frame_resized = cv2.resize(
                        frame, (img_width, img_height), interpolation=cv2.INTER_AREA
                    )
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_t = (
                        torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                    )
                    self.frames.append(frame_t)

                frame_idx += 1
                ret, frame = cap.read()

            cap.release()

    def __len__(self) -> int:
        """Return the total number of extracted video frames.

        Returns:
            Count of frames stored in dataset.
        """
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Fetch a single normalized frame tensor.

        Args:
            idx: Index of frame.

        Returns:
            Normalized frame tensor of shape (3, H, W).
        """
        return self.frames[idx]


def discover_video_files(cfg: DictConfig) -> List[Path]:
    """Discovers source video paths matching category settings.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        List of matching video file paths.
    """
    root = Path(str(cfg.data_root))
    if not root.exists():
        return []

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    selected_videos: List[Path] = []
    max_per_cat = (
        int(cfg.max_videos_per_category)
        if cfg.max_videos_per_category is not None
        else None
    )

    for cat in cfg.categories:
        cat_dir = root / str(cat)
        if not cat_dir.exists():
            continue

        cat_videos = sorted(
            [p for p in cat_dir.rglob("*") if p.suffix.lower() in video_exts]
        )
        if max_per_cat is not None:
            cat_videos = cat_videos[:max_per_cat]

        selected_videos.extend(cat_videos)

    return selected_videos


@hydra.main(version_base=None, config_path="../configs", config_name="train_vae")
def main(cfg: DictConfig) -> None:
    """Entry point for VAE training on real video datasets.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("Executing VAE training pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_paths = discover_video_files(cfg)
    print(f"Discovered {len(video_paths)} video files across categories.")
    for vp in video_paths:
        print(f"  - {vp}")

    dataset = RealVideoFramesDataset(
        video_paths=video_paths,
        frameskip=int(cfg.frameskip),
        img_width=int(cfg.img_width),
        img_height=int(cfg.img_height),
    )
    print(f"Extracted {len(dataset)} training frames in total.")

    if len(dataset) == 0:
        raise RuntimeError("No frames extracted from specified video paths.")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )

    model = VAE(latent_dim=int(cfg.latent_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    print(f"Starting VAE training on {device}...")
    model.train()

    beta = float(cfg.beta_kl)
    for epoch in range(int(cfg.max_epochs)):
        total_loss_accum = 0.0
        recon_loss_accum = 0.0
        kl_loss_accum = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()
            recon_loss_accum += recon_loss.item()
            kl_loss_accum += kl_loss.item()

        n_batches = max(len(loader), 1)
        avg_total = total_loss_accum / n_batches
        avg_recon = recon_loss_accum / n_batches
        avg_kl = kl_loss_accum / n_batches
        print(
            f"Epoch {epoch + 1}/{cfg.max_epochs} - "
            f"Loss: {avg_total:.6f} "
            f"(Recon MSE: {avg_recon:.6f}, KL: {avg_kl:.6f})"
        )

    # Save trained checkpoint
    out_path = Path(str(cfg.output_checkpoint))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"VAE checkpoint successfully saved to: {out_path}")


if __name__ == "__main__":
    main()

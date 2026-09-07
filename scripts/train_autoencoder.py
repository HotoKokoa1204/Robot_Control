"""
Module: train_autoencoder
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Autoencoder for Latent Vector representation learning on real videos.
"""

from pathlib import Path
from typing import List

import cv2
import hydra
import torch
import torch.nn.functional as F
from agilab_lib.models.autoencoder import Autoencoder
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


class RealVideoFramesDataset(Dataset[torch.Tensor]):
    """Dataset extracting normalized frame tensors from real video files."""

    def __init__(
        self,
        video_paths: List[Path],
        frameskip: int = 4,
        img_width: int = 192,
        img_height: int = 108,
    ) -> None:
        """Initialize real video frames dataset.

        Args:
            video_paths: List of paths to video files.
            frameskip: Frame skipping step size.
            img_width: Width to resize frames.
            img_height: Height to resize frames.
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
                    frame_tensor = (
                        torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                    )
                    self.frames.append(frame_tensor)

                frame_idx += 1
                ret, frame = cap.read()

            cap.release()

    def __len__(self) -> int:
        """Return total number of extracted frames.

        Returns:
            Frame count.
        """
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Fetch frame tensor at index.

        Args:
            idx: Frame index.

        Returns:
            RGB tensor of shape (3, H, W).
        """
        return self.frames[idx]


def discover_video_files(cfg: DictConfig) -> List[Path]:
    """Find video files in specified categories according to configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        List of discovered video file paths.
    """
    root = Path(str(cfg.data_root))
    selected_videos: List[Path] = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

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


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_autoencoder"
)
def main(cfg: DictConfig) -> None:
    """Entry point for Autoencoder training on real video datasets.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("Executing Autoencoder training pipeline with config:")
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

    model = Autoencoder(latent_dim=int(cfg.latent_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    print(f"Starting Autoencoder training on {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(
            f"Epoch {epoch + 1}/{cfg.max_epochs} - "
            f"MSE Reconstruction Loss: {avg_loss:.6f}"
        )

    # Save trained checkpoint
    out_path = Path(str(cfg.output_checkpoint))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Autoencoder checkpoint successfully saved to: {out_path}")


if __name__ == "__main__":
    main()

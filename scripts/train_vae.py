"""Module: train_vae
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train VAE for Latent Vector representation learning on real videos.
"""

import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from agilab_lib.models.vae import VAE, vae_loss
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    resolve_project_path,
)


def _extract_video_frames(args: Tuple[str, int, int, int]) -> np.ndarray:
    """Extracts resized uint8 RGB frames from a single video file.

    Args:
        args: Tuple of (video_path, frameskip, img_width, img_height).

    Returns:
        Numpy array of shape (N, H, W, 3) with uint8 dtype.
    """
    video_path, frameskip, img_width, img_height = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, img_height, img_width, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    frame_idx = 0
    ret, frame = cap.read()
    while ret:
        if frame_idx % frameskip == 0:
            frame_resized = cv2.resize(
                frame, (img_width, img_height), interpolation=cv2.INTER_AREA
            )
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
        ret, frame = cap.read()

    cap.release()
    if frames:
        return np.stack(frames, axis=0)
    return np.empty((0, img_height, img_width, 3), dtype=np.uint8)


class RealVideoFramesDataset(Dataset[torch.Tensor]):
    """Extracts, caches, and yields normalized frames from video files."""

    def __init__(
        self,
        video_paths: List[Path],
        frameskip: int = 1,
        img_width: int = 192,
        img_height: int = 108,
        cache_path: Optional[str] = None,
        num_workers: int = 4,
    ) -> None:
        """Initialize video frames dataset.

        Args:
            video_paths: List of file paths to source videos.
            frameskip: Frame skipping step size.
            img_width: Target frame width in pixels.
            img_height: Target frame height in pixels.
            cache_path: Optional path to save/load pre-extracted numpy frames.
            num_workers: Number of parallel worker processes for extraction.
        """
        super().__init__()
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached frames from {cache_path}...")
            self.frames: np.ndarray = np.load(cache_path)
            print(f"Loaded {len(self.frames)} frames from cache.")
            return

        print(
            f"Extracting frames from {len(video_paths)} videos "
            f"(workers={num_workers})..."
        )
        tasks = [(str(vp), frameskip, img_width, img_height) for vp in video_paths]
        extracted_arrays: List[np.ndarray] = []
        with mp.Pool(processes=num_workers) as pool:
            for idx, arr in enumerate(pool.imap(_extract_video_frames, tasks)):
                if len(arr) > 0:
                    extracted_arrays.append(arr)
                if (idx + 1) % 20 == 0 or (idx + 1) == len(tasks):
                    print(
                        f"  Processed {idx + 1}/{len(tasks)} videos...",
                        flush=True,
                    )

        if extracted_arrays:
            self.frames = np.concatenate(extracted_arrays, axis=0)
        else:
            self.frames = np.empty((0, img_height, img_width, 3), dtype=np.uint8)

        print(f"Extracted total of {len(self.frames)} frames.")

        if cache_path and len(self.frames) > 0:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.frames)
            print(f"Saved dataset cache to: {cache_path}")

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
        frame_np = self.frames[idx]
        return torch.from_numpy(frame_np).float().permute(2, 0, 1) / 255.0


def discover_video_files(cfg: DictConfig) -> List[Path]:
    """Discovers source video paths matching category settings.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        List of matching video file paths.
    """
    root = resolve_project_path(str(cfg.data_root))
    if not root.exists():
        return []

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    selected_videos: List[Path] = []
    max_per_cat = (
        int(cfg.max_videos_per_category)
        if cfg.max_videos_per_category is not None
        else None
    )
    one_per_sub_cats = set(cfg.get("one_per_subfolder_categories", []))

    for cat in cfg.categories:
        cat_dir = root / str(cat)
        if not cat_dir.exists():
            continue

        if str(cat) in one_per_sub_cats:
            subdirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
            cat_videos: List[Path] = []
            for d in subdirs:
                vids = sorted(
                    [p for p in d.rglob("*") if p.suffix.lower() in video_exts]
                )
                if vids:
                    cat_videos.append(vids[0])
        else:
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
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    video_paths = discover_video_files(cfg)
    print(f"Discovered {len(video_paths)} video files across categories.")

    cache_path = (
        str(resolve_project_path(str(cfg.cache_path)))
        if cfg.get("cache_path")
        else None
    )
    dataset = RealVideoFramesDataset(
        video_paths=video_paths,
        frameskip=int(cfg.frameskip),
        img_width=int(cfg.img_width),
        img_height=int(cfg.img_height),
        cache_path=cache_path,
        num_workers=4,
    )
    print(f"Total training frames in dataset: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("No frames extracted from specified video paths.")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )

    model = VAE(latent_dim=int(cfg.latent_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    print(f"Starting VAE training for {cfg.max_epochs} epochs on {device}...")
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
        if (epoch + 1) % 5 == 0 or (epoch + 1) == int(cfg.max_epochs):
            print(
                f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - "
                f"Loss: {avg_total:.6f} "
                f"(Recon MSE: {avg_recon:.6f}, KL: {avg_kl:.6f})",
                flush=True,
            )

    # Save trained checkpoint
    out_path = ensure_writable_output_path(str(cfg.output_checkpoint))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"VAE checkpoint successfully saved to: {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

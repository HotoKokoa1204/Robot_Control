"""Module: train_vae
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train VAE for Latent Vector representation learning on real videos.
"""

# ruff: noqa: E402

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure 'src' is in sys.path when invoked directly
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from agilab_lib.models.vae import VAE, vae_loss
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    get_eval_dir,
    get_plots_dir,
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
        0.5 * scale,
        (0, 255, 0),
        scale,
    )
    cv2.putText(
        recon_scaled,
        "Reconstructed VAE",
        (10, 25 * scale // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * scale,
        (0, 200, 255),
        scale,
    )

    return np.concatenate([orig_scaled, recon_scaled], axis=1)


def plot_vae_loss_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plots training loss curves and saves figure to disk.

    Args:
        history: Dictionary containing loss trajectories across epochs.
        output_path: Destination path for the saved plot image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epoch", list(range(1, len(history["total_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, history["total_loss"], label="Total Loss", color="#1f77b4", lw=2)
    ax1.plot(
        epochs,
        history["recon_mse"],
        label="Reconstruction MSE",
        color="#2ca02c",
        lw=2,
        linestyle="--",
    )
    ax1.set_ylabel("Reconstruction Loss", fontsize=12)
    ax1.set_title(
        "VAE Training Loss Progression (512-dim Latent)", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(
        epochs,
        history["kl_loss"],
        label="KL Divergence (summed)",
        color="#d62728",
        lw=2,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("KL Loss", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Loss curves saved to: {output_path}")


def evaluate_visual_reconstruction(
    model: VAE,
    video_path: Path,
    device: torch.device,
    img_width: int = 192,
    img_height: int = 108,
) -> Tuple[Path, Path, float]:
    """Generates side-by-side reconstruction video and 9-panel image comparison.

    Args:
        model: Trained VAE model.
        video_path: Path to evaluation video file.
        device: Torch execution device.
        img_width: Target image width.
        img_height: Target image height.

    Returns:
        Tuple of (video_output_path, panel_output_path, avg_std).
    """
    model.eval()
    cap = cv2.VideoCapture(str(video_path))
    orig_bgr_frames: List[np.ndarray] = []
    recon_bgr_frames: List[np.ndarray] = []
    recon_stds: List[float] = []

    ret, frame = cap.read()
    with torch.no_grad():
        while ret:
            resized = cv2.resize(
                frame, (img_width, img_height), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            recon_tensor, _, _ = model(tensor, deterministic=True)
            recon_rgb = (
                recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
                * 255.0
            ).astype(np.uint8)
            recon_bgr = cv2.cvtColor(recon_rgb, cv2.COLOR_RGB2BGR)

            orig_bgr_frames.append(resized)
            recon_bgr_frames.append(recon_bgr)
            recon_stds.append(float(np.std(recon_rgb)))

            ret, frame = cap.read()
    cap.release()

    avg_std = float(np.mean(recon_stds)) if recon_stds else 0.0

    # 1. Export Side-by-Side Video
    eval_dir = get_eval_dir()
    eval_dir.mkdir(parents=True, exist_ok=True)
    video_out = eval_dir / "vae_recon_eval.mp4"

    scale = 2
    out_h = img_height * scale
    out_w = img_width * scale * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, 30.0, (out_w, out_h))

    for orig_bgr, recon_bgr in zip(orig_bgr_frames, recon_bgr_frames):
        composite = build_side_by_side_frame(orig_bgr, recon_bgr, scale=scale)
        writer.write(composite)
    writer.release()
    print(f"Evaluation reconstruction video saved to: {video_out}")

    # 2. Export 9-Frame Comparison Panel
    panel_out = eval_dir / "vae_recon_comparison_panel.png"
    n_total = len(orig_bgr_frames)
    if n_total >= 9:
        indices = np.linspace(0, n_total - 1, 9, dtype=int)
    else:
        indices = np.arange(n_total, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        f"VAE Visual Reconstruction Samples (Avg Std: {avg_std:.2f})",
        fontsize=16,
        fontweight="bold",
    )

    for i, idx in enumerate(indices[:9]):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        orig_rgb = cv2.cvtColor(orig_bgr_frames[idx], cv2.COLOR_BGR2RGB)
        recon_rgb = cv2.cvtColor(recon_bgr_frames[idx], cv2.COLOR_BGR2RGB)

        # Concatenate horizontally
        pair = np.concatenate([orig_rgb, recon_rgb], axis=1)
        ax.imshow(pair)
        ax.set_title(f"Frame #{idx} (Orig | Recon)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(panel_out, dpi=150)
    plt.close(fig)
    print(f"Evaluation comparison panel saved to: {panel_out}")

    return video_out, panel_out, avg_std


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
    history: Dict[str, List[float]] = {
        "epoch": [],
        "total_loss": [],
        "recon_mse": [],
        "kl_loss": [],
    }

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

        history["epoch"].append(epoch + 1)
        history["total_loss"].append(avg_total)
        history["recon_mse"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == int(cfg.max_epochs):
            print(
                f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - "
                f"Loss: {avg_total:.6f} "
                f"(Recon MSE: {avg_recon:.6f}, KL: {avg_kl:.6f})",
                flush=True,
            )

    # 1. Save trained checkpoint
    out_path = ensure_writable_output_path(str(cfg.output_checkpoint))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"VAE checkpoint successfully saved to: {out_path}")

    # 2. Save loss history JSON & plot curves
    plots_dir = get_plots_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = plots_dir / "vae_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    plot_vae_loss_curves(history, plots_dir / "vae_loss_curves.png")

    # 3. Evaluate visual reconstruction on the first video
    if video_paths:
        eval_video = video_paths[0]
        print(f"Evaluating visual reconstruction on: {eval_video}")
        video_out, panel_out, avg_std = evaluate_visual_reconstruction(
            model=model,
            video_path=eval_video,
            device=device,
            img_width=int(cfg.img_width),
            img_height=int(cfg.img_height),
        )

        min_std = 15.0
        if avg_std >= min_std:
            print(
                f"[PASSED] Sanity Gate: No posterior collapse detected "
                f"(avg_std = {avg_std:.2f} >= {min_std:.1f})"
            )
        else:
            print(
                f"[WARNING] Potential posterior collapse: low pixel variance "
                f"(avg_std = {avg_std:.2f} < {min_std:.1f})"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()

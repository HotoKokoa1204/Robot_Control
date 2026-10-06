"""
Module: extract_latents
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Extract and cache Latent Vectors from videos via VAE.
"""

from pathlib import Path
from typing import Dict, List, Union

import cv2
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import resolve_project_path


def extract_video_latents(
    video_path: Path,
    model: VAE,
    device: torch.device,
    frameskip: int = 1,
    img_width: int = 192,
    img_height: int = 108,
    batch_size: int = 32,
) -> Dict[str, Union[torch.Tensor, int, float, str]]:
    """Extract and serialize Latent Vectors from a video file.

    Args:
        video_path: Path to video file.
        model: Trained VAE model.
        device: Torch compute device.
        frameskip: Frame skipping step size.
        img_width: Frame resize width.
        img_height: Frame resize height.
        batch_size: Encoding batch size.

    Returns:
        Dictionary containing extracted latent tensor and metadata.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames_batch: List[torch.Tensor] = []
    latents_list: List[torch.Tensor] = []

    frame_idx = 0
    ret, frame = cap.read()

    while ret:
        if frame_idx % frameskip == 0:
            frame_resized = cv2.resize(
                frame, (img_width, img_height), interpolation=cv2.INTER_AREA
            )
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
            frames_batch.append(frame_t)

            if len(frames_batch) >= batch_size:
                batch_tensor = torch.stack(frames_batch).to(device)
                with torch.no_grad():
                    z = model.get_latent(batch_tensor).cpu()
                latents_list.append(z)
                frames_batch.clear()

        frame_idx += 1
        ret, frame = cap.read()

    cap.release()

    if frames_batch:
        batch_tensor = torch.stack(frames_batch).to(device)
        with torch.no_grad():
            z = model.get_latent(batch_tensor).cpu()
        latents_list.append(z)

    cat_latents = (
        torch.cat(latents_list, dim=0)
        if latents_list
        else torch.empty((0, model.latent_dim))
    )

    return {
        "latents": cat_latents,
        "total_frames": total_frames,
        "fps": fps,
        "video_name": video_path.stem,
    }


def extract_latents(cfg: DictConfig) -> Path:
    """Extract and cache Latent Vectors from videos via VAE.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        Path to the output directory containing extracted latents.

    Raises:
        FileNotFoundError: If VAE checkpoint is not found.
    """
    print("Executing Latent Vector offline extraction with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=int(cfg.latent_dim)).to(device)
    vae_ckpt = resolve_project_path(str(cfg.vae_checkpoint))
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")

    model.load_state_dict(torch.load(vae_ckpt, map_location=device))
    model.eval()
    print(f"Loaded VAE checkpoint from: {vae_ckpt}")

    root = resolve_project_path(str(cfg.data_root))
    out_root = resolve_project_path(str(cfg.output_dir))
    out_root.mkdir(parents=True, exist_ok=True)
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

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

        cat_out = out_root / str(cat)
        cat_out.mkdir(parents=True, exist_ok=True)

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

        print(f"\nProcessing category [{cat}]: {len(cat_videos)} videos to extract.")

        for vp in cat_videos:
            out_name = (
                f"{vp.parent.name}_{vp.stem}.pt"
                if vp.parent != cat_dir
                else f"{vp.stem}.pt"
            )
            out_file = cat_out / out_name
            print(f"  Encoding {vp.parent.name}/{vp.name} -> {out_file.name}...")
            record = extract_video_latents(
                video_path=vp,
                model=model,
                device=device,
                frameskip=int(cfg.frameskip),
                img_width=int(cfg.img_width),
                img_height=int(cfg.img_height),
                batch_size=int(cfg.batch_size),
            )
            torch.save(record, out_file)
            print(
                f"    Extracted {len(record['latents'])} latents of shape "
                f"{record['latents'].shape} "
                f"(total frames: {record['total_frames']})"
            )

    print("\nOffline Latent Vector extraction completed successfully.")
    return out_root


@hydra.main(version_base=None, config_path="../configs", config_name="extract_latents")
def main(cfg: DictConfig) -> None:
    """Entry point for offline Latent Vector extraction script.

    Args:
        cfg: Hydra configuration dictionary.
    """
    extract_latents(cfg)


if __name__ == "__main__":
    main()

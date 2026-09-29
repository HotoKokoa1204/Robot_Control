"""
Module: train_multibranch
Stage: Script
Author: KafuuChino
Date: 2026-09-29
Description: Joint multi-branch training script for end-to-end latent navigation
    learning with AMP, gradient accumulation, and standalone weight exports.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Windows DLL initialization guard for PIL/torchvision
from PIL import Image  # isort: skip # noqa: F401

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Ensure 'src' is in sys.path when invoked directly
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tqdm import tqdm  # noqa: E402

from agilab_lib.datasets.dual_source_dataset import (  # noqa: E402
    DualSourceBatch,
    DualSourceVideoDataset,
    DummyDualSourceVideoDataset,
    dual_source_collate_fn,
)
from agilab_lib.models.joint_loss import JointNavigationLoss  # noqa: E402
from agilab_lib.models.joint_navigation import JointNavigationModel  # noqa: E402


def get_dataset(cfg: DictConfig) -> Dataset[DualSourceBatch]:
    """Build or retrieve the multi-branch video training dataset.

    Attempts to load real video pairs from the configured directories.
    If the video directories do not exist or contain no valid video files,
    falls back to DummyDualSourceVideoDataset if use_dummy_if_missing is True.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        Dataset yielding DualSourceBatch instances.

    Raises:
        FileNotFoundError: If data directories are missing and
            use_dummy_if_missing is False.
        ValueError: If directories contain no videos and
            use_dummy_if_missing is False.
    """
    one_path_dir = DualSourceVideoDataset._resolve_dir(
        cfg.get("one_path_dir", "data/one_path")
    )
    rotation_dir = DualSourceVideoDataset._resolve_dir(
        cfg.get("rotation_dir", "data/360")
    )
    use_dummy = bool(cfg.get("use_dummy_if_missing", True))
    recon_source = str(cfg.get("recon_source", "random"))

    if one_path_dir.exists() and rotation_dir.exists():
        try:
            num_samples_cfg = cfg.get("num_samples", None)
            num_samples_val = (
                int(num_samples_cfg) if num_samples_cfg is not None else None
            )
            max_vids_cfg = cfg.get("max_videos_per_source", None)
            max_vids_val = int(max_vids_cfg) if max_vids_cfg is not None else None
            samples_per_vid_cfg = cfg.get("samples_per_video", 50)
            samples_per_vid_val = (
                int(samples_per_vid_cfg) if samples_per_vid_cfg is not None else None
            )

            dataset = DualSourceVideoDataset(
                one_path_dir=one_path_dir,
                rotation_dir=rotation_dir,
                img_height=int(cfg.get("img_height", 108)),
                img_width=int(cfg.get("img_width", 192)),
                video_fps=float(cfg.get("video_fps", 60.0)),
                straight_video_speed_mps=float(
                    cfg.get("straight_video_speed_mps", 2.5)
                ),
                max_videos_per_source=max_vids_val,
                samples_per_video=samples_per_vid_val,
                num_samples=num_samples_val,
                recon_source=recon_source,
                rotation_one_per_subfolder=bool(
                    cfg.get("rotation_one_per_subfolder", True)
                ),
            )
            if len(dataset) > 0:
                print(
                    f"Loaded DualSourceVideoDataset with {len(dataset)} paired samples."
                )
                return dataset
        except (FileNotFoundError, ValueError) as exc:
            if not use_dummy:
                raise
            print(
                f"Notice: Cannot initialize DualSourceVideoDataset ({exc}). "
                "Falling back to DummyDualSourceVideoDataset."
            )
    else:
        if not use_dummy:
            raise FileNotFoundError(
                f"Missing data directories: {one_path_dir} or {rotation_dir}"
            )
        print(
            "Notice: Video data directories not found. "
            "Falling back to DummyDualSourceVideoDataset."
        )

    print("Instantiating DummyDualSourceVideoDataset for training.")
    return DummyDualSourceVideoDataset(
        num_samples=int(cfg.get("num_samples", 64)),
        img_height=int(cfg.get("img_height", 108)),
        img_width=int(cfg.get("img_width", 192)),
        recon_source=recon_source,
        seed=cfg.get("seed", 42),
    )


def save_all_checkpoints(
    model: JointNavigationModel,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save composite and exported standalone checkpoints.

    Saves:
    1. Composite model weights: joint_navigation_model.pt
    2. Standalone VAE weights: vae_512.pt
    3. Standalone forward transformer weights: forward_transformer.pt
    4. Standalone rotation transformer weights: rotation_transformer.pt

    Args:
        model: JointNavigationModel instance.
        output_dir: Destination folder path for saved checkpoints.

    Returns:
        Dictionary mapping model component names to their saved checkpoint file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    composite_path = out_path / "joint_navigation_model.pt"
    torch.save(model.state_dict(), composite_path)

    vae_path = out_path / "vae_512.pt"
    model.export_vae_checkpoint(vae_path)

    fwd_path = out_path / "forward_transformer.pt"
    model.export_forward_checkpoint(fwd_path)

    rot_path = out_path / "rotation_transformer.pt"
    model.export_rotation_checkpoint(rot_path)

    print(f"Saved checkpoints successfully to {out_path}:")
    print(f"  - Composite: {composite_path.name}")
    print(f"  - Standalone VAE: {vae_path.name}")
    print(f"  - Standalone Forward: {fwd_path.name}")
    print(f"  - Standalone Rotation: {rot_path.name}")

    return {
        "composite": composite_path,
        "vae": vae_path,
        "forward": fwd_path,
        "rotation": rot_path,
    }


def train_multibranch(
    cfg: DictConfig,
    dataset: Optional[Dataset[DualSourceBatch]] = None,
    model: Optional[JointNavigationModel] = None,
    loss_fn: Optional[JointNavigationLoss] = None,
) -> Tuple[JointNavigationModel, Dict[str, List[float]]]:
    """Execute end-to-end multi-branch training loop with AMP and gradient accumulation.

    Args:
        cfg: Configuration parameters dictionary.
        dataset: Optional pre-constructed dataset instance.
        model: Optional pre-constructed JointNavigationModel instance.
        loss_fn: Optional pre-constructed JointNavigationLoss instance.

    Returns:
        Tuple of (trained_model, history_metrics_dict).

    Raises:
        ValueError: If training dataset is empty.
    """
    # 1. Device configuration
    device_cfg = cfg.get("device", None)
    if device_cfg is not None:
        device = torch.device(str(device_cfg))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing multi-branch training on device: {device}")

    # 2. Dataset and DataLoader setup
    if dataset is None:
        dataset = get_dataset(cfg)

    if len(dataset) == 0:
        raise ValueError("Training dataset contains 0 samples.")

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dual_source_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Model instantiation and optional warm-start
    latent_dim = int(cfg.get("latent_dim", 512))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    num_blocks = int(cfg.get("num_blocks", 5))
    block_inner_dim = int(cfg.get("block_inner_dim", hidden_dim))

    if model is None:
        model = JointNavigationModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            block_inner_dim=block_inner_dim,
        )

        vae_ckpt = cfg.get("vae_checkpoint", None)
        if (
            vae_ckpt is not None
            and str(vae_ckpt).strip() != ""
            and str(vae_ckpt).lower() != "none"
            and str(vae_ckpt).lower() != "null"
        ):
            vae_path = DualSourceVideoDataset._resolve_dir(str(vae_ckpt))
            if vae_path.exists():
                model.load_vae_pretrained(vae_path)
                print(f"Warm-started VAE from checkpoint: {vae_path}")
            else:
                print(
                    f"Warning: VAE checkpoint '{vae_path}' specified but not found. "
                    "Continuing from scratch."
                )

    model.to(device)

    # 4. Joint Loss instantiation
    if loss_fn is None:
        alpha_perc = float(cfg.get("alpha_perc", 0.5))
        beta_kl = float(cfg.get("beta_kl", 0.0001))
        w_fwd = float(cfg.get("w_fwd", 1.0))
        w_rot = float(cfg.get("w_rot", 1.0))
        w_recon = float(cfg.get("w_recon", 1.0))
        pretrained_vgg = bool(cfg.get("pretrained_vgg", True))
        loss_fn = JointNavigationLoss(
            alpha_perc=alpha_perc,
            beta_kl=beta_kl,
            w_fwd=w_fwd,
            w_rot=w_rot,
            w_recon=w_recon,
            pretrained_vgg=pretrained_vgg,
        )

    loss_fn.to(device)

    # 5. Optimizer setup
    lr = float(cfg.get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. AMP and Gradient Accumulation setup
    use_amp = bool(cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(use_amp and device.type == "cuda")
    )
    accum_steps = max(1, int(cfg.get("gradient_accumulation_steps", 2)))

    # 7. Training loop parameters and telemetry tracking
    max_epochs = int(cfg.get("max_epochs", 50))
    save_interval = int(cfg.get("save_interval_epochs", 5))
    output_dir = DualSourceVideoDataset._resolve_dir(
        cfg.get("output_dir", "checkpoints")
    )

    history: Dict[str, List[float]] = {
        "loss": [],
        "recon": [],
        "fwd": [],
        "rot": [],
        "perc": [],
        "kl": [],
    }

    print(
        f"Starting training for {max_epochs} epochs "
        f"(batch_size={batch_size}, accum_steps={accum_steps}, "
        f"use_amp={use_amp})..."
    )

    optimizer.zero_grad()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_fwd = 0.0
        epoch_rot = 0.0
        epoch_perc = 0.0
        epoch_kl = 0.0
        num_batches = len(train_loader)

        batch_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:2d}/{max_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, raw_batch in enumerate(batch_iter):
            if isinstance(raw_batch, DualSourceBatch) or hasattr(raw_batch, "to"):
                batch = raw_batch.to(device)
            elif isinstance(raw_batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_batch.items()
                }
            else:
                batch = raw_batch

            with torch.amp.autocast(
                "cuda", enabled=bool(use_amp and device.type == "cuda")
            ):
                loss_output = loss_fn(model, batch)
                total_loss = loss_output.total_loss

            # Telemetry accumulation from unscaled loss
            epoch_loss += total_loss.item()
            epoch_recon += loss_output.metrics["loss_recon"].item()
            epoch_fwd += loss_output.metrics["loss_fwd"].item()
            epoch_rot += loss_output.metrics["loss_rot"].item()
            perc_val = (
                loss_output.metrics["recon_perc"]
                + loss_output.metrics["fwd_perc"]
                + loss_output.metrics["rot_perc"]
            ).item()
            epoch_perc += perc_val
            epoch_kl += loss_output.metrics["recon_kl"].item()

            # Gradient accumulation scaling and backward pass
            is_accum_step = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches

            rem = (batch_idx % accum_steps) + 1
            actual_steps = rem if (is_last_batch and not is_accum_step) else accum_steps
            loss_scaled = total_loss / actual_steps
            scaler.scale(loss_scaled).backward()

            if is_accum_step or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_iter.set_postfix(
                loss=f"{total_loss.item():.4f}",
                recon=f"{loss_output.metrics['loss_recon'].item():.3f}",
                fwd=f"{loss_output.metrics['loss_fwd'].item():.3f}",
                rot=f"{loss_output.metrics['loss_rot'].item():.3f}",
            )

        # Compute average metrics across all batches for the epoch
        n_b = max(num_batches, 1)
        avg_loss = epoch_loss / n_b
        avg_recon = epoch_recon / n_b
        avg_fwd = epoch_fwd / n_b
        avg_rot = epoch_rot / n_b
        avg_perc = epoch_perc / n_b
        avg_kl = epoch_kl / n_b

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["fwd"].append(avg_fwd)
        history["rot"].append(avg_rot)
        history["perc"].append(avg_perc)
        history["kl"].append(avg_kl)

        if (epoch + 1) % 1 == 0 or (epoch + 1) == max_epochs:
            print(
                f"Epoch {epoch + 1:3d}/{max_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Recon: {avg_recon:.4f} | "
                f"Fwd: {avg_fwd:.4f} | "
                f"Rot: {avg_rot:.4f} | "
                f"Perc: {avg_perc:.4f} | "
                f"KL: {avg_kl:.6f}",
                flush=True,
            )

        # Periodic checkpoint saving
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_all_checkpoints(model, output_dir)

    # Final checkpoint saving
    save_all_checkpoints(model, output_dir)

    return model, history


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_multibranch",
)
def main(cfg: DictConfig) -> None:
    """Main CLI entrypoint for multi-branch joint training script.

    Args:
        cfg: Hydra DictConfig parsed from configs/train_multibranch.yaml.
    """
    print("=== Joint Multi-Branch Training Configuration ===")
    print(OmegaConf.to_yaml(cfg))
    train_multibranch(cfg)


if __name__ == "__main__":
    main()

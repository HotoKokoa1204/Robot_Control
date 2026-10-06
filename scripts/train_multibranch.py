"""
Module: train_multibranch
Stage: Script
Author: KafuuChino
Date: 2026-09-29
Description: Joint multi-branch training script for end-to-end latent navigation
    learning with three-stage curriculum, AMP, gradient accumulation, and
    standalone weight exports.
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

            buffer_dist = float(cfg.get("buffer_distance_meters", 2.0))
            dataset = DualSourceVideoDataset(
                one_path_dir=one_path_dir,
                rotation_dir=rotation_dir,
                img_height=int(cfg.get("img_height", 108)),
                img_width=int(cfg.get("img_width", 192)),
                video_fps=float(cfg.get("video_fps", 60.0)),
                straight_video_speed_mps=float(
                    cfg.get("straight_video_speed_mps", 2.5)
                ),
                buffer_distance_meters=buffer_dist,
                max_videos_per_source=max_vids_val,
                samples_per_video=samples_per_vid_val,
                num_samples=num_samples_val,
                recon_source=recon_source,
                rotation_one_per_subfolder=bool(
                    cfg.get("rotation_one_per_subfolder", True)
                ),
                preload_frames=bool(cfg.get("preload_frames", False)),
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
        buffer_distance_meters=float(cfg.get("buffer_distance_meters", 2.0)),
        seed=cfg.get("seed", 42),
    )


def plot_loss_curves(
    history: Dict[str, List[float]],
    output_dir: Union[str, Path],
) -> Optional[Path]:
    """Plot and save training loss curves across all epochs and stages.

    Visualizes total loss, reconstruction, dynamics (forward/rotation),
    latent dynamics alignment MSE, and regularization (perceptual/KL).

    Args:
        history: Dictionary mapping metric names to lists of epoch averages.
        output_dir: Destination folder path for loss_curves.png.

    Returns:
        Path to the saved loss_curves.png plot, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    epochs_list = history.get("loss", [])
    if not epochs_list:
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plot_file = out_path / "loss_curves.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = list(range(1, len(epochs_list) + 1))

    # Subplot 1: Total loss & Recon loss
    axes[0, 0].plot(
        epochs, history.get("loss", []), label="Total Loss", color="tab:blue"
    )
    axes[0, 0].plot(
        epochs,
        history.get("recon", []),
        label="Recon Loss",
        color="tab:orange",
        linestyle="--",
    )
    axes[0, 0].set_title("Total & Reconstruction Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Dynamics losses (Fwd & Rot)
    axes[0, 1].plot(
        epochs, history.get("fwd", []), label="Forward Loss", color="tab:green"
    )
    axes[0, 1].plot(
        epochs, history.get("rot", []), label="Rotation Loss", color="tab:red"
    )
    axes[0, 1].set_title("Dynamics Losses")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Latent MSE losses
    axes[1, 0].plot(
        epochs,
        history.get("fwd_latent", []),
        label="Fwd Latent MSE",
        color="tab:cyan",
    )
    axes[1, 0].plot(
        epochs,
        history.get("rot_latent", []),
        label="Rot Latent MSE",
        color="tab:pink",
    )
    axes[1, 0].set_title("Latent Dynamics Alignment")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Regularization (Perceptual & optional KL)
    ax4 = axes[1, 1]
    p1 = ax4.plot(
        epochs,
        history.get("perc", []),
        label="Perceptual Loss",
        color="tab:purple",
        linewidth=1.8,
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Perceptual Loss", color="tab:purple")
    ax4.tick_params(axis="y", labelcolor="tab:purple")
    ax4.grid(True, alpha=0.3)

    kl_vals = history.get("kl", [])
    if any(k > 0.001 for k in kl_vals):
        ax4_twin = ax4.twinx()
        p2 = ax4_twin.plot(
            epochs,
            kl_vals,
            label="KL Divergence",
            color="tab:brown",
            linestyle="--",
        )
        ax4_twin.set_ylabel("KL Divergence", color="tab:brown")
        ax4_twin.tick_params(axis="y", labelcolor="tab:brown")
        ax4.set_title("Perceptual (Left) & KL (Right) Regularization")
        lines = p1 + p2
        labels = [handle.get_label() for handle in lines]
        ax4.legend(lines, labels, loc="upper right")
    else:
        ax4.set_title("Perceptual Regularization")
        ax4.legend(p1, ["Perceptual Loss"], loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    return plot_file


def save_all_checkpoints(
    model: JointNavigationModel,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save composite and exported standalone checkpoints.

    Saves:
    1. Composite model weights: joint_navigation_model.pt
    2. Final joint model weights alias: final_joint_model.pt
    3. Standalone VAE weights: vae_512.pt
    4. Standalone forward transformer weights: forward_transformer.pt
    5. Standalone rotation transformer weights: rotation_transformer.pt

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

    final_joint_path = out_path / "final_joint_model.pt"
    torch.save(model.state_dict(), final_joint_path)

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
    """Execute end-to-end multi-branch training loop with three-stage curriculum.

    Curriculum stages:
    - Stage 1 (Autoencoder Warm-up): model.vae trained at lr_vae_stage1;
      transformers frozen. Sanity gate validated at stage 1 boundary,
      saving stage1_vae.pt.
    - Stage 2 (Transformer Dynamics Optimization): model.vae frozen;
      transformers trained at lr_trans with w_latent=1.0. Saves stage2_transformers.pt
      at stage 2 boundary.
    - Stage 3 (Autoencoder Manifold Refinement): transformers frozen;
      model.vae fine-tuned at lr_vae_stage3 with w_latent=0.0. Final checkpoints
      and loss curves exported at completion.

    Args:
        cfg: Configuration parameters dictionary.
        dataset: Optional pre-constructed dataset instance.
        model: Optional pre-constructed JointNavigationModel instance.
        loss_fn: Optional pre-constructed JointNavigationLoss instance.

    Returns:
        Tuple of (trained_model, history_metrics_dict).

    Raises:
        ValueError: If training dataset is empty.
        RuntimeError: If Stage 1 sanity gate check fails.
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
        w_latent = float(cfg.get("w_latent", 1.0))
        pretrained_vgg = bool(cfg.get("pretrained_vgg", True))
        loss_fn = JointNavigationLoss(
            alpha_perc=alpha_perc,
            beta_kl=beta_kl,
            w_fwd=w_fwd,
            w_rot=w_rot,
            w_recon=w_recon,
            w_latent=w_latent,
            pretrained_vgg=pretrained_vgg,
        )

    loss_fn.to(device)

    # 5. Optimizer and curriculum hyperparameters
    lr = float(cfg.get("lr", 1e-4))
    lr_trans = float(cfg.get("lr_trans", cfg.get("lr", 1e-4)))
    lr_vae_stage1 = float(cfg.get("lr_vae_stage1", lr))
    lr_vae_stage3 = float(cfg.get("lr_vae_stage3", cfg.get("lr_vae", 1e-5)))

    max_epochs = int(cfg.get("max_epochs", 150))
    if "stage1_epochs" in cfg and cfg.get("stage1_epochs") is not None:
        stage1_epochs = int(cfg["stage1_epochs"])
        stage2_epochs = int(cfg.get("stage2_epochs", 50))
        stage3_epochs = int(cfg.get("stage3_epochs", 50))
    elif "warmup_vae_epochs" in cfg and cfg.get("warmup_vae_epochs") is not None:
        stage1_epochs = int(cfg["warmup_vae_epochs"])
        if "stage2_epochs" in cfg and cfg.get("stage2_epochs") is not None:
            stage2_epochs = int(cfg["stage2_epochs"])
            stage3_epochs = int(
                cfg.get(
                    "stage3_epochs",
                    max(0, max_epochs - stage1_epochs - stage2_epochs),
                )
            )
        else:
            stage2_epochs = max(0, max_epochs - stage1_epochs)
            stage3_epochs = 0
    else:
        stage1_epochs = 50
        stage2_epochs = 50
        stage3_epochs = 50

    target_w_latent = float(cfg.get("w_latent", 1.0))
    legacy_ramp = (
        stage3_epochs == 0
        and "w_latent_ramp_start_epoch" in cfg
        and cfg.get("w_latent_ramp_start_epoch") is not None
        and "stage1_epochs" not in cfg
    )
    w_latent_min = float(cfg.get("w_latent_min", 0.01))
    w_latent_max = float(cfg.get("w_latent_max", 1.0))
    ramp_start = int(cfg.get("w_latent_ramp_start_epoch", 61))
    sanity_check_min_std = float(cfg.get("sanity_check_min_std", 15.0))

    def get_stage_for_epoch(epoch_idx: int) -> int:
        if epoch_idx < stage1_epochs:
            return 1
        elif epoch_idx < stage1_epochs + stage2_epochs:
            return 2
        else:
            return 3

    def setup_stage_and_optimizer(stage_num: int) -> Tuple[torch.optim.Optimizer, str]:
        if stage_num == 1:
            model.vae.requires_grad_(True)
            model.forward_transformer.requires_grad_(False)
            model.rotation_transformer.requires_grad_(False)
            stage_lr = lr_vae_stage1
            desc = "Stage 1 (Autoencoder Warm-up)"
        elif stage_num == 2:
            model.vae.requires_grad_(False)
            model.forward_transformer.requires_grad_(True)
            model.rotation_transformer.requires_grad_(True)
            stage_lr = lr_trans
            desc = "Stage 2 (Transformer Dynamics)"
        else:  # stage_num == 3
            model.vae.requires_grad_(True)
            model.forward_transformer.requires_grad_(False)
            model.rotation_transformer.requires_grad_(False)
            stage_lr = lr_vae_stage3
            desc = "Stage 3 (Autoencoder Refinement)"

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        stage_opt = torch.optim.Adam(trainable_params, lr=stage_lr)
        print(f"\n=== Entering {desc} (LR: {stage_lr}) ===")
        return stage_opt, desc

    active_stage: int = get_stage_for_epoch(0)
    optimizer, stage_desc = setup_stage_and_optimizer(active_stage)
    optimizer.zero_grad()

    # 6. AMP and Gradient Accumulation setup
    use_amp = bool(cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(use_amp and device.type == "cuda")
    )
    accum_steps = max(1, int(cfg.get("gradient_accumulation_steps", 2)))

    # 7. Training loop parameters and telemetry tracking
    save_interval = int(cfg.get("save_interval_epochs", 5))
    output_dir = DualSourceVideoDataset._resolve_dir(
        cfg.get("output_dir", "checkpoints")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "loss": [],
        "recon": [],
        "fwd": [],
        "rot": [],
        "perc": [],
        "kl": [],
        "fwd_latent": [],
        "rot_latent": [],
    }

    print(
        f"Starting three-stage training for {max_epochs} epochs "
        f"(Stage1: {stage1_epochs}, Stage2: {stage2_epochs}, Stage3: {stage3_epochs}, "
        f"batch_size={batch_size}, accum_steps={accum_steps}, use_amp={use_amp})..."
    )

    for epoch in range(max_epochs):
        epoch_num = epoch + 1
        current_stage = get_stage_for_epoch(epoch)

        # Check stage transition at boundary
        if current_stage != active_stage:
            active_stage = current_stage
            optimizer, stage_desc = setup_stage_and_optimizer(current_stage)
            optimizer.zero_grad()

        # Compute dynamic w_latent for current epoch
        if legacy_ramp:
            if current_stage == 1:
                current_w_latent = 0.0
            elif epoch_num < ramp_start:
                current_w_latent = w_latent_min
            else:
                denom = max(1, max_epochs - ramp_start)
                progress = min(1.0, (epoch_num - ramp_start) / denom)
                current_w_latent = w_latent_min + progress * (
                    w_latent_max - w_latent_min
                )
        else:
            current_w_latent = target_w_latent if current_stage == 2 else 0.0

        loss_fn.w_latent = current_w_latent

        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_fwd = 0.0
        epoch_rot = 0.0
        epoch_perc = 0.0
        epoch_kl = 0.0
        epoch_fwd_latent = 0.0
        epoch_rot_latent = 0.0
        num_batches = len(train_loader)

        iter_desc = (
            f"Ep {epoch_num:3d}/{max_epochs} [{stage_desc}] "
            f"[w_lat={current_w_latent:.3f}]"
        )
        batch_iter = tqdm(
            train_loader,
            desc=iter_desc,
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
                loss_output = loss_fn.forward_model(model, batch, stage=current_stage)
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
            epoch_fwd_latent += loss_output.metrics["fwd_latent_mse"].item()
            epoch_rot_latent += loss_output.metrics["rot_latent_mse"].item()

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
                w_lat=f"{current_w_latent:.3f}",
            )

        # Compute average metrics across all batches for the epoch
        n_b = max(num_batches, 1)
        avg_loss = epoch_loss / n_b
        avg_recon = epoch_recon / n_b
        avg_fwd = epoch_fwd / n_b
        avg_rot = epoch_rot / n_b
        avg_perc = epoch_perc / n_b
        avg_kl = epoch_kl / n_b
        avg_fwd_latent = epoch_fwd_latent / n_b
        avg_rot_latent = epoch_rot_latent / n_b

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["fwd"].append(avg_fwd)
        history["rot"].append(avg_rot)
        history["perc"].append(avg_perc)
        history["kl"].append(avg_kl)
        history["fwd_latent"].append(avg_fwd_latent)
        history["rot_latent"].append(avg_rot_latent)

        if (epoch + 1) % 1 == 0 or (epoch + 1) == max_epochs:
            print(
                f"Epoch {epoch + 1:3d}/{max_epochs} ({stage_desc}) | "
                f"Loss: {avg_loss:.4f} | "
                f"Recon: {avg_recon:.4f} | "
                f"Fwd: {avg_fwd:.4f} | "
                f"Rot: {avg_rot:.4f} | "
                f"Perc: {avg_perc:.4f} | "
                f"KL: {avg_kl:.6f} | "
                f"w_lat: {current_w_latent:.3f} | "
                f"FwdLat: {avg_fwd_latent:.6f} | "
                f"RotLat: {avg_rot_latent:.6f}",
                flush=True,
            )

        # Stage 1 Sanity Gating and Milestone Checkpoint
        if stage1_epochs > 0 and epoch == stage1_epochs - 1:
            model.eval()
            print("\nEvaluating Stage 1 Autoencoder Sanity Gate...")
            sample_stds: List[float] = []
            with torch.no_grad():
                for sample_idx, sample_batch in enumerate(train_loader):
                    if sample_idx >= 5:
                        break
                    s_recon = (
                        sample_batch.recon_frame
                        if isinstance(sample_batch, DualSourceBatch)
                        else sample_batch["recon_frame"]
                    ).to(device)
                    s_out, _, _ = model.forward_reconstruction(s_recon)
                    # Convert to pixel range 0-255 for standard deviation evaluation
                    s_out_px = s_out * 255.0
                    sample_stds.append(float(s_out_px.std().item()))

            avg_recon_std = sum(sample_stds) / max(1, len(sample_stds))
            print(
                f"Stage 1 Recon Pixel Standard Deviation: {avg_recon_std:.2f} "
                f"(Threshold: {sanity_check_min_std:.2f})"
            )
            if avg_recon_std < sanity_check_min_std:
                raise RuntimeError(
                    f"Sanity Check Failed: Stage 1 Autoencoder reconstruction "
                    f"std ({avg_recon_std:.2f}) is below threshold "
                    f"({sanity_check_min_std:.2f}). Flat gray outputs detected!"
                )
            print("Sanity Check Passed! Autoencoder representations are diverse.")

            stage1_ckpt_path = output_dir / "stage1_vae.pt"
            model.export_vae_checkpoint(stage1_ckpt_path)
            print(f"Saved Stage 1 milestone checkpoint: {stage1_ckpt_path.name}\n")
            model.train()

        # Stage 2 Milestone Checkpoint
        if stage2_epochs > 0 and epoch == (stage1_epochs + stage2_epochs - 1):
            print("\nSaving Stage 2 Latent Transformers Checkpoint Milestone...")
            stage2_ckpt_path = output_dir / "stage2_transformers.pt"
            torch.save(
                {
                    "forward_transformer": model.forward_transformer.state_dict(),
                    "rotation_transformer": model.rotation_transformer.state_dict(),
                },
                stage2_ckpt_path,
            )
            model.export_forward_checkpoint(output_dir / "forward_transformer.pt")
            model.export_rotation_checkpoint(output_dir / "rotation_transformer.pt")
            print(
                f"Saved Stage 2 milestone checkpoints: {stage2_ckpt_path.name}, "
                "forward_transformer.pt, rotation_transformer.pt\n"
            )

        # Stage 3 Milestone Checkpoint
        if epoch == max_epochs - 1:
            print("\nSaving Final Stage 3 Checkpoints and Loss Telemetry...")
            save_all_checkpoints(model, output_dir)
            plot_loss_curves(history, output_dir)

        # Periodic checkpoint saving
        if (
            save_interval > 0
            and (epoch + 1) % save_interval == 0
            and epoch != max_epochs - 1
        ):
            save_all_checkpoints(model, output_dir)
            plot_loss_curves(history, output_dir)

    # Final checkpoint saving and telemetry plotting
    save_all_checkpoints(model, output_dir)
    plot_loss_curves(history, output_dir)

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

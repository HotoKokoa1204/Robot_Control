"""Module: pipeline
Stage: Library
Author: KafuuChino
Date: 2026-10-12
Description: Encapsulated curricular training pipeline for JointNavigationModel.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from agilab_lib.datasets.dual_source_dataset import (
    DualSourceBatch,
    dual_source_collate_fn,
)
from agilab_lib.models.joint_loss import JointNavigationLoss
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    get_checkpoints_dir,
    resolve_project_path,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class TrainingResult:
    """Telemetry, milestones, and sanity metrics produced by CurricularTrainingPipeline.

    Attributes:
        history: Loss telemetry curves across all epochs (total loss,
            reconstruction loss, forward prediction loss, rotation prediction
            loss, perceptual loss, KL loss, and latent MSE alignment loss).
        checkpoint_paths: Milestone checkpoint paths saved during training
            (e.g., stage1_vae.pt, stage2_transformers.pt, final_joint_model.pt).
        sanity_gate_passed: Whether the empirical diversity sanity check passed at the
            Stage 1 curriculum boundary.
        sanity_metrics: Empirical sanity gate metrics recorded at Stage 1 boundary
            (e.g., average standard deviation, threshold).
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    checkpoint_paths: Dict[str, Path] = field(default_factory=dict)
    sanity_gate_passed: bool = True
    sanity_metrics: Dict[str, float] = field(default_factory=dict)


def save_all_checkpoints(
    model: JointNavigationModel,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save composite and exported standalone component weights to output directory.

    Saves:
        - joint_navigation_model.pt (Full composite model)
        - final_joint_model.pt (Alias for composite model)
        - vae_512.pt (Standalone VAE)
        - forward_transformer.pt (Standalone forward transformer)
        - rotation_transformer.pt (Standalone rotation transformer)

    Args:
        model: JointNavigationModel instance.
        output_dir: Directory path for checkpoint files.

    Returns:
        Dictionary mapping component keys ('composite', 'vae', 'forward',
        'rotation', etc.) to saved checkpoint file paths.
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

    return {
        "composite": composite_path,
        "joint_navigation_model": composite_path,
        "joint_navigation_model.pt": composite_path,
        "final_joint_model": final_joint_path,
        "final_joint_model.pt": final_joint_path,
        "vae": vae_path,
        "vae_512.pt": vae_path,
        "forward": fwd_path,
        "forward_transformer": fwd_path,
        "forward_transformer.pt": fwd_path,
        "rotation": rot_path,
        "rotation_transformer": rot_path,
        "rotation_transformer.pt": rot_path,
    }


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


class CurricularTrainingPipeline:
    """Encapsulated 3-stage curricular training orchestrator for model.

    Orchestrates the three-stage curricular progression:
        Stage 1 (Autoencoder Warm-up): model.vae trained at lr_vae_stage1 with
            frozen forward and rotation transformers. Sanity gate validated at
            the Stage 1 boundary, saving stage1_vae.pt milestone.
        Stage 2 (Transformer Dynamics Optimization): model.vae frozen; forward and
            rotation transformers trained at lr_trans with active latent alignment
            loss (w_latent=1.0). Saves stage2_transformers.pt milestone at boundary.
        Stage 3 (Autoencoder Manifold Refinement): transformers frozen; model.vae
            fine-tuned at lr_vae_stage3 under joint multi-branch loss. Final
            checkpoints (joint_navigation_model.pt, final_joint_model.pt) and loss
            telemetry exported upon completion.
    """

    def __init__(
        self,
        cfg: Optional[Union[DictConfig, dict]] = None,
        loss_fn: Optional[JointNavigationLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize CurricularTrainingPipeline.

        Args:
            cfg: Optional default configuration parameters dictionary.
            loss_fn: Optional pre-configured JointNavigationLoss instance.
            device: Optional target device for pipeline execution.
        """
        self.default_cfg = cfg
        self.loss_fn = loss_fn
        self.default_device = device

    @staticmethod
    def _get_cfg_val(
        cfg: Optional[Union[DictConfig, dict]],
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a key from cfg dictionary or DictConfig with fallback.

        Args:
            cfg: Configuration object or dictionary.
            key: Parameter key to look up.
            default: Default value returned if key is missing or None.

        Returns:
            The parameter value or default.
        """
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            val = cfg.get(key, default)
            return default if val is None else val
        if hasattr(cfg, "get"):
            val = cfg.get(key, default)
            return default if val is None else val
        val = getattr(cfg, key, default)
        return default if val is None else val

    def _resolve_device(
        self,
        device_arg: Optional[Union[str, torch.device]],
        cfg: Optional[Union[DictConfig, dict]],
    ) -> torch.device:
        """Resolve torch device based on explicit argument, config, or availability.

        Args:
            device_arg: Explicit device string or torch.device passed to fit().
            cfg: Configuration dictionary or DictConfig.

        Returns:
            Resolved torch.device instance.
        """
        if device_arg is not None:
            return torch.device(device_arg)
        if self.default_device is not None:
            return torch.device(self.default_device)

        device_from_cfg = self._get_cfg_val(cfg, "device", None)
        if device_from_cfg is not None:
            return torch.device(str(device_from_cfg))

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_output_dir(
        self,
        cfg: Optional[Union[DictConfig, dict]],
    ) -> Path:
        """Resolve and guarantee valid writable output directory.

        Args:
            cfg: Configuration dictionary or DictConfig.

        Returns:
            Canonical Path to the writable output directory.
        """
        output_dir_val = self._get_cfg_val(cfg, "output_dir", None)
        if output_dir_val is not None:
            out_path = ensure_writable_output_path(output_dir_val)
        else:
            out_path = get_checkpoints_dir()
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path

    def _setup_stage_and_optimizer(
        self,
        model: JointNavigationModel,
        stage_num: int,
        lr_vae_stage1: float,
        lr_trans: float,
        lr_vae_stage3: float,
        verbose: bool = True,
    ) -> Tuple[torch.optim.Optimizer, str]:
        """Configure parameter freezing and re-instantiate optimizer for current stage.

        Args:
            model: JointNavigationModel instance.
            stage_num: Current curriculum stage index (1, 2, or 3).
            lr_vae_stage1: Learning rate for Autoencoder in Stage 1.
            lr_trans: Learning rate for transformers in Stage 2.
            lr_vae_stage3: Fine-tuning learning rate for Autoencoder in Stage 3.
            verbose: Whether to print stage entry details.

        Returns:
            Tuple of (optimizer, stage_description_string).
        """
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
        if verbose:
            print(f"\n=== Entering {desc} (LR: {stage_lr}) ===")
        return stage_opt, desc

    def _evaluate_sanity_gate(
        self,
        model: JointNavigationModel,
        loader: DataLoader,
        device: torch.device,
        min_std_threshold: float,
        raise_on_fail: bool = True,
        verbose: bool = True,
    ) -> Tuple[bool, Dict[str, float]]:
        """Evaluate empirical reconstruction pixel standard deviation sanity gate.

        Args:
            model: JointNavigationModel instance.
            loader: Training DataLoader yielding video frames.
            device: Execution device.
            min_std_threshold: Minimum acceptable pixel standard deviation.
            raise_on_fail: Whether to raise RuntimeError if check fails.
            verbose: Whether to print gate metrics.

        Returns:
            Tuple of (passed_boolean, sanity_metrics_dictionary).

        Raises:
            RuntimeError: If empirical standard deviation is below threshold and
                raise_on_fail is True.
        """
        model.eval()
        sample_stds: List[float] = []
        with torch.no_grad():
            for sample_idx, sample_batch in enumerate(loader):
                if sample_idx >= 5:
                    break
                if isinstance(sample_batch, DualSourceBatch):
                    s_recon = sample_batch.recon_frame.to(device)
                elif isinstance(sample_batch, dict):
                    s_recon = sample_batch["recon_frame"].to(device)
                else:
                    s_recon = sample_batch.to(device)

                s_out, _, _ = model.forward_reconstruction(s_recon)
                # Convert to pixel range 0-255 for standard deviation evaluation
                s_out_px = s_out * 255.0
                sample_stds.append(float(s_out_px.std().item()))

        avg_recon_std = sum(sample_stds) / max(1, len(sample_stds))
        metrics = {
            "avg_std": avg_recon_std,
            "threshold": min_std_threshold,
            "reconstruction_std": avg_recon_std,
        }
        passed = avg_recon_std >= min_std_threshold

        if verbose:
            print(
                f"Stage 1 Recon Pixel Standard Deviation: {avg_recon_std:.2f} "
                f"(Threshold: {min_std_threshold:.2f})"
            )

        if not passed:
            msg = (
                f"Sanity Check Failed: Stage 1 Autoencoder reconstruction "
                f"std ({avg_recon_std:.2f}) is below threshold "
                f"({min_std_threshold:.2f}). Flat gray outputs detected!"
            )
            if raise_on_fail:
                raise RuntimeError(msg)
            elif verbose:
                print(f"Warning: {msg}")
        elif verbose:
            print("Sanity Check Passed! Autoencoder representations are diverse.")

        return passed, metrics

    def fit(
        self,
        model: JointNavigationModel,
        dataset: Union[Dataset, DataLoader],
        cfg: Optional[Union[DictConfig, dict]] = None,
        device: Optional[Union[str, torch.device]] = None,
        epoch_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> TrainingResult:
        """Execute end-to-end multi-branch training with three-stage curriculum.

        Args:
            model: JointNavigationModel instance to train.
            dataset: Dataset yielding DualSourceBatch instances or DataLoader.
            cfg: Configuration parameters dictionary or DictConfig.
            device: Optional compute device string or torch.device.
            epoch_callback: Optional callback invoked after each epoch with
                (epoch_idx, stage_idx, metrics_dict).

        Returns:
            TrainingResult containing loss history, checkpoint paths, and
            sanity metrics.

        Raises:
            ValueError: If training dataset is empty.
            RuntimeError: If Stage 1 sanity gate check fails and
                raise_on_sanity_fail is True.
        """
        # 1. Merge configurations
        active_cfg = cfg if cfg is not None else self.default_cfg
        device_obj = self._resolve_device(device, active_cfg)
        verbose = bool(self._get_cfg_val(active_cfg, "verbose", True))

        if verbose:
            print(f"Executing CurricularTrainingPipeline on device: {device_obj}")

        # 2. Setup DataLoader
        batch_size = int(self._get_cfg_val(active_cfg, "batch_size", 16))
        num_workers = int(self._get_cfg_val(active_cfg, "num_workers", 0))

        if isinstance(dataset, DataLoader):
            train_loader = dataset
        else:
            if len(dataset) == 0:
                raise ValueError("Training dataset contains 0 samples.")
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=dual_source_collate_fn,
                num_workers=num_workers,
                pin_memory=(device_obj.type == "cuda"),
            )

        if len(train_loader) == 0:
            raise ValueError("Training DataLoader contains 0 batches.")

        # 3. Model warm-start and device placement
        vae_ckpt = self._get_cfg_val(active_cfg, "vae_checkpoint", None)
        if (
            vae_ckpt is not None
            and str(vae_ckpt).strip() != ""
            and str(vae_ckpt).lower() != "none"
            and str(vae_ckpt).lower() != "null"
        ):
            vae_path = resolve_project_path(str(vae_ckpt))
            if vae_path.exists():
                model.load_vae_pretrained(vae_path)
                if verbose:
                    print(f"Warm-started VAE from checkpoint: {vae_path}")
            elif verbose:
                print(
                    f"Warning: VAE checkpoint '{vae_path}' specified but not found. "
                    "Continuing from scratch."
                )

        model.to(device_obj)

        # 4. Loss function instantiation
        loss_fn = self.loss_fn
        if loss_fn is None:
            alpha_perc = float(self._get_cfg_val(active_cfg, "alpha_perc", 0.5))
            beta_kl = float(self._get_cfg_val(active_cfg, "beta_kl", 0.0001))
            w_fwd = float(self._get_cfg_val(active_cfg, "w_fwd", 1.0))
            w_rot = float(self._get_cfg_val(active_cfg, "w_rot", 1.0))
            w_recon = float(self._get_cfg_val(active_cfg, "w_recon", 1.0))
            w_latent = float(self._get_cfg_val(active_cfg, "w_latent", 1.0))
            pretrained_vgg = bool(self._get_cfg_val(active_cfg, "pretrained_vgg", True))
            loss_fn = JointNavigationLoss(
                alpha_perc=alpha_perc,
                beta_kl=beta_kl,
                w_fwd=w_fwd,
                w_rot=w_rot,
                w_recon=w_recon,
                w_latent=w_latent,
                pretrained_vgg=pretrained_vgg,
            )

        loss_fn.to(device_obj)

        # 5. Curriculum stages and optimization parameters
        lr = float(self._get_cfg_val(active_cfg, "lr", 1e-4))
        lr_trans = float(self._get_cfg_val(active_cfg, "lr_trans", lr))
        lr_vae_stage1 = float(self._get_cfg_val(active_cfg, "lr_vae_stage1", lr))
        lr_vae_stage3 = float(
            self._get_cfg_val(
                active_cfg,
                "lr_vae_stage3",
                self._get_cfg_val(active_cfg, "lr_vae", 1e-5),
            )
        )

        max_epochs_cfg = self._get_cfg_val(active_cfg, "max_epochs", None)

        if (
            active_cfg is not None
            and hasattr(active_cfg, "__contains__")
            and "stage1_epochs" in active_cfg
            and self._get_cfg_val(active_cfg, "stage1_epochs") is not None
        ):
            stage1_epochs = int(self._get_cfg_val(active_cfg, "stage1_epochs"))
            stage2_epochs = int(self._get_cfg_val(active_cfg, "stage2_epochs", 50))
            stage3_epochs = int(self._get_cfg_val(active_cfg, "stage3_epochs", 50))
        elif (
            active_cfg is not None
            and hasattr(active_cfg, "__contains__")
            and "warmup_vae_epochs" in active_cfg
            and self._get_cfg_val(active_cfg, "warmup_vae_epochs") is not None
        ):
            stage1_epochs = int(self._get_cfg_val(active_cfg, "warmup_vae_epochs"))
            max_epochs_int = int(max_epochs_cfg) if max_epochs_cfg else 150
            if (
                "stage2_epochs" in active_cfg
                and self._get_cfg_val(active_cfg, "stage2_epochs") is not None
            ):
                stage2_epochs = int(self._get_cfg_val(active_cfg, "stage2_epochs"))
                stage3_epochs = int(
                    self._get_cfg_val(
                        active_cfg,
                        "stage3_epochs",
                        max(0, max_epochs_int - stage1_epochs - stage2_epochs),
                    )
                )
            else:
                stage2_epochs = max(0, max_epochs_int - stage1_epochs)
                stage3_epochs = 0
        else:
            stage1_epochs = int(self._get_cfg_val(active_cfg, "stage1_epochs", 50))
            stage2_epochs = int(self._get_cfg_val(active_cfg, "stage2_epochs", 50))
            stage3_epochs = int(self._get_cfg_val(active_cfg, "stage3_epochs", 50))

        if max_epochs_cfg is not None:
            max_epochs = int(max_epochs_cfg)
        else:
            max_epochs = stage1_epochs + stage2_epochs + stage3_epochs

        target_w_latent = float(self._get_cfg_val(active_cfg, "w_latent", 1.0))
        legacy_ramp = bool(
            stage3_epochs == 0
            and active_cfg is not None
            and hasattr(active_cfg, "__contains__")
            and "w_latent_ramp_start_epoch" in active_cfg
            and self._get_cfg_val(active_cfg, "w_latent_ramp_start_epoch") is not None
            and "stage1_epochs" not in active_cfg
        )
        w_latent_min = float(self._get_cfg_val(active_cfg, "w_latent_min", 0.01))
        w_latent_max = float(self._get_cfg_val(active_cfg, "w_latent_max", 1.0))
        ramp_start = int(self._get_cfg_val(active_cfg, "w_latent_ramp_start_epoch", 61))
        sanity_check_min_std = float(
            self._get_cfg_val(active_cfg, "sanity_check_min_std", 15.0)
        )
        raise_on_sanity_fail = bool(
            self._get_cfg_val(active_cfg, "raise_on_sanity_fail", True)
        )

        def get_stage_for_epoch(epoch_idx: int) -> int:
            if epoch_idx < stage1_epochs:
                return 1
            elif epoch_idx < stage1_epochs + stage2_epochs:
                return 2
            else:
                return 3

        active_stage = get_stage_for_epoch(0)
        optimizer, stage_desc = self._setup_stage_and_optimizer(
            model=model,
            stage_num=active_stage,
            lr_vae_stage1=lr_vae_stage1,
            lr_trans=lr_trans,
            lr_vae_stage3=lr_vae_stage3,
            verbose=verbose,
        )
        optimizer.zero_grad()

        # 6. AMP and Gradient Accumulation setup
        use_amp = bool(self._get_cfg_val(active_cfg, "use_amp", True))
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(use_amp and device_obj.type == "cuda")
        )
        accum_steps = max(
            1,
            int(self._get_cfg_val(active_cfg, "gradient_accumulation_steps", 2)),
        )

        # 7. Output directory and telemetry setup
        output_dir = self._resolve_output_dir(active_cfg)
        save_interval = int(self._get_cfg_val(active_cfg, "save_interval_epochs", 5))

        history: Dict[str, List[float]] = {
            "loss": [],
            "recon": [],
            "fwd": [],
            "rot": [],
            "perc": [],
            "kl": [],
            "fwd_latent": [],
            "rot_latent": [],
            "latent_mse": [],
            "total_loss": [],
            "recon_loss": [],
            "perceptual_loss": [],
            "kl_loss": [],
            "forward_loss": [],
            "rotation_loss": [],
        }
        checkpoint_paths: Dict[str, Path] = {}
        sanity_gate_passed = True
        sanity_metrics: Dict[str, float] = {}

        if verbose:
            print(
                f"Starting curricular training for {max_epochs} epochs "
                f"(Stage 1: {stage1_epochs}, Stage 2: {stage2_epochs}, "
                f"Stage 3: {stage3_epochs}, batch_size={batch_size}, "
                f"accum_steps={accum_steps}, use_amp={use_amp})..."
            )

        # 8. Training loop
        for epoch in range(max_epochs):
            epoch_num = epoch + 1
            current_stage = get_stage_for_epoch(epoch)

            # Check stage transition boundary
            if current_stage != active_stage:
                active_stage = current_stage
                optimizer, stage_desc = self._setup_stage_and_optimizer(
                    model=model,
                    stage_num=current_stage,
                    lr_vae_stage1=lr_vae_stage1,
                    lr_trans=lr_trans,
                    lr_vae_stage3=lr_vae_stage3,
                    verbose=verbose,
                )
                optimizer.zero_grad()

            # Dynamic w_latent scheduling
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

            batch_iterator = train_loader
            if verbose and tqdm is not None:
                iter_desc = (
                    f"Ep {epoch_num:3d}/{max_epochs} [{stage_desc}] "
                    f"[w_lat={current_w_latent:.3f}]"
                )
                batch_iterator = tqdm(
                    train_loader,
                    desc=iter_desc,
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                )

            for batch_idx, raw_batch in enumerate(batch_iterator):
                if isinstance(raw_batch, DualSourceBatch) or hasattr(raw_batch, "to"):
                    batch = raw_batch.to(device_obj)
                elif isinstance(raw_batch, dict):
                    batch = {
                        k: (v.to(device_obj) if isinstance(v, torch.Tensor) else v)
                        for k, v in raw_batch.items()
                    }
                else:
                    batch = raw_batch

                with torch.amp.autocast(
                    "cuda",
                    enabled=bool(use_amp and device_obj.type == "cuda"),
                ):
                    loss_output = loss_fn.forward_model(
                        model, batch, stage=current_stage
                    )
                    total_loss = loss_output.total_loss

                # Telemetry accumulation
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
                actual_steps = (
                    rem if (is_last_batch and not is_accum_step) else accum_steps
                )
                loss_scaled = total_loss / actual_steps
                scaler.scale(loss_scaled).backward()

                if is_accum_step or is_last_batch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if (
                    verbose
                    and tqdm is not None
                    and hasattr(batch_iterator, "set_postfix")
                ):
                    batch_iterator.set_postfix(
                        loss=f"{total_loss.item():.4f}",
                        recon=f"{loss_output.metrics['loss_recon'].item():.3f}",
                        fwd=f"{loss_output.metrics['loss_fwd'].item():.3f}",
                        rot=f"{loss_output.metrics['loss_rot'].item():.3f}",
                        w_lat=f"{current_w_latent:.3f}",
                    )

            # Record epoch averages
            n_b = max(num_batches, 1)
            avg_loss = epoch_loss / n_b
            avg_recon = epoch_recon / n_b
            avg_fwd = epoch_fwd / n_b
            avg_rot = epoch_rot / n_b
            avg_perc = epoch_perc / n_b
            avg_kl = epoch_kl / n_b
            avg_fwd_latent = epoch_fwd_latent / n_b
            avg_rot_latent = epoch_rot_latent / n_b
            avg_latent_mse = avg_fwd_latent + avg_rot_latent

            history["loss"].append(avg_loss)
            history["recon"].append(avg_recon)
            history["fwd"].append(avg_fwd)
            history["rot"].append(avg_rot)
            history["perc"].append(avg_perc)
            history["kl"].append(avg_kl)
            history["fwd_latent"].append(avg_fwd_latent)
            history["rot_latent"].append(avg_rot_latent)
            history["latent_mse"].append(avg_latent_mse)

            # Canonical descriptive aliases
            history["total_loss"].append(avg_loss)
            history["recon_loss"].append(avg_recon)
            history["perceptual_loss"].append(avg_perc)
            history["kl_loss"].append(avg_kl)
            history["forward_loss"].append(avg_fwd)
            history["rotation_loss"].append(avg_rot)

            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{max_epochs} ({stage_desc}) | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Recon: {avg_recon:.4f} | "
                    f"Fwd: {avg_fwd:.4f} | "
                    f"Rot: {avg_rot:.4f} | "
                    f"Perc: {avg_perc:.4f} | "
                    f"KL: {avg_kl:.6f} | "
                    f"w_lat: {current_w_latent:.3f} | "
                    f"LatentMSE: {avg_latent_mse:.6f}",
                    flush=True,
                )

            if epoch_callback is not None:
                epoch_callback(
                    epoch,
                    current_stage,
                    {
                        "loss": avg_loss,
                        "recon": avg_recon,
                        "fwd": avg_fwd,
                        "rot": avg_rot,
                        "perc": avg_perc,
                        "kl": avg_kl,
                        "latent_mse": avg_latent_mse,
                    },
                )

            # Stage 1 boundary: Sanity Gate and Milestone Checkpoint
            if stage1_epochs > 0 and epoch == stage1_epochs - 1:
                if verbose:
                    print("\nEvaluating Stage 1 Autoencoder Sanity Gate...")
                gate_passed, gate_metrics = self._evaluate_sanity_gate(
                    model=model,
                    loader=train_loader,
                    device=device_obj,
                    min_std_threshold=sanity_check_min_std,
                    raise_on_fail=raise_on_sanity_fail,
                    verbose=verbose,
                )
                sanity_gate_passed = gate_passed
                sanity_metrics = gate_metrics

                stage1_ckpt_path = output_dir / "stage1_vae.pt"
                model.export_vae_checkpoint(stage1_ckpt_path)
                checkpoint_paths["stage1_vae"] = stage1_ckpt_path
                checkpoint_paths["stage1_vae.pt"] = stage1_ckpt_path
                if verbose:
                    print(
                        f"Saved Stage 1 milestone checkpoint: {stage1_ckpt_path.name}\n"
                    )
                model.train()

            # Stage 2 boundary: Latent Transformers Milestone Checkpoint
            if stage2_epochs > 0 and epoch == (stage1_epochs + stage2_epochs - 1):
                if verbose:
                    print("\nSaving Stage 2 Latent Transformers Milestone...")
                stage2_ckpt_path = output_dir / "stage2_transformers.pt"
                torch.save(
                    {
                        "forward_transformer": (model.forward_transformer.state_dict()),
                        "rotation_transformer": (
                            model.rotation_transformer.state_dict()
                        ),
                    },
                    stage2_ckpt_path,
                )
                checkpoint_paths["stage2_transformers"] = stage2_ckpt_path
                checkpoint_paths["stage2_transformers.pt"] = stage2_ckpt_path

                fwd_pt = output_dir / "forward_transformer.pt"
                rot_pt = output_dir / "rotation_transformer.pt"
                model.export_forward_checkpoint(fwd_pt)
                model.export_rotation_checkpoint(rot_pt)
                checkpoint_paths["forward_transformer"] = fwd_pt
                checkpoint_paths["rotation_transformer"] = rot_pt
                checkpoint_paths["forward_transformer.pt"] = fwd_pt
                checkpoint_paths["rotation_transformer.pt"] = rot_pt

                if verbose:
                    print(
                        f"Saved Stage 2 checkpoints: {stage2_ckpt_path.name}, "
                        f"{fwd_pt.name}, {rot_pt.name}\n"
                    )

            # Periodic saving
            if (
                save_interval > 0
                and (epoch + 1) % save_interval == 0
                and epoch != max_epochs - 1
            ):
                self._save_checkpoint_bundle(model, output_dir, checkpoint_paths)
                plot_file = plot_loss_curves(history, output_dir)
                if plot_file is not None:
                    checkpoint_paths["loss_curves"] = plot_file
                    checkpoint_paths["loss_curves.png"] = plot_file

        # Final milestone saving at completion of all epochs
        if verbose:
            print("\nSaving Final Stage 3 Checkpoints and Loss Telemetry...")
        self._save_checkpoint_bundle(model, output_dir, checkpoint_paths)
        plot_file = plot_loss_curves(history, output_dir)
        if plot_file is not None:
            checkpoint_paths["loss_curves"] = plot_file
            checkpoint_paths["loss_curves.png"] = plot_file

        return TrainingResult(
            history=history,
            checkpoint_paths=checkpoint_paths,
            sanity_gate_passed=sanity_gate_passed,
            sanity_metrics=sanity_metrics,
        )

    def _save_checkpoint_bundle(
        self,
        model: JointNavigationModel,
        output_dir: Path,
        checkpoint_paths: Dict[str, Path],
    ) -> None:
        """Save final and standalone component weights to output directory.

        Args:
            model: JointNavigationModel instance.
            output_dir: Directory path for checkpoint files.
            checkpoint_paths: In-out dictionary of saved checkpoint paths.
        """
        saved = save_all_checkpoints(model, output_dir)
        checkpoint_paths.update(saved)

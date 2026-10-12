"""Module: train_angle_predictor
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Angle Predictor for relative robot Motion Command prediction.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import torch
import torch.nn.functional as F
from agilab_lib.models.angle_predictor import AnglePredictor
from agilab_lib.models.rlt import ResidualLatentTransformer
from agilab_lib.utils.eval_metrics import evaluate_angle_prediction_mae
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    resolve_project_path,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset


class DummyLatentPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Mock dataset generating pairs of Latent Vectors and target angles."""

    def __init__(
        self,
        num_samples: int = 64,
        latent_dim: int = 128,
        return_sin_cos: bool = True,
    ) -> None:
        """Initialize mock latent pair dataset.

        Args:
            num_samples: Number of sample pairs.
            latent_dim: Dimension of latent vectors.
            return_sin_cos: Whether to return sin/cos vector.
        """
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.return_sin_cos = return_sin_cos

    def __len__(self) -> int:
        """Return sample count."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random pair and target."""
        z_i = torch.randn(self.latent_dim, dtype=torch.float32)
        z_j = torch.randn(self.latent_dim, dtype=torch.float32)
        angle_deg = random.uniform(-180.0, 180.0)
        if self.return_sin_cos:
            rad = angle_deg * math.pi / 180.0
            target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor([angle_deg], dtype=torch.float32)
        return z_i, z_j, target


class CachedLatentDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset loading pre-extracted Latent Vector records (.pt) for angle training."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        mode: str = "rotation",
        max_frame_offset: Optional[int] = None,
        samples_per_frame: int = 5,
        return_sin_cos: bool = True,
    ) -> None:
        """Initialize cached latent dataset.

        Args:
            cache_dir: Directory containing .pt files.
            mode: Motion mode (rotation).
            max_frame_offset: Max frame offset between paired latents.
            samples_per_frame: Target samples per base frame.
            return_sin_cos: Whether to return sin/cos representation.
        """
        super().__init__()
        self.mode = mode.lower()
        self.max_frame_offset = max_frame_offset
        self.return_sin_cos = return_sin_cos
        self.records: List[Dict[str, Any]] = []
        self.pairs: List[Tuple[int, int, int]] = []

        dir_path = Path(cache_dir)
        if dir_path.exists():
            pt_files = sorted(dir_path.rglob("*.pt"))
            for r_idx, pf in enumerate(pt_files):
                rec = torch.load(pf, weights_only=False)
                if isinstance(rec, dict) and "latents" in rec:
                    self.records.append(rec)
                    latents = rec["latents"]
                    num_frames = len(latents)
                    half_turn = num_frames // 2
                    effective_offset = (
                        half_turn if max_frame_offset is None else max_frame_offset
                    )
                    for i in range(num_frames):
                        for _ in range(samples_per_frame):
                            offset = random.randint(-effective_offset, effective_offset)
                            if max_frame_offset is None:
                                j = (i + offset) % num_frames
                            else:
                                j = max(0, min(num_frames - 1, i + offset))
                            self.pairs.append((r_idx, i, j))

    def __len__(self) -> int:
        """Return total sample pair count."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch start latent, target latent, and target angle."""
        r_idx, i, j = self.pairs[idx]
        rec = self.records[r_idx]
        latents = rec["latents"]
        z_i = latents[i]
        z_j = latents[j]
        frame_diff = j - i
        total_frames = float(rec.get("total_frames", len(latents)))
        angle_raw = float(frame_diff) * (360.0 / max(total_frames, 1.0))
        angle = ((angle_raw + 180.0) % 360.0) - 180.0
        if self.return_sin_cos:
            rad = angle * math.pi / 180.0
            target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor([angle], dtype=torch.float32)
        return z_i, z_j, target

    def to_tensor_dataset(self) -> TensorDataset:
        """Convert sampled pairs into an in-memory TensorDataset."""
        if not self.pairs:
            dim = int(self.records[0]["latents"].shape[-1]) if self.records else 128
            return TensorDataset(
                torch.empty((0, dim)),
                torch.empty((0, dim)),
                torch.empty((0, 2 if self.return_sin_cos else 1)),
            )
        z_i_list = []
        z_j_list = []
        target_list = []
        for idx in range(len(self.pairs)):
            zi, zj, tgt = self[idx]
            z_i_list.append(zi)
            z_j_list.append(zj)
            target_list.append(tgt)
        return TensorDataset(
            torch.stack(z_i_list),
            torch.stack(z_j_list),
            torch.stack(target_list),
        )


def get_dataset(
    cfg: DictConfig,
) -> Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build or retrieve the training dataset based on configuration.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        A Dataset providing (start_latent, target_latent, sin_cos) tuples.
    """
    data_dir_str = str(cfg.get("data_dir", "")).strip()
    data_path = (
        resolve_project_path(data_dir_str)
        if data_dir_str
        else resolve_project_path(".")
    )
    if data_dir_str and data_path.exists():
        max_offset = cfg.get("max_frame_offset", None)
        cached_dataset = CachedLatentDataset(
            cache_dir=data_path,
            mode="rotation",
            max_frame_offset=int(max_offset) if max_offset is not None else None,
            return_sin_cos=True,
        )
        if len(cached_dataset) > 0:
            print(
                f"Loaded CachedLatentDataset with {len(cached_dataset)} pairs "
                f"from: {data_path}"
            )
            return cached_dataset.to_tensor_dataset()

    if cfg.use_dummy_if_missing:
        print(
            "No valid cached latents found in data directory. "
            "Using DummyLatentPairDataset."
        )
        return DummyLatentPairDataset(
            num_samples=64,
            latent_dim=int(cfg.latent_dim),
            return_sin_cos=True,
        )

    raise FileNotFoundError(f"No valid latent datasets found in: {data_path}")


def run_validation_inference(
    model: AnglePredictor,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    """Execute a validation inference forward pass on the trained model.

    Args:
        model: Trained Angle Predictor model.
        cfg: Configuration parameters.
        device: Torch compute device.
    """
    model.eval()
    latent_dim = int(cfg.latent_dim)
    current_latent = torch.randn(2, latent_dim, device=device)
    target_keyframe_latent = torch.randn(2, latent_dim, device=device)

    with torch.no_grad():
        predicted_sin_cos = model(
            latent1=current_latent,
            latent2=target_keyframe_latent,
        )
        predicted_angles = model.predict_angle_deg(
            latent1=current_latent,
            latent2=target_keyframe_latent,
        )

    print(
        f"Validation predicted [sin, cos] unit vectors:\n"
        f"{predicted_sin_cos.cpu().tolist()}\n"
        f"Validation predicted Motion Command angles (degrees): "
        f"{predicted_angles.squeeze(-1).cpu().tolist()}"
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_angle_predictor",
)
def main(cfg: DictConfig) -> None:
    """Entry point for Angle Predictor training script.

    Args:
        cfg: Hydra configuration dictionary.
    """
    print("Executing Angle Predictor training pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims: List[int] = [int(h) for h in cfg.hidden_dims]
    model = AnglePredictor(
        latent_dim=int(cfg.latent_dim),
        hidden_dims=hidden_dims,
        use_batch_norm=bool(cfg.use_batch_norm),
    ).to(device)

    rlt_model: Optional[ResidualLatentTransformer] = None
    rlt_ckpt_str = cfg.get("rlt_checkpoint", "checkpoints/rlt_model.pt")
    rlt_path = (
        resolve_project_path(str(rlt_ckpt_str))
        if rlt_ckpt_str and str(rlt_ckpt_str).strip()
        else None
    )
    if rlt_path and rlt_path.exists():
        print(f"Loading pre-trained RotModel (teacher) from: {rlt_path}")
        rlt_model = ResidualLatentTransformer(
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.get("rlt_hidden_dim", 128)),
            num_blocks=int(cfg.get("rlt_num_blocks", 5)),
            block_inner_dim=int(cfg.get("rlt_block_inner_dim", 128)),
        ).to(device)
        rlt_model.load_state_dict(
            torch.load(rlt_path, map_location=device, weights_only=True)
        )
        rlt_model.eval()
        for p in rlt_model.parameters():
            p.requires_grad = False
        print("RotModel teacher loaded and frozen.")
    else:
        print(f"No RotModel checkpoint found at {rlt_path}. Using direct supervision.")

    dataset = get_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    lambda_recon = float(cfg.get("lambda_recon", 1.0))
    lambda_angle = float(cfg.get("lambda_angle", 1.0))

    print(f"Starting Angle Predictor training on device: {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_loss_accum = 0.0
        total_recon_accum = 0.0
        total_mae_deg = 0.0

        for z_current, z_target, target_sin_cos in loader:
            z_current = z_current.to(device, non_blocking=True)
            z_target = z_target.to(device, non_blocking=True)
            target_sin_cos = target_sin_cos.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_sin_cos = model(z_current, z_target)

            if rlt_model is not None:
                # Self-supervised latent reconstruction via frozen RLT teacher
                pred_z_target = rlt_model(z_current, sin_cos=pred_sin_cos)
                loss_recon = F.mse_loss(pred_z_target, z_target)
                loss_angle = F.mse_loss(pred_sin_cos, target_sin_cos)
                loss = lambda_recon * loss_recon + lambda_angle * loss_angle
                total_recon_accum += loss_recon.item()
            else:
                loss = F.mse_loss(pred_sin_cos, target_sin_cos)
                total_recon_accum += 0.0

            loss.backward()
            optimizer.step()

            total_loss_accum += loss.item()
            total_mae_deg += evaluate_angle_prediction_mae(pred_sin_cos, target_sin_cos)

        n_batches = max(len(loader), 1)
        avg_loss = total_loss_accum / n_batches
        avg_recon = total_recon_accum / n_batches
        avg_mae_deg = total_mae_deg / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == int(cfg.max_epochs):
            if rlt_model is not None:
                print(
                    f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - "
                    f"Total Loss: {avg_loss:.6f}, "
                    f"Latent Recon MSE: {avg_recon:.6f}, "
                    f"Angular MAE: {avg_mae_deg:.2f} deg",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - "
                    f"Angle MSE: {avg_loss:.6f}, "
                    f"Angular MAE: {avg_mae_deg:.2f} deg",
                    flush=True,
                )

    # Validation inference step
    print("Running validation control angle inference...")
    run_validation_inference(model, cfg, device)

    # Save trained checkpoint
    output_path = ensure_writable_output_path(str(cfg.output_checkpoint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

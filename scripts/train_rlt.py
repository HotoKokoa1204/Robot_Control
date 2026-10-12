"""
Module: train_rlt
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Residual Latent Transformer with 3D Motion Command condition.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from agilab_lib.models.rlt import (
    ForwardLatentTransformer,
    ResidualLatentTransformer,
    RotationLatentTransformer,
)
from agilab_lib.models.vae import VAE
from agilab_lib.utils.storage import (
    ensure_writable_output_path,
    resolve_project_path,
)
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset


class DummyLatentPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Mock dataset generating pairs of Latent Vectors and target motion values."""

    def __init__(
        self,
        num_samples: int = 64,
        latent_dim: int = 128,
        return_sin_cos: bool = True,
    ) -> None:
        """Initialize mock latent pair dataset.

        Args:
            num_samples: Number of synthetic sample pairs.
            latent_dim: Dimension of each latent vector.
            return_sin_cos: Whether to return 2D sin/cos vector instead of 1D scalar.
        """
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.return_sin_cos = return_sin_cos

    def __len__(self) -> int:
        """Return sample count."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random pair and motion value."""
        z_i = torch.randn(self.latent_dim, dtype=torch.float32)
        z_j = torch.randn(self.latent_dim, dtype=torch.float32)
        if self.return_sin_cos:
            angle_deg = random.uniform(-180.0, 180.0)
            rad = angle_deg * math.pi / 180.0
            motion = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        else:
            motion = torch.tensor([random.uniform(0.0, 5.0)], dtype=torch.float32)
        return z_i, z_j, motion


class CachedLatentDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset loading pre-extracted Latent Vector records (.pt) for RLT training."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        mode: str = "rotation",
        max_frame_offset: Optional[int] = None,
        samples_per_frame: int = 5,
        step_distance_meters: Optional[float] = None,
        return_sin_cos: bool = False,
        video_fps: float = 60.0,
        straight_video_speed_mps: float = 2.5,
        buffer_distance_meters: Optional[float] = None,
        buffer_frames: Optional[int] = None,
    ) -> None:
        """Initialize cached latent dataset.

        Args:
            cache_dir: Directory containing .pt files.
            mode: Motion mode ('rotation' or 'forward').
            max_frame_offset: Maximum frame offset between pairs.
            samples_per_frame: Target samples per base frame.
            step_distance_meters: Distance in meters per frame offset.
            return_sin_cos: Whether to return sin/cos representation.
            video_fps: Video frame rate.
            straight_video_speed_mps: Robot cruise velocity in m/s.
            buffer_distance_meters: Distance to trim from head and tail.
            buffer_frames: Explicit frame count to trim.
        """
        super().__init__()
        self.mode: str = mode.lower()
        self.max_frame_offset: Optional[int] = max_frame_offset
        self.return_sin_cos: bool = return_sin_cos
        self.video_fps: float = float(video_fps)
        self.straight_video_speed_mps: float = float(straight_video_speed_mps)

        if step_distance_meters is not None:
            self.step_distance_meters: float = float(step_distance_meters)
        else:
            self.step_distance_meters = self.straight_video_speed_mps / max(
                self.video_fps, 1.0
            )

        if self.mode == "forward":
            if buffer_frames is not None:
                self.buffer_frames: int = max(0, int(buffer_frames))
            elif buffer_distance_meters is not None and buffer_distance_meters > 0:
                self.buffer_frames = int(
                    buffer_distance_meters / self.step_distance_meters
                )
            else:
                self.buffer_frames = 0
        else:
            self.buffer_frames = 0

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

                    if self.mode == "forward" and self.buffer_frames > 0:
                        buf = self.buffer_frames
                        if num_frames <= 2 * buf:
                            continue
                        forward_limit = (
                            max_frame_offset if max_frame_offset is not None else buf
                        )
                        start_idx = buf
                        end_idx = num_frames - buf
                        for i in range(start_idx, end_idx):
                            for _ in range(samples_per_frame):
                                max_diff = min(end_idx - 1 - i, forward_limit)
                                if max_diff <= 0:
                                    continue
                                offset = random.randint(1, max_diff)
                                j = i + offset
                                self.pairs.append((r_idx, i, j))
                    else:
                        half_turn = num_frames // 2
                        effective_offset = (
                            half_turn
                            if max_frame_offset is None and self.mode == "rotation"
                            else (
                                max_frame_offset if max_frame_offset is not None else 15
                            )
                        )
                        for i in range(num_frames):
                            for _ in range(samples_per_frame):
                                offset = random.randint(
                                    -effective_offset, effective_offset
                                )
                                if self.mode == "rotation" and max_frame_offset is None:
                                    j = (i + offset) % num_frames
                                else:
                                    j = max(0, min(num_frames - 1, i + offset))
                                self.pairs.append((r_idx, i, j))

    def __len__(self) -> int:
        """Return total sample pair count."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch start latent, target latent, and motion value tensor."""
        r_idx, i, j = self.pairs[idx]
        rec = self.records[r_idx]
        latents = rec["latents"]
        z_i = latents[i]
        z_j = latents[j]
        frame_diff = j - i

        if self.mode == "rotation":
            total_frames = float(rec.get("total_frames", len(latents)))
            angle_raw = float(frame_diff) * (360.0 / max(total_frames, 1.0))
            angle = ((angle_raw + 180.0) % 360.0) - 180.0
            if self.return_sin_cos:
                rad = angle * math.pi / 180.0
                sc = [math.sin(rad), math.cos(rad)]
                return z_i, z_j, torch.tensor(sc, dtype=torch.float32)
            return z_i, z_j, torch.tensor([angle], dtype=torch.float32)
        else:
            dist = float(frame_diff) * self.step_distance_meters
            return z_i, z_j, torch.tensor([dist], dtype=torch.float32)

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
        motion_list = []
        for idx in range(len(self.pairs)):
            zi, zj, mot = self[idx]
            z_i_list.append(zi)
            z_j_list.append(zj)
            motion_list.append(mot)
        return TensorDataset(
            torch.stack(z_i_list),
            torch.stack(z_j_list),
            torch.stack(motion_list),
        )


def get_dataset(
    cfg: DictConfig,
) -> Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build or retrieve the training dataset based on configuration.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        A Dataset providing (start_latent, target_latent, motion_value) tuples.

    Raises:
        FileNotFoundError: If neither valid cached datasets nor dummy datasets
            are available in the specified directory.
    """
    data_dir_str = str(cfg.get("data_dir", "")).strip()
    data_path: Path = (
        resolve_project_path(data_dir_str)
        if data_dir_str
        else resolve_project_path(".")
    )
    if data_dir_str and data_path.exists():
        mode = str(cfg.mode).lower()
        return_sin_cos = bool(cfg.get("return_sin_cos", mode == "rotation"))
        max_offset = cfg.get("max_frame_offset", None)
        video_fps = float(cfg.get("video_fps", 60.0))
        speed_mps = float(cfg.get("straight_video_speed_mps", 2.5))
        buf_dist = cfg.get("buffer_distance_meters", None)
        buf_dist_val = float(buf_dist) if buf_dist is not None else None
        cached_dataset = CachedLatentDataset(
            cache_dir=data_path,
            mode=mode,
            max_frame_offset=int(max_offset) if max_offset is not None else None,
            return_sin_cos=return_sin_cos,
            video_fps=video_fps,
            straight_video_speed_mps=speed_mps,
            buffer_distance_meters=buf_dist_val,
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
            return_sin_cos=bool(cfg.get("return_sin_cos", True)),
        )

    raise FileNotFoundError(f"No valid latent datasets found in: {data_path}")


def run_validation_inference(
    model: nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    """Execute a validation inference forward pass on the trained model.

    Args:
        model: Trained Latent Transformer model.
        cfg: Configuration parameters.
        device: Torch compute device.
    """
    model.eval()
    latent_dim = int(cfg.latent_dim)
    dummy_latent = torch.randn(2, latent_dim, device=device)

    test_angle = torch.tensor([[15.0], [-10.0]], device=device)
    test_distance = torch.tensor([[1.0], [0.5]], device=device)

    with torch.no_grad():
        if isinstance(model, RotationLatentTransformer):
            pred_latent = model(latent=dummy_latent, angle_deg=test_angle)
        elif isinstance(model, ForwardLatentTransformer):
            pred_latent = model(latent=dummy_latent, distance_meters=test_distance)
        else:
            pred_latent = model(
                latent=dummy_latent,
                angle_deg=test_angle,
                distance_meters=test_distance,
            )

    print(f"Validation inference output Latent Vector shape: {pred_latent.shape}")

    # Optional VAE decoding validation if checkpoint provided
    if cfg.get("vae_checkpoint") and str(cfg.vae_checkpoint).strip():
        vae_ckpt_path = resolve_project_path(str(cfg.vae_checkpoint))
        if vae_ckpt_path.exists():
            vae = VAE(latent_dim=latent_dim).to(device)
            vae_state = torch.load(
                vae_ckpt_path, map_location=device, weights_only=True
            )
            vae.load_state_dict(vae_state)
            vae.eval()
            with torch.no_grad():
                recon_frames = vae.decode(pred_latent)
            print(f"Reconstructed frames from predicted latents: {recon_frames.shape}")


@hydra.main(version_base=None, config_path="../configs", config_name="train_rlt")
def main(cfg: DictConfig) -> None:
    """Entry point for Residual Latent Transformer training script.

    Args:
        cfg: Hydra configuration dictionary.

    Raises:
        ValueError: If an unsupported training mode is supplied.
    """
    print("Executing Residual Latent Transformer training pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = str(cfg.mode).lower()

    if mode == "rotation":
        model: nn.Module = RotationLatentTransformer(
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_blocks=int(cfg.num_blocks),
            block_inner_dim=int(cfg.block_inner_dim),
        ).to(device)
    elif mode == "forward":
        model = ForwardLatentTransformer(
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_blocks=int(cfg.num_blocks),
            block_inner_dim=int(cfg.block_inner_dim),
        ).to(device)
    elif mode == "unified":
        model = ResidualLatentTransformer(
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_blocks=int(cfg.num_blocks),
            block_inner_dim=int(cfg.block_inner_dim),
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported mode: '{mode}'. Must be 'rotation', 'forward', or 'unified'."
        )

    dataset = get_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    use_img_loss = bool(cfg.get("use_image_loss", False))
    img_loss_weight = float(cfg.get("image_loss_weight", 1.0))
    vae: Optional[VAE] = None
    vae_ckpt_str = cfg.get("vae_checkpoint", "")
    if use_img_loss:
        if vae_ckpt_str and str(vae_ckpt_str).strip():
            vae_ckpt_path = resolve_project_path(str(vae_ckpt_str))
            if vae_ckpt_path.exists():
                vae = VAE(latent_dim=int(cfg.latent_dim)).to(device)
                vae.load_state_dict(
                    torch.load(vae_ckpt_path, map_location=device, weights_only=True)
                )
                vae.eval()
                for param in vae.parameters():
                    param.requires_grad = False
                print(
                    f"Loaded frozen VAE decoder from: {vae_ckpt_path} "
                    f"(loss weight: {img_loss_weight})"
                )
            else:
                print(
                    f"Warning: use_image_loss is True but vae_checkpoint not found at: "
                    f"{vae_ckpt_path}. Falling back to pure latent MSE."
                )
        else:
            print(
                "Warning: use_image_loss is True but vae_checkpoint is empty. "
                "Falling back to pure latent MSE."
            )

    print(f"Starting training in [{mode}] mode on device: {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_loss = 0.0
        total_latent_loss = 0.0
        total_img_loss = 0.0
        for z_i, z_j, motion_val in loader:
            z_i = z_i.to(device, non_blocking=True)
            z_j = z_j.to(device, non_blocking=True)
            motion_val = motion_val.to(device, non_blocking=True)

            optimizer.zero_grad()
            if mode == "rotation":
                if motion_val.shape[-1] == 2:
                    z_pred = model(latent=z_i, sin_cos=motion_val)
                else:
                    z_pred = model(latent=z_i, angle_deg=motion_val)
            elif mode == "forward":
                z_pred = model(latent=z_i, distance_meters=motion_val)
            elif mode == "unified":
                z_pred = model(
                    latent=z_i,
                    angle_deg=motion_val[:, :1],
                    distance_meters=motion_val[:, 1:],
                )
            else:
                raise ValueError(
                    f"Unsupported mode: {mode}. Must be 'rotation' or 'forward'."
                )

            loss_latent = F.mse_loss(z_pred, z_j)
            if vae is not None and img_loss_weight > 0.0:
                pred_img = vae.decode(z_pred)
                with torch.no_grad():
                    target_img = vae.decode(z_j)
                loss_img = F.mse_loss(pred_img, target_img)
                loss = loss_latent + img_loss_weight * loss_img
                total_img_loss += loss_img.item()
            else:
                loss = loss_latent

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_latent_loss += loss_latent.item()

        avg_loss = total_loss / max(len(loader), 1)
        avg_latent_loss = total_latent_loss / max(len(loader), 1)
        avg_img_loss = total_img_loss / max(len(loader), 1)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == int(cfg.max_epochs):
            if vae is not None and img_loss_weight > 0.0:
                print(
                    f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - "
                    f"Loss: {avg_loss:.6f} "
                    f"(latent: {avg_latent_loss:.6f}, img: {avg_img_loss:.6f})",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - Loss: {avg_loss:.6f}",
                    flush=True,
                )

    # Validation inference step
    print("Running validation inference...")
    run_validation_inference(model, cfg, device)

    # Save trained checkpoint
    output_path = ensure_writable_output_path(str(cfg.output_checkpoint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

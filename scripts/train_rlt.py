"""
Module: train_rlt
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Residual Latent Transformer with 3D Motion Command condition.
"""

import os
from pathlib import Path
from typing import Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from agilab_lib.datasets.latent_dataset import (
    CachedLatentDataset,
    DummyLatentPairDataset,
)
from agilab_lib.models.rlt import (
    ForwardLatentTransformer,
    ResidualLatentTransformer,
    RotationLatentTransformer,
)
from agilab_lib.models.vae import VAE
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


def get_dataset(
    cfg: DictConfig,
) -> Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build or retrieve the training dataset based on configuration.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        A Dataset providing (start_latent, target_latent, motion_value) tuples.
    """
    data_dir_str = str(cfg.get("data_dir", "")).strip()
    if data_dir_str:
        data_path = Path(data_dir_str)
        if data_path.exists():
            mode = str(cfg.mode).lower()
            return_sin_cos = bool(
                cfg.get("return_sin_cos", True if mode == "rotation" else False)
            )
            max_offset = cfg.get("max_frame_offset", None)
            cached_dataset = CachedLatentDataset(
                cache_dir=data_path,
                mode=mode,
                max_frame_offset=int(max_offset) if max_offset is not None else None,
                return_sin_cos=return_sin_cos,
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
    if cfg.vae_checkpoint and os.path.exists(str(cfg.vae_checkpoint)):
        vae = VAE(latent_dim=latent_dim).to(device)
        vae_state = torch.load(
            str(cfg.vae_checkpoint), map_location=device, weights_only=True
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

    print(f"Starting training in [{mode}] mode on device: {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_loss = 0.0
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

            loss = F.mse_loss(z_pred, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == int(cfg.max_epochs):
            print(
                f"Epoch {epoch + 1:3d}/{cfg.max_epochs} - Loss: {avg_loss:.6f}",
                flush=True,
            )

    # Validation inference step
    print("Running validation inference...")
    run_validation_inference(model, cfg, device)

    # Save trained checkpoint
    output_path = Path(str(cfg.output_checkpoint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

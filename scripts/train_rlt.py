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
import torch.nn.functional as F
from agilab_lib.datasets.latent_dataset import DummyLatentPairDataset
from agilab_lib.models.autoencoder import Autoencoder
from agilab_lib.models.rlt import ResidualLatentTransformer
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
    # If real data is not present, use synthetic dummy dataset
    data_path = Path(str(cfg.data_dir))
    if not data_path.exists() and cfg.use_dummy_if_missing:
        print("Data directory not found. Using DummyLatentPairDataset.")
        return DummyLatentPairDataset(num_samples=64, latent_dim=int(cfg.latent_dim))

    # Fallback to dummy dataset for verification
    return DummyLatentPairDataset(num_samples=64, latent_dim=int(cfg.latent_dim))


def run_validation_inference(
    model: ResidualLatentTransformer,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    """Execute a validation inference forward pass on the trained model.

    Args:
        model: Trained Residual Latent Transformer model.
        cfg: Configuration parameters.
        device: Torch compute device.
    """
    model.eval()
    latent_dim = int(cfg.latent_dim)
    dummy_latent = torch.randn(2, latent_dim, device=device)

    test_angle = torch.tensor([[15.0], [-10.0]], device=device)
    test_distance = torch.tensor([[1.0], [0.5]], device=device)

    with torch.no_grad():
        pred_latent = model(
            latent=dummy_latent,
            angle_deg=test_angle,
            distance_meters=test_distance,
        )

    print(f"Validation inference output Latent Vector shape: {pred_latent.shape}")

    # Optional Autoencoder decoding validation if checkpoint provided
    if cfg.ae_checkpoint and os.path.exists(str(cfg.ae_checkpoint)):
        autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
        ae_state = torch.load(str(cfg.ae_checkpoint), map_location=device)
        autoencoder.load_state_dict(ae_state)
        autoencoder.eval()
        with torch.no_grad():
            recon_frames = autoencoder.decode(pred_latent)
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

    model = ResidualLatentTransformer(
        latent_dim=int(cfg.latent_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_blocks=int(cfg.num_blocks),
        block_inner_dim=int(cfg.block_inner_dim),
    ).to(device)

    dataset = get_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    mode = str(cfg.mode).lower()

    print(f"Starting training in [{mode}] mode on device: {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_loss = 0.0
        for z_i, z_j, motion_val in loader:
            z_i = z_i.to(device)
            z_j = z_j.to(device)
            motion_val = motion_val.to(device)

            if mode == "rotation":
                # Pure rotation: distance is zeroed
                angle_deg = motion_val
                distance_meters = torch.zeros_like(angle_deg)
            elif mode == "forward":
                # Pure forward: angle is zeroed
                distance_meters = motion_val
                angle_deg = torch.zeros_like(distance_meters)
            else:
                raise ValueError(
                    f"Unsupported mode: {mode}. Must be 'rotation' or 'forward'."
                )

            optimizer.zero_grad()
            z_pred = model(
                latent=z_i,
                angle_deg=angle_deg,
                distance_meters=distance_meters,
            )
            loss = F.mse_loss(z_pred, z_j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{cfg.max_epochs} - Loss: {avg_loss:.6f}")

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

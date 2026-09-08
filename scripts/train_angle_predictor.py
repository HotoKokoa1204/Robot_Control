"""Module: train_angle_predictor
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Angle Predictor for relative robot Motion Command prediction.
"""

from pathlib import Path
from typing import List, Tuple

import hydra
import torch
import torch.nn.functional as F
from agilab_lib.datasets.latent_dataset import (
    CachedLatentDataset,
    DummyLatentPairDataset,
)
from agilab_lib.models.angle_predictor import AnglePredictor
from agilab_lib.utils.eval_metrics import evaluate_angle_prediction_mae
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


def get_dataset(
    cfg: DictConfig,
) -> Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build or retrieve the training dataset based on configuration.

    Args:
        cfg: Hydra configuration dictionary.

    Returns:
        A Dataset providing (start_latent, target_latent, sin_cos) tuples.
    """
    data_path = Path(str(cfg.data_dir))
    if data_path.exists():
        cached_dataset = CachedLatentDataset(
            cache_dir=data_path, mode="rotation", return_sin_cos=True
        )
        if len(cached_dataset) > 0:
            print(
                f"Loaded CachedLatentDataset with {len(cached_dataset)} pairs "
                f"from: {data_path}"
            )
            return cached_dataset

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

    dataset = get_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    print(f"Starting Angle Predictor training on device: {device}...")
    model.train()

    for epoch in range(int(cfg.max_epochs)):
        total_mse_loss = 0.0
        total_mae_deg = 0.0

        for z_current, z_target, target_sin_cos in loader:
            z_current = z_current.to(device)
            z_target = z_target.to(device)
            target_sin_cos = target_sin_cos.to(device)

            optimizer.zero_grad()
            pred_sin_cos = model(z_current, z_target)
            loss = F.mse_loss(pred_sin_cos, target_sin_cos)
            loss.backward()
            optimizer.step()

            total_mse_loss += loss.item()
            total_mae_deg += evaluate_angle_prediction_mae(pred_sin_cos, target_sin_cos)

        n_batches = max(len(loader), 1)
        avg_mse = total_mse_loss / n_batches
        avg_mae_deg = total_mae_deg / n_batches
        print(
            f"Epoch {epoch + 1:2d}/{cfg.max_epochs} - "
            f"MSE Loss: {avg_mse:.6f}, "
            f"Angular MAE: {avg_mae_deg:.2f} deg"
        )

    # Validation inference step
    print("Running validation control angle inference...")
    run_validation_inference(model, cfg, device)

    # Save trained checkpoint
    output_path = Path(str(cfg.output_checkpoint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

"""Module: train_angle_predictor
Stage: Script
Author: KafuuChino
Date: 2026-09-07
Description: Train Angle Predictor for relative robot Motion Command prediction.
"""

from typing import List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from agilab_lib.datasets.latent_dataset import (
    CachedLatentDataset,
    DummyLatentPairDataset,
)
from agilab_lib.models.angle_predictor import AnglePredictor
from agilab_lib.models.rlt import ResidualLatentTransformer
from agilab_lib.utils.eval_metrics import evaluate_angle_prediction_mae
from agilab_lib.utils.storage import resolve_project_path
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
    output_path = resolve_project_path(str(cfg.output_checkpoint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model checkpoint successfully saved to: {output_path}")


if __name__ == "__main__":
    main()

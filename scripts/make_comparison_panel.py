"""Generate visual comparison panel for 3-stage curricular trained model."""

import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agilab_lib.datasets.dual_source_dataset import (
    DualSourceVideoDataset,
    dual_source_collate_fn,
)
from agilab_lib.models.joint_navigation import JointNavigationModel


def main() -> None:
    """Evaluate trained model and generate visual comparison panel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset without preloading (fast seek)
    ds = DualSourceVideoDataset(
        one_path_dir="data/one_path",
        rotation_dir="data/360",
        img_height=108,
        img_width=192,
        buffer_distance_meters=2.0,
        rotation_one_per_subfolder=True,
        preload_frames=False,
        recon_source="random",
        seed=123,
    )

    # Load trained model
    ckpt_path = Path("checkpoints/final_joint_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/joint_navigation_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    model = JointNavigationModel().to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    # Sample representative batch
    indices = [15, 60, 150, 320]
    samples = [ds[i] for i in indices]
    batch = dual_source_collate_fn(samples).to(device)

    with torch.no_grad():
        # Reconstruction
        recon_fwd, _, _ = model.forward_reconstruction(batch.fwd_current)

        # Forward dynamics prediction
        mu_fwd = model.vae.encode(batch.fwd_current)[0]
        pred_z_fwd = model.forward_transformer(
            mu_fwd, distance_meters=batch.fwd_distance
        )
        pred_fwd = model.vae.decode(pred_z_fwd)

        # Rotation dynamics prediction
        mu_rot = model.vae.encode(batch.rot_current)[0]
        pred_z_rot = model.rotation_transformer(mu_rot, angle_deg=batch.rot_sin_cos)
        pred_rot = model.vae.decode(pred_z_rot)

    # Helper: tensor to uint8 numpy
    def to_np(t: torch.Tensor) -> np.ndarray:
        """Convert float image tensor to uint8 numpy array."""
        arr = t.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
        return (arr * 255.0).astype(np.uint8)

    fwd_curr = to_np(batch.fwd_current)
    fwd_recon = to_np(recon_fwd)
    fwd_tgt = to_np(batch.fwd_target)
    fwd_pred = to_np(pred_fwd)

    rot_curr = to_np(batch.rot_current)
    rot_tgt = to_np(batch.rot_target)
    rot_pred = to_np(pred_rot)

    print("\n=== Validation Pixel Statistics (0-255 scale) ===")
    print(f"GT Forward Current Std:    {fwd_curr.std():.2f}")
    print(f"Reconstruction Current Std:{fwd_recon.std():.2f}")
    print(f"GT Forward Target Std:     {fwd_tgt.std():.2f}")
    print(f"Predicted Forward Std:     {fwd_pred.std():.2f}")
    print(f"GT Rotation Current Std:   {rot_curr.std():.2f}")
    print(f"GT Rotation Target Std:    {rot_tgt.std():.2f}")
    print(f"Predicted Rotation Std:    {rot_pred.std():.2f}")

    # Plot panel
    n = len(indices)
    fig, axes = plt.subplots(n, 6, figsize=(18, 3.2 * n))
    cols = [
        "Fwd Current (GT)",
        "Fwd Recon",
        "Fwd Target (GT)",
        "Fwd Pred",
        "Rot Target (GT)",
        "Rot Pred",
    ]
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        dist = batch.fwd_distance[i].item()
        s = batch.rot_sin_cos[i, 0].item()
        c = batch.rot_sin_cos[i, 1].item()
        deg = math.degrees(math.atan2(s, c))

        axes[i, 0].imshow(fwd_curr[i])
        axes[i, 0].set_ylabel(f"Pair #{i + 1}", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(fwd_recon[i])
        axes[i, 1].axis("off")

        axes[i, 2].imshow(fwd_tgt[i])
        axes[i, 2].set_title(f"Target (+{dist:.2f}m)", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(fwd_pred[i])
        axes[i, 3].set_title(f"Pred (+{dist:.2f}m)", fontsize=10)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(rot_tgt[i])
        axes[i, 4].set_title(f"Rot Target ({deg:+.1f} deg)", fontsize=10)
        axes[i, 4].axis("off")

        axes[i, 5].imshow(rot_pred[i])
        axes[i, 5].set_title(f"Rot Pred ({deg:+.1f} deg)", fontsize=10)
        axes[i, 5].axis("off")

    plt.tight_layout()
    out_file = Path(
        "C:/Users/KafuuChino/.gemini/antigravity/brain/4303624a-de40-4c9b-b2bf-6891112bda38/three_stage_ae_comparison_panel.png"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison panel to: {out_file}", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()

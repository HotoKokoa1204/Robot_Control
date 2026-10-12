"""Module: train_multibranch
Stage: Script
Author: KafuuChino
Date: 2026-10-12
Description: CLI driver for multi-branch joint navigation curricular training.
"""

# ruff: noqa: E402
import sys
from pathlib import Path

# Ensure 'src' is in sys.path when invoked directly
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PIL import Image  # isort: skip # noqa: F401
import hydra
from omegaconf import DictConfig, OmegaConf

from agilab_lib.datasets.dual_source_dataset import (
    DualSourceVideoDataset,
    DummyDualSourceVideoDataset,
)
from agilab_lib.models.joint_navigation import JointNavigationModel
from agilab_lib.training import CurricularTrainingPipeline
from agilab_lib.utils.storage import resolve_project_path


@hydra.main(
    config_path="../configs", config_name="train_multibranch", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Execute curricular training from parsed Hydra configuration."""
    print(f"=== Multi-Branch Joint Navigation Training ===\n{OmegaConf.to_yaml(cfg)}")
    model = JointNavigationModel(
        cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks, cfg.block_inner_dim
    )
    p1 = resolve_project_path(cfg.one_path_dir)
    p2 = resolve_project_path(cfg.rotation_dir)
    if p1.exists() and p2.exists():
        kw = {k: cfg[k] for k in ("img_height", "img_width", "video_fps") if k in cfg}
        dataset = DualSourceVideoDataset(p1, p2, **kw)
    elif cfg.get("use_dummy_if_missing", True):
        dataset = DummyDualSourceVideoDataset(
            int(cfg.get("num_samples", 100)), int(cfg.img_height), int(cfg.img_width)
        )
    else:
        raise FileNotFoundError(f"Missing data dirs: '{p1}', '{p2}'")

    pipeline = CurricularTrainingPipeline()
    result = pipeline.fit(model, dataset, cfg)
    _ = result


if __name__ == "__main__":
    main()

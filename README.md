# Visual Navigation System

[繁體中文](README_zh.md) | English

> **Note**: This repository was bootstrapped with the [AGILAB Software Template](https://github.com/AGILAB-NTNU/SoftwareTemplate).

## Overview

**Visual Navigation System** is a research library and experimental pipeline developed by AGILAB for video representation learning and self-supervised robot visual navigation.

The system compresses continuous video observation streams into compact **Latent Vectors** using an **Autoencoder**, detects informative navigation milestones (**Keyframes**) based on Euclidean distance threshold **Tau ($\tau$)**, predicts future states using a 3D **Motion Command** conditioned **Residual Latent Transformer**, upscales decoded observations using an **RRDN** super-resolution decoder, and estimates robot steering actions with an **Angle Predictor**.

### Core Architecture Components

- **Autoencoder / VAE**: Compresses RGB video frames $(3, 108, 192)$ into compact 128-dimensional **Latent Vectors** and reconstructs them back to image space.
- **Residual Latent Transformer**: Predicts future Latent Vectors conditioned on a unified 3D **Motion Command** $[\sin \theta, \cos \theta, d]$ combining rotation angle $\theta$ (degrees) and linear translation distance $d$ (meters).
- **RRDN** (Residual in Residual Dense Network): Enhanced super-resolution decoder restoring high-fidelity spatial details to decoded latent reconstructions (see [ADR-0001](docs/adr/0001-rrdn-as-enhanced-decoder.md)).
- **Angle Predictor**: Multi-layer perceptron taking a pair of Latent Vectors (current frame and target Keyframe) to predict relative rotation angle Motion Commands for navigation control.
- **Keyframe Extraction**: Extracts keyframe indices along video trajectories where the Latent Vector Euclidean distance exceeds threshold $\tau$.

---

## Installation

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- PyTorch 2.0+
- CUDA-compatible GPU (optional, CPU fallback supported)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Visual_Navigation_System
   ```

2. **Install in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

---

## Project Structure

```text
Visual_Navigation_System/
├── configs/                          # Hydra YAML configuration files
│   ├── extract_keyframes.yaml        # Keyframe extraction settings
│   ├── generate_video.yaml           # Video interpolation & RRDN decode settings
│   ├── train_rlt.yaml                # Residual Latent Transformer training
│   └── train_angle_predictor.yaml    # Angle Predictor training
├── docs/                             # Architecture decision records
│   └── adr/
│       └── 0001-rrdn-as-enhanced-decoder.md
├── scripts/                          # Main pipeline execution scripts
│   ├── extract_keyframes.py          # Script 1: Extract Keyframes via Tau (τ)
│   ├── generate_video.py             # Script 2: Latent interpolation & video export
│   ├── train_rlt.py                  # Script 3: Train Residual Latent Transformer
│   └── train_angle_predictor.py      # Script 4: Train Angle Predictor
├── src/
│   └── agilab_lib/                   # Installable Python package
│       ├── datasets/                 # Video, latent offset, and SR datasets
│       ├── models/                   # Autoencoder, RLT, Angle Predictor, RRDN
│       └── utils/                    # Interpolation, PCA, keyframes, metrics
└── tests/                            # Automated PyTest suite (32 unit tests)
```

---

## Pipeline Execution Guide

All pipeline scripts utilize [Hydra](https://hydra.cc/) for hierarchical configuration and command-line overrides.

### 1. Keyframe Extraction

Extract representative Keyframes along video trajectories based on latent Euclidean distance threshold Tau ($\tau \ge 1.5$):

```bash
python scripts/extract_keyframes.py video_path=data/input_video.mp4 tau=1.5 output_json=data/keyframes.json
```

### 2. Video Generation & Latent Interpolation

Linearly interpolate Latent Vectors between extracted Keyframes, decode to frames, and optionally upscale with RRDN to export a `.mp4` video:

```bash
# Standard decoding
python scripts/generate_video.py video_path=data/input_video.mp4 keyframes_json=data/keyframes.json interp_steps=5

# Enhanced decoding with RRDN super-resolution
python scripts/generate_video.py video_path=data/input_video.mp4 keyframes_json=data/keyframes.json use_rrdn=true
```

### 3. Residual Latent Transformer Training

Train the Residual Latent Transformer using single-mode motion trajectories:

```bash
# Pure rotation training (distance_meters zeroed)
python scripts/train_rlt.py mode=rotation max_epochs=10 batch_size=16

# Pure forward training (angle_deg zeroed)
python scripts/train_rlt.py mode=forward max_epochs=10 batch_size=16
```

### 4. Angle Predictor Training

Train the Angle Predictor on Latent Vector pairs using MAE loss to output steering angle control commands:

```bash
python scripts/train_angle_predictor.py max_epochs=10 batch_size=16
```

---

## Testing & Quality Verification

Run the automated test suite across all library modules:

```bash
pytest tests/ -v
```

Execute code formatting and AGILAB lab standard pre-commit hooks:

```bash
pre-commit run --all-files
```

---

## Contributing

This project adheres to the unified AGILAB development workflow and coding standards. Please refer to `AGENTS.md` and the [AGILAB Software Lab Guide](https://agilab-ntnu.github.io/AGILAB_Software_Lab_Guide/en/contributing/) before submitting changes.

## Citation

```bibtex
@article{kafuuchino_2026_visual_navigation,
  author = {KafuuChino},
  title = {Visual Navigation System: Representation Learning and Motion Prediction},
  journal = {AGILAB Research},
  year = {2026},
  url = {https://github.com/AGILAB-NTNU/Visual_Navigation_System}
}
```

## License

MIT License

# 視覺導航系統 (Visual Navigation System)

繁體中文 | [English](README.md)

> **註記**: 此專案由 [AGILAB Software Template](https://github.com/AGILAB-NTNU/SoftwareTemplate) 基礎架構初始化而成。

## 專案簡介

**Visual Navigation System** 是由 AGILAB 實驗室開發的視覺導航與影片表示學習研究函式庫與實驗管線。

本系統透過 **Autoencoder** 將連續影像觀測壓縮為低維度的 **Latent Vector**（潛在向量），依據歐式距離門檻 **Tau ($\tau$)** 自動篩選導航地標（**Keyframe**，關鍵影格），使用以 3D **Motion Command**（運動指令）為條件的 **Residual Latent Transformer** 預測未來的潛在狀態，藉由 **RRDN** 超解析度解碼器提升重建影像畫質，並由 **Angle Predictor** 預測相對轉角控制指令以引導機器人移動。

### 核心架構模組

- **Autoencoder / VAE**：將 RGB 影像影格 $(3, 108, 192)$ 壓縮編碼為 128 維的 **Latent Vector**，並負責將潛在向量還原回影像空間。
- **Residual Latent Transformer**：以結合旋轉角度 $\theta$（度數）與直線位移距離 $d$（公尺）的 3D **Motion Command** 控制向量 $[\sin \theta, \cos \theta, d]$ 為條件，預測未來的 Latent Vector。
- **RRDN** (Residual in Residual Dense Network)：增強型超解析度解碼器，提升 Latent Vector 解碼還原影像之邊緣與空間細節（參見 [ADR-0001](docs/adr/0001-rrdn-as-enhanced-decoder.md)）。
- **Angle Predictor**：接收當前影格與目標 Keyframe 的 Latent Vector 組合，預測相對旋轉角度作為 Motion Command 控制指令。
- **Keyframe Extraction**：沿影片軌跡計算連續影格之 Latent Vector 歐式距離，當距離大於等於 $\tau$ 時記錄關鍵影格索引。

---

## 安裝指南

### 系統需求

- Python 3.10+（已在 Python 3.13 進行相容性測試）
- PyTorch 2.0+
- 支援 CUDA 之 GPU（選用，支援 CPU 運作）

### 環境設定步驟

1. **複製專案：**
   ```bash
   git clone <repository_url>
   cd Visual_Navigation_System
   ```

2. **以開發模式 (editable mode) 安裝套件與開發相依套件：**
   ```bash
   pip install -e ".[dev]"
   ```

3. **安裝 pre-commit hooks：**
   ```bash
   pre-commit install
   ```

---

## 專案架構

```text
Visual_Navigation_System/
├── configs/                          # Hydra YAML 組態設定檔
│   ├── extract_keyframes.yaml        # Keyframe 提取設定
│   ├── generate_video.yaml           # 影片插值生成與 RRDN 解碼設定
│   ├── train_rlt.yaml                # Residual Latent Transformer 訓練設定
│   └── train_angle_predictor.yaml    # Angle Predictor 訓練設定
├── docs/                             # 架構決策記錄 (ADR)
│   └── adr/
│       └── 0001-rrdn-as-enhanced-decoder.md
├── scripts/                          # 核心管線執行腳本
│   ├── extract_keyframes.py          # Script 1: 依據 Tau (τ) 提取 Keyframes
│   ├── generate_video.py             # Script 2: Latent 插值解碼與影片匯出
│   ├── train_rlt.py                  # Script 3: 訓練 Residual Latent Transformer
│   └── train_angle_predictor.py      # Script 4: 訓練 Angle Predictor
├── src/
│   └── agilab_lib/                   # 可安裝之核心 Python 函式庫
│       ├── datasets/                 # 影片、Latent 軌跡與 SR 資料集
│       ├── models/                   # Autoencoder, RLT, Angle Predictor, RRDN
│       └── utils/                    # 內插, PCA, Keyframe 提取, 評估指標
└── tests/                            # PyTest 自動化測試套件 (32 個單元測試)
```

---

## 管線使用說明

所有管線腳本均採用 [Hydra](https://hydra.cc/) 進行階層式參數管理與 CLI 覆寫。

### 1. Keyframe 提取

依據 Latent Vector 歐式距離門檻 Tau ($\tau \ge 1.5$) 自動提取最具代表性的關鍵影格：

```bash
python scripts/extract_keyframes.py video_path=data/input_video.mp4 tau=1.5 output_json=data/keyframes.json
```

### 2. 影片生成與潛在向量插值

在提取的 Keyframes 之間進行線性潛在向量插值，解碼為影格，並可選擇性啟用 RRDN 超解析度增強，輸出為 `.mp4` 格式：

```bash
# 標準解碼
python scripts/generate_video.py video_path=data/input_video.mp4 keyframes_json=data/keyframes.json interp_steps=5

# 啟用 RRDN 超解析度解碼
python scripts/generate_video.py video_path=data/input_video.mp4 keyframes_json=data/keyframes.json use_rrdn=true
```

### 3. 訓練 Residual Latent Transformer

支援單一運動模式（純旋轉或純前進）的未來潛在狀態預測訓練：

```bash
# 純旋轉模式 (distance_meters 自動歸零)
python scripts/train_rlt.py mode=rotation max_epochs=10 batch_size=16

# 純前進模式 (angle_deg 自動歸零)
python scripts/train_rlt.py mode=forward max_epochs=10 batch_size=16
```

### 4. 訓練 Angle Predictor

輸入當前影格與目標 Keyframe 的潛在向量對，採用 MAE 損失優化相對轉角預測器：

```bash
python scripts/train_angle_predictor.py max_epochs=10 batch_size=16
```

---

## 測試與品質驗證

執行完整的自動化測試套件：

```bash
pytest tests/ -v
```

執行全模組靜態格式化與 AGILAB 代碼規範檢查：

```bash
pre-commit run --all-files
```

---

## 貢獻指南

本專案遵循 AGILAB 的統一開發規範。在開始貢獻之前，請參閱 `AGENTS.md` 與 [AGILAB Software Lab Guide](https://agilab-ntnu.github.io/AGILAB_Software_Lab_Guide/zh/contributing/) 以瞭解分支策略與程式碼規範。

## 引用 (Citation)

```bibtex
@article{kafuuchino_2026_visual_navigation,
  author = {KafuuChino},
  title = {Visual Navigation System: Representation Learning and Motion Prediction},
  journal = {AGILAB Research},
  year = {2026},
  url = {https://github.com/AGILAB-NTNU/Visual_Navigation_System}
}
```

## 授權條款

MIT License

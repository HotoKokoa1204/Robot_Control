# [純視覺定位與導航系統 (Visual Navigation System)]

繁體中文 | [English](README.md)

> **註記**: 此專案由 [AGILAB Software Template](https://github.com/AGILAB-NTNU/SoftwareTemplate) 基礎架構初始化而成。
>

## 專案簡介

本專案提出一套**純視覺定位與導航系統 (Visual Navigation System)**，專為低成本移動載具與室內環境設計。無需依賴 GPS、LiDAR 及深度相機等高昂且在室內易受環境干擾的額外感測器，僅透過純 RGB 影像即可完成環境感知、即時定位與導航。

---

## 安裝指南

### 系統需求

- [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12
- NVIDIA GPU (建議支援 CUDA 13.0 以加速 PyTorch 與 PyG 運算)

### 環境設定步驟

1. **複製專案：**

    ```bash
    git clone <repository_url>
    cd visual_navigation_system
    ```

2. **建立並啟動虛擬環境：**
此指令會安裝所有底層系統依賴（如 CUDA）並將本地的 Python 套件以開發模式 (editable mode) 安裝。

    ```bash
    conda env create -f environment.yml
    conda activate agilab_env
    ```

3. **安裝 pre-commit hooks（強烈建議）：**
確保程式碼提交前自動進行排版與檢查。

    ```bash
    pre-commit install
    ```


---

## 📁 專案架構

```
.
├── configs/
│   └── config.yaml          # 全域設定檔：包含所有超參數、路徑與繪圖節點設定
├── docs/
│   └── tku_visual_navigation/
│       ├── 需手動處理的步驟.md
│       ├── README_zh.md
│       └── 最佳影片的節點數推導過程.md
├── src/
│   └── agilab/
│       └── tku/
│           └── visual_navigation_system/
│           ├── dataset.py       # 資料集建構 (包含 SegmentAligner 幀數對齊機制)
│           ├── evaluator.py     # 特徵提取器 (FeatureExtractor 特徵抽取)
│           ├── graph_builder.py # PyTorch Geometric 圖資料結構建立器
│           ├── lstm_model.py    # LocationLSTM (LSTM) 模型架構
│           ├── models.py        # AutoEncoder (Encoder/F_Decoder) 模型架構
│           ├── predictor.py     # 預測輔助工具 (包含貪婪法預測與真實索引映射)
│           ├── trainer.py       # 訓練器 (Trainer, LSTMTrainer) 含混合精度機制
│           ├── video_selector.py# BestVideoSelector 最佳基準影片挑選與合成
│           ├── video_utils.py   # 影片底層處理工具箱 (Resize, FPS 降採樣, 陣列拼接)
│           └── visualizer.py    # GraphVisualizer 圖論視覺化與最短路徑繪製
├── scripts/
│   └── tku/
│       └── visual_navigation_system/
│           ├── 01_preprocess_videos.py        # [預處理] 影片降畫質與降取樣
│           ├── 02_prepare_dataset.py          # [資料集] 抽樣與打包 Tensor (.pt)
│           ├── 03_train.py                    # [模型訓練] 訓練 AutoEncoder 模型
│           ├── 04_inference.py                # [測試] 測試 AutoEncoder 影像還原能力
│           ├── 05_reverse_video.py            # [影片工具] 影片倒轉工具
│           ├── 06_capture_video.py            # [影片工具] 擷取影片特定片段工具
│           ├── 07_uniform_downsample.py       # [資料集] 影片均勻下採樣對齊
│           ├── 08_extract_features.py         # [特徵抽取] 批次抽取影片特徵 (.npy)
│           ├── 09_make_best_video.py          # [建構地圖] 挑選與合成各路徑最佳影片
│           ├── 10_check_npy_shape.py          # [特徵工具] 檢查 .npy 特徵檔案維度
│           ├── 11_adjust_best_video_length.py # [建構地圖] 調整影片與特徵長度
│           ├── 12_concat_best_videos.py       # [建構地圖] 最佳影片串接
│           ├── 13_greedy_prediction.py        # [定位預測] 貪婪法比對，輸出預測軌跡動畫
│           ├── 14_train_lstm.py               # [模型訓練] 訓練 LSTM 模型
│           ├── 15_lstm_prediction.py          # [定位預測] LSTM 預測，輸出預測軌跡動畫
│           ├── 16_visualize_graph.py          # [圖論分析] 視覺化整體室內地圖 (PyG)
│           ├── 17_shortest_path.py            # [圖論分析] 計算並繪製最短路徑
│           └── 18_view_frame.py               # [除錯工具] 本機端逐個 frame 檢視影片工具
├── Database/                # 存放原始影片、預處理 Tensor (.pt) 與特徵向量 (.npy)
├── Map/                     # 存放合成後的虛擬地圖(最佳影片)、測試影片與預測結果動畫
├── model/                   # 存放訓練好的 Node Encoder 與 LSTM 模型權重 (.pth)
├── tests/                   # 自動化測試程式碼 (PyTest)
├── docker/                  # 容器化部署 Docker 設定
├── pyproject.toml           # 專案建置與套件相依性設定
└── environment.yml          # Conda 虛擬環境套件清單
```

---

## 貢獻指南

本專案遵循 AGILAB 的統一開發規範。在開始貢獻之前，請先參閱 [AGILAB Software Lab Guide](https://agilab-ntnu.github.io/AGILAB_Software_Lab_Guide/zh/contributing/) 以瞭解分支策略與程式碼規範。

---

## 引用 (Citation)

如果您在研究中使用了本專案，請使用以下格式進行引用：

```
@article{author_year_title,
  author = {Author, First and Author, Second},
  title = {Project Title},
  journal = {Journal or Conference Name},
  year = {2026},
  url = {https://github.com/AGILAB-NTNU/SoftwareTemplate}
}
```

---

## 授權條款

*(請在此加上您的開源授權條款)*

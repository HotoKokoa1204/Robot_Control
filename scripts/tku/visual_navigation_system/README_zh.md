# [純視覺定位與導航系統 (Visual Navigation System)]

繁體中文 | [English](README.md)

> **註記**: 此專案由 [AGILAB Software Template](https://github.com/AGILAB-NTNU/SoftwareTemplate) 基礎架構初始化而成。
>

## 專案簡介

本專案提出一套**純視覺定位與導航系統 (Visual Navigation System)**，專為低成本移動載具與室內環境設計。無需依賴 GPS、LiDAR 及深度相機等高昂且在室內易受環境干擾的額外感測器，僅透過純 RGB 影像即可完成環境感知、即時定位與導航。

### 🌟 核心創新與研究貢獻

1. **低硬體門檻與實務可行性**：
在資料集建構階段僅需使用一般相機或手機拍攝場景影片，簡化了傳統 SLAM 或 LiDAR 建圖所需的高度專用設備與繁瑣程序，大幅降低資料蒐集與硬體部署門檻。
2. **基於多重約束的 Node Encoder 特徵萃取**：
系統設計了自定義的 Node Encoder 神經網路，能夠將原始影像轉換為具備空間辨識能力的 512 維特徵向量。
3. **最佳虛擬地圖建構 (Optimal Video / Best Video)**：
提出 Optimal Video 生成演算法，將多路徑影片降採樣對齊並計算歐式距離特徵序列，挑選出具空間感之能力的特徵作為虛擬地圖節點，免去傳統 SLAM 複雜的即時建圖與閉環檢測運算。
4. **結合 LSTM 與歐式距離的高精度時序定位**：
在定位階段，透過長短期記憶網路 (LSTM) 學習移動軌跡的時間序列變化趨勢。推論時僅需輸入當前視覺特徵向量，即可預測對應的虛擬地圖節點位置。

## 📁 專案架構

```bash
pre-commit install
```

---

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

## 使用說明

**執行任何工具前，請先至** `configs/config.yaml` **確認對應區塊的路徑與參數是否正確。**
\> 💡 **進階設定**：部分步驟（如擷取影片片段、調整 Node 數、標記測試集等）需要手動檢視與設定。詳情請務必參閱 `需手動處理的步驟.md`。

### ✏️專題檔案重現SOP

以下詳述了如何從零開始，透過腳本工具重現完整的純視覺定位與導航系統。

### 1. 製作訓練資料集

- **檔案放置結構說明**：
在開始製作資料集前，請將原始錄製的影片依照特定結構放置於 `Database/video/Original/` 目錄下：
    - **直線路徑 (如** `直線A`**,** `直線B` **等)**：資料夾內需再區分 `順向` 與 `逆向` 兩個子資料夾，並將對應影片 (.mp4) 放入。
    - **旋轉路徑 (如** `旋轉A`**,** `旋轉B` **等)**：不需建立子資料夾，直接將影片 (.mp4) 放入該路徑資料夾內即可。

    📁 結構範例如下：

    ```
    Database/video/Original/
    ├── 直線A/
    │   ├── 順向/
    │   │   ├── IMG_13_new.mp4
    │   │   └── IMG_14_new.mp4
    │   └── 逆向/
    │       ├── IMG_1_new.mp4
    │       ├── IMG_2_new.mp4
    │       └── IMG_3_new.mp4
    └── 旋轉A/
        ├── IMG_2629.mp4
        └── IMG_2630.mp4
    ```

- **1-1. 資料預處理**：針對原始影片進行降畫質、降取樣。
\> 執行：`python scripts/tku/visual_navigation_system/01_preprocess_videos.py`
- **1-2. 打包資料集**：將 `.mp4` 轉換成 Tensor `.pt` 檔，並進行隨機抽樣合併成訓練資料集。
\> 執行：`python scripts/tku/visual_navigation_system/02_prepare_dataset.py`

### 2. 訓練 AutoEncoder 模型

- **2-1. 檢視模型架構**：若需更改架構，請修改 `src/agilab/tku/visual_navigation_system/models.py`。
- **2-2. 檢視訓練邏輯**：訓練迴圈與 Loss Function 封裝於 `src/agilab/tku/visual_navigation_system/trainer.py`。
- **2-3. 執行模型訓練**：
\> 執行：`python scripts/tku/visual_navigation_system/03_train.py`
- **2-4. 測試還原效果** (可選)：
    - 測試 `.mp4` 影像還原（需先修改 `config.yaml` 路徑）：
    \> 執行：`python scripts/tku/visual_navigation_system/04_inference.py --mode video`
    - 測試 `.npy` 特徵向量還原（需先修改 `config.yaml` 路徑）：
    \> 執行：`python scripts/tku/visual_navigation_system/04_inference.py --mode feature`

### 3. 製作虛擬地圖 (最佳影片)

- **3-1. 旋轉影片預處理** *(僅針對旋轉影片)*：
    - 倒轉影片以調整旋轉方向：
    \> 執行：`python scripts/tku/visual_navigation_system/05_reverse_video.py --input "您的影片.mp4" --output "倒轉後的影片.mp4"`
    - 擷取特定片段以保留特定旋轉角度：

        \> 請將檔案儲存至：`Database\video\Preprocessed_resize_downsampling_videos`

        \> 執行：`python scripts/tku/visual_navigation_system/06_capture_video.py --input "原始影片.mp4" --output "擷取影片.mp4" --start <擷取起點 frame 數> --end <擷取終點 frame 數>`

- **3-2. 對齊影片**：將資料夾內所有相同路徑影片均勻下取樣對齊。
\> 執行：`python scripts/tku/visual_navigation_system/07_uniform_downsample.py`
- **3-3. 抽取特徵**：使用 Encoder 將所有影片編碼並儲存成 `.npy` 檔案。
\> 執行：`python scripts/tku/visual_navigation_system/08_extract_features.py`
- **3-4. 製作最佳影片**：取前 X 部影片製作最佳影片，其餘作為測試集。
\> 執行：`python scripts/tku/visual_navigation_system/09_make_best_video.py`
- **3-5. 調整節點 (Node) 數**：將每個「直線」路徑的最佳影片 frame 數（即為 Node 數），依照具有比例尺的室內地圖的歐式距離（即為現實的距離）比例進行線性內插補 frame 或下取樣。
    - 可先查看每個路徑目前的 Node 數：
    \> 執行：`python scripts/tku/visual_navigation_system/10_check_npy_shape.py --input "Map\Best_Video\Best_Video_Features"`
    - 計算每個路徑應有的 Node 數（詳見 `docs\tku_visual_navigation\最佳影片的節點數推導過程.md`）。
    - 將最佳影片實際 frame 數，平均下取樣或線性內插至該有的 frame 數：

        \> 執行：`python scripts/tku/visual_navigation_system/11_adjust_best_video_length.py --frames <目標數字>`

- **3-6. 建立全局特徵庫**：若沒有全路徑影片，串接所有路徑的 Best_Video 作為基本的 Graph 算準確率。
\> 執行：`python scripts/tku/visual_navigation_system/12_concat_best_videos.py`
- **3-7. 人工微調**：判斷是否需要微調串接結果。

### 4. 定位模型之 LSTM 訓練與預測評估

將預錄好的全路徑影片 Resize、Downsampling 後，當作測試影片，用來比對最佳影片（即為虛擬地圖）計算定位準確率

#### 階段 A：貪婪演算法預測 (Baseline)

- **4-1. 測試影片前處理**：將預錄的全路徑測試影片進行 Resize 與 Downsampling 後，儲存於：`Map/Full Path Test Video`。
\> 執行：`python scripts/tku/visual_navigation_system/01_preprocess_videos.py`
- **4-2. 紀錄路徑資訊**：將測試影片的每個路徑的 frame 數、總 frame 數，手動紀錄至：`Map\Full Path Test Video\Full_Path_TestVideo_information.txt`。
- **4-3. 執行貪婪法比對**：將要測試的全路徑影片，用貪婪法比對最佳影片 `.npy` 檔，選出最相似的特徵作為預測，並將預測結果畫出來，此準確率將作為後續的 Baseline。
\> 執行：`python scripts/tku/visual_navigation_system/13_greedy_prediction.py`
    - **【貪婪法定位與繪圖原理詳解】**：
        1. **路徑點標註**：輸入的測試影片必須手動標註哪些 frame 是「切換路徑」的節點（紀錄於 `Map\Full Path Test Video\Full_Path_TestVideo_information.txt` 中）。
        2. **路徑特徵映射**：使用映射函數，將「輸入測試影片」與「最佳影片」中的「轉彎路徑」皆視為 1 個獨立的點。
        3. **繪製與誤差區間**：畫圖時的實際位置，是根據輸入影片未映射前的 frame 分布來決定。即使每個路徑行走速度不同也無妨，但整部影片的誤差容許區間必須保持一致。
        4. **準確率計算方式**：
            - 將輸入影片第 X frame 去比對最佳影片，得到最相似的第 Y frame。
            - 將第 X frame 與第 Y frame 分別透過映射函數，從「原始影格索引」轉換為考慮了「直線/旋轉」的「實際進度索引」。
            - 將輸入影片映射後的結果，針對同個路段的 frame 數進行正規化（縮放對齊至最佳影片映射後的尺度）。
            - 相減計算誤差，若誤差小於「正負 5% 最佳影片映射後的總 frame 數」，即判定為預測正確。
            - **最終準確率** = 預測正確的 frame 數 / 輸入影片的總 frame 數。
        5. **評估意義**：將貪婪法的準確率記錄下來，未來可與 LSTM 預測的準確率比較。若差距很小，代表 LSTM 可能欠擬合；若貪婪法準確率本來就很低，則代表前方的 AutoEncoder 萃取特徵能力需要改進。

#### 階段 B：LSTM 增強定位模型

- **4-4. LSTM 模型訓練**：為了加強定位效果，引入 LSTM 模型學習時序特徵。
    - **【準備訓練集】** (輸入 X 為連續 9 frame 的特徵，Label Y 為對齊最佳影片後的第 t 個索引特徵)：
        1. 拍攝特定路段影片（例如路徑 ABCD）並進行 Resize、Downsampling，經過 Encoder 編碼後儲存成 `.npy` 放置於 `Database\Data\Full Path Tensor`。
        \> 先執行：`python scripts/tku/visual_navigation_system/01_preprocess_videos.py`

            \> 再執行：`python scripts/tku/visual_navigation_system/08_extract_features.py`

        2. 手動紀錄每個 `.npy` 中各路徑的 frame 數與總 frame 數，並儲存至對應的 `Database\Data\Full Path Tensor\<.npy 的檔名>_video_information.txt`。*(注意：路徑名稱需與最佳影片定義的名稱完全相同)*。
        3. 撰寫 `Full_Path_tensor_info.txt` 紀錄所有訓練檔案清單（格式為 `[檔名, 總frame數]`，每檔一行）。
    - **【執行實際訓練】**：
    \> 執行：`python scripts/tku/visual_navigation_system/14_train_lstm.py`
        - **原理**：將連續 9 個 frame 的特徵向量輸入 LSTM，LSTM 會預測第 10 個 frame 的特徵向量。接著將此預測出的特徵與最佳影片的「所有特徵」進行貪婪法比對，選出最相似的特徵作為「最終預測特徵」。最後計算該「預測特徵」與「實際答案（最佳影片的第 t 幀特徵）」之間的 MSE (Mean Squared Error) 當作 Loss 進行模型訓練。
- **4-5. 執行 LSTM 預測**：結合 LSTM 與貪婪法的最終定位預測。
\> 執行：`python scripts/tku/visual_navigation_system/15_lstm_prediction.py`
    - **【LSTM 預測原理與準確率計算】**：
        1. 將輸入測試影片的前 9 幀輸入 LSTM，預測出第 10 幀 (X 幀) 的特徵編碼。
        2. 將該預測編碼與最佳影片比對，找出最相似的第 Y 幀。
        3. 將第 X 幀與第 Y 幀分別經過映射函數（轉換為實際進度索引）。
        4. 將輸入影片映射後的結果，針對同路段 frame 數標準化至最佳影片的尺度。
        5. 相減計算誤差，若誤差小於「正負 5% 最佳影片映射後的總 frame 數」，則預測正確。
        6. **最終準確率** = 預測正確的 frame 數 / (輸入影片的總 frame 數 - 9)。
    - **【進階優化】** (目前預設關閉，可視需求開啟)：
        - **動態投票機制 (Voting Mechanism)**：若 LSTM 當前預測 frame 的位置，距離上一 frame 的預測位置跳躍大於一定閥值（不合理移動），則強制維持上一 frame 的預測位置以穩定軌跡。
        - *(若要啟用，請至 `configs/config.yaml` 裡的 `lstm` 區塊，將 `enable_voting_mechanism` 改為 `True`)*。

### 5. 建立 Graph 與導航視覺化 (Graph)

- **5-1. 建立節點邊緣 (Edges)**：手動建立各個不同路段最佳影片之間銜接的 frame 作為跨影片的邊緣。
    - **5-1-1. 尋找銜接點小工具**：使用本機端 GUI 工具逐個 frame 查看影片以決定銜接點。
    \> 執行：`python scripts/tku/visual_navigation_system/18_view_frame.py` (注意：僅支援有桌面環境的本地端執行)
    - 決定後請至 `config.yaml` 修改 `graph.cross_video_edges` 區塊。

        \> 執行：`python scripts/tku/visual_navigation_system/16_visualize_graph.py`

- **5-2. 最短路徑導航**：在建立好的 Graph 中，透過 BFS 演算法尋找起點與終點之間的最短路徑。
\> 執行：`python scripts/tku/visual_navigation_system/17_shortest_path.py --start_feat <起始點的.npy資料夾路徑> --end_feat <終點的.npy資料夾路徑>`

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

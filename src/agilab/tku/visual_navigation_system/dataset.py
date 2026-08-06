import glob
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


def get_dataloader(data_path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    SOP 步驟 2-3: 載入預處理好的 Tensor 資料集並建立 DataLoader。

    原始資料格式應為 torch.Size([N_frames, H, W, C])。
    載入後會將其 Permute 轉換為 PyTorch 標準的 [N_frames, C, H, W] 格式。

    Args:
        data_path (str): 預處理好的 .pt 檔案路徑。
        batch_size (int): 批次大小。
        shuffle (bool, optional): 是否在每個 Epoch 打亂資料。預設為 True。

    Returns:
        DataLoader: PyTorch DataLoader 實例，用以提供訓練資料。
    """
    # 讀取資料集
    # 資料形狀: torch.Size([n_frame, 108, 192, 3])
    data = torch.load(data_path, weights_only=True)

    # 轉為 [num_samples, C, H, W]，其中 C=3 (RGB)
    data = data.permute(0, 3, 1, 2)

    # 建立 Dataset
    dataset = TensorDataset(data)

    # 建立 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


class SegmentAligner:
    """
    負責載入與解析段落資訊，並提供不同影片間的幀數對齊功能 (給 LSTM 訓練使用)。
    """

    def __init__(self, info_path):
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"檔案不存在: {info_path}")
        self.info_path = info_path
        self.segments = self._parse_info_file_to_dict()

    def _parse_info_file_to_dict(self):
        segs_dict = OrderedDict()
        raw_s = 0
        with open(self.info_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip("[] \n'\"") for p in line.split(",")]
                if len(parts) >= 2:
                    name, length = parts[0], int(parts[1])
                    segs_dict[name] = {
                        "name": name,
                        "raw_start": raw_s,
                        "raw_end": raw_s + length - 1,
                        "length": length,
                    }
                    raw_s += length
        return segs_dict

    def find_segment_for_frame(self, raw_frame_idx):
        for seg_info in self.segments.values():
            if seg_info["raw_start"] <= raw_frame_idx <= seg_info["raw_end"]:
                return seg_info
        return None

    @staticmethod
    def align_frame(input_frame_idx, input_aligner, reference_aligner):
        in_seg = input_aligner.find_segment_for_frame(input_frame_idx)
        if not in_seg:
            return None
        ref_seg = reference_aligner.segments.get(in_seg["name"])
        if not ref_seg:
            return None
        progress = (
            (input_frame_idx - in_seg["raw_start"]) / (in_seg["length"] - 1)
            if in_seg["length"] > 1
            else 0.0
        )
        aligned_idx = (
            ref_seg["raw_start"] + progress * (ref_seg["length"] - 1)
            if ref_seg["length"] > 1
            else ref_seg["raw_start"]
        )
        return aligned_idx


class AlignedSequenceDataset(Dataset):
    """
    將輸入的特徵陣列根據對齊資訊轉換為序列預測標籤 (X -> Y)，
    用以訓練 LocationLSTM。
    """

    def __init__(
        self, data_dir, info_dir, ref_info_path, best_video_tensor, seq_length
    ):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.samples = []  # 存放 (video_idx, seq_start_idx) 指針
        self.input_aligners = []  # 存放每個輸入影片的 SegmentAligner 物件

        # 1. 創建參考影片的對齊工具
        self.ref_aligner = SegmentAligner(ref_info_path)

        if not os.path.exists(best_video_tensor):
            raise FileNotFoundError(
                f"參考影片的特徵檔案 (.npy) 不存在: {best_video_tensor}"
            )
        self.ref_tensor = torch.from_numpy(np.load(best_video_tensor)).float()

        # 2. 為每個輸入影片創建對齊工具並建立樣本指針
        info_files = sorted(
            glob.glob(os.path.join(info_dir, "*_video_information.txt"))
        )
        print(f"找到 {len(info_files)} 部影片檔案")

        for video_idx, info_path in enumerate(tqdm(info_files, desc="處理影片")):
            aligner = SegmentAligner(info_path)
            self.input_aligners.append(aligner)
            total_frames = list(aligner.segments.values())[-1]["raw_end"] + 1
            if total_frames > seq_length:
                for i in range(total_frames - seq_length):
                    self.samples.append((video_idx, i))

        if not self.samples:
            raise ValueError("資料不足以創建任何訓練序列！")

        # 快取機制，只保留最近載入的輸入影片特徵
        self.cached_input_idx = -1
        self.cached_input_tensor = None
        print(f"資料集準備完成！總共包含 {len(self.samples)} 個訓練樣本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # a. 根據索引獲取指針
        video_idx, seq_start_idx = self.samples[idx]

        # b. 使用快取機制載入【輸入影片】的特徵
        if video_idx != self.cached_input_idx:
            input_aligner = self.input_aligners[video_idx]
            video_name = os.path.basename(input_aligner.info_path).replace(
                "_video_information.txt", ""
            )
            npy_path = os.path.join(self.data_dir, f"IMG_{video_name}.npy")
            # 若 IMG_ 前綴的檔案不存在，嘗試不帶前綴的檔名
            if not os.path.exists(npy_path):
                npy_path = os.path.join(self.data_dir, f"{video_name}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(
                    f"找不到特徵檔案: IMG_{video_name}.npy \
                        或 {video_name}.npy (於 {self.data_dir})"
                )
            self.cached_input_tensor = torch.from_numpy(np.load(npy_path)).float()
            self.cached_input_idx = video_idx
        input_tensor = self.cached_input_tensor

        # c. 切分出輸入序列 X
        x = input_tensor[seq_start_idx : (seq_start_idx + self.seq_length)]

        # d. 計算【對齊後的標籤 Y】
        label_original_idx = seq_start_idx + self.seq_length
        input_aligner = self.input_aligners[video_idx]

        aligned_ref_idx_float = SegmentAligner.align_frame(
            label_original_idx, input_aligner, self.ref_aligner
        )

        # 四捨五入到最近的整數幀
        aligned_ref_idx = int(round(aligned_ref_idx_float))

        # 確保索引不越界
        aligned_ref_idx = max(0, min(aligned_ref_idx, len(self.ref_tensor) - 1))

        # 從【參考影片】的特徵中取出最終的標籤 Y
        y = self.ref_tensor[aligned_ref_idx]

        return x, y

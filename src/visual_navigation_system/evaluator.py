import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from visual_navigation_system.models import Encoder


class FeatureExtractor:
    """
    負責載入 Encoder 模型，將影片逐幀轉換為特徵向量 (512維)。
    """

    def __init__(self, encoder_weights_path: str, device: torch.device):
        """
        初始化並載入預訓練的 Encoder 權重。
        """
        self.device = device
        self.encoder = Encoder().to(device)

        if os.path.exists(encoder_weights_path):
            state_dict = torch.load(
                encoder_weights_path, map_location=device, weights_only=True
            )
            # 處理可能帶有 'encoder.' 前綴的權重
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            # 若過濾後為空 (可能直接儲存了 Encoder 的權重)，直接使用原始 state_dict
            if not encoder_state_dict:
                encoder_state_dict = state_dict
            # 因為直接存 Encoder 或 AutoEncoder 可能結構不同，加入 strict=False
            missing_keys, unexpected_keys = self.encoder.load_state_dict(
                encoder_state_dict, strict=False
            )
            print(f"載入 Encoder 權重：{encoder_weights_path}")
        else:
            print(f"未找到 Encoder 權重：{encoder_weights_path}，使用隨機初始化。")

        self.encoder.eval()

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """讀取影片所有幀。"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def compute_feature_vectors(
        self, frames: List[np.ndarray], target_size: Tuple[int, int] = (192, 108)
    ) -> np.ndarray:
        """對幀列表計算特徵向量。"""
        features = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, target_size)
            normalized = resized.astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(normalized)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                f_vector = self.encoder(tensor)

            features.append(f_vector.squeeze(0).cpu().numpy())

        return np.array(features)

    def process_video(
        self, video_path: str, output_npy_path: Optional[str] = None
    ) -> np.ndarray:
        """處理單一影片並回傳/儲存特徵陣列。"""
        frames = self.extract_frames(video_path)
        features = self.compute_feature_vectors(frames)

        if output_npy_path:
            os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
            np.save(output_npy_path, features)
            print(f"已儲存 {output_npy_path}")

        return features

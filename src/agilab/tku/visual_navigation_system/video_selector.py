import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class BestVideoSelector:
    """
    負責從多部下取樣影片中，根據特徵挑選出每一幀最穩定的影像，並組合成「最佳影片」。
    """

    def __init__(self, target_size: Tuple[int, int] = (192, 108)):
        """
        初始化最佳影片選擇器。

        Args:
            target_size (Tuple[int, int]): 影片幀的目標解析度 (width, height)。
        """
        self.target_size = target_size

    def select_best_feature_for_frame(
        self, feature_list: List[np.ndarray]
    ) -> Tuple[int, float]:
        """
        對於同一幀（來自不同影片）的特徵向量進行比對，利用上三角計算每個向量的平均歐式距離分數，
        選出分數最低的那個（與其他影片最相似，即最穩定的代表）。

        Args:
            feature_list (List[np.ndarray]): 每個元素為某部影片該幀的特徵向量。

        Returns:
            Tuple[int, float]: (最佳影片索引, 該幀的最佳分數)
        """
        num = len(feature_list)
        scores = np.zeros(num)
        counts = np.zeros(num)

        for i in range(num):
            for j in range(i + 1, num):
                d = float(np.linalg.norm(feature_list[i] - feature_list[j]))
                scores[i] += d
                scores[j] += d
                counts[i] += 1
                counts[j] += 1

        # 避免除以 0 的錯誤
        counts[counts == 0] = 1
        avg_scores = scores / counts
        best_idx = int(np.argmin(avg_scores))
        return best_idx, avg_scores[best_idx]

    def get_frame_from_video(
        self, video_path: str, frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        從影片中讀取指定 frame_idx 的幀，並調整尺寸為 target_size (width, height)。

        Args:
            video_path (str): 影片路徑。
            frame_idx (int): 目標幀索引。

        Returns:
            np.ndarray: 讀取並調整大小後的影像 (BGR 格式)，失敗回傳 None。
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"無法從 {video_path} 讀取幀 {frame_idx}")
            return None

        # 調整尺寸
        frame = cv2.resize(frame, self.target_size)
        return frame

    def create_best_video(
        self,
        feature_folder: str,
        video_folder: str,
        output_feature_path: str,
        output_path: str,
        max_videos: int = 22,
    ) -> None:
        """
        遍歷資料夾中的特徵檔與影片檔，逐幀挑選最佳影像並組合為新的影片與特徵檔案。

        Args:
            feature_folder (str): 存放影片特徵的 .npy 檔目錄。
            video_folder (str): 存放原始影片的 .mp4 檔目錄。
            output_feature_path (str): 最佳特徵輸出的 .npy 檔案路徑。
            output_path (str): 最佳影片輸出的 .mp4 檔案路徑。
            max_videos (int): 最多取幾部影片參與比對。
        """
        npy_paths = sorted(glob.glob(os.path.join(feature_folder, "*.npy")))[
            :max_videos
        ]
        video_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))[
            :max_videos
        ]

        num_videos = len(npy_paths)
        if num_videos == 0 or len(video_paths) == 0:
            raise ValueError("找不到特徵檔或影片檔，請檢查資料夾路徑！")

        if num_videos != len(video_paths):
            print("特徵檔數量與影片數量不一致！請確認檔案是否對齊。")

        print(f"使用 {num_videos} 部影片生成最佳影片")

        # 載入所有影片的特徵，假設每個 npy 檔 shape 為 (num_frames, 512)
        all_videos_features = []
        for npy_path in npy_paths:
            features = np.load(npy_path)
            all_videos_features.append(features)

        # 假設所有影片幀數一致
        num_frames = all_videos_features[0].shape[0]
        print(f"每部影片共有 {num_frames} 幀")

        # 讀取第一部影片的 FPS 以保持播放速度一致
        cap = cv2.VideoCapture(video_paths[0])
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"原始影片 FPS：{orig_fps}")

        best_video_features = []
        best_frames = []
        best_scores_list = []

        for frame_idx in tqdm(range(num_frames), desc="選取最佳幀"):
            # 從每部影片取出該幀的特徵向量
            frame_features = [
                all_videos_features[v][frame_idx] for v in range(num_videos)
            ]
            best_vid_idx, best_score = self.select_best_feature_for_frame(
                frame_features
            )
            best_scores_list.append(best_score)

            # 檢查索引是否在合法範圍內
            if best_vid_idx < 0 or best_vid_idx >= len(video_paths):
                print(f"幀 {frame_idx} 選出的影片索引 {best_vid_idx} 超出範圍")
                continue

            # 利用影片索引及幀索引讀取該影片的原始影像
            video_path = video_paths[best_vid_idx]
            best_frame = self.get_frame_from_video(video_path, frame_idx)
            if best_frame is None:
                continue

            best_frames.append(best_frame)
            best_video_features.append(all_videos_features[best_vid_idx][frame_idx])

        # 檢查是否有選取到最佳幀
        if len(best_frames) == 0:
            print("未選取到任何幀，請檢查資料")
            return

        # 儲存最佳影片特徵
        os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
        best_video_features_np = np.array(best_video_features)
        np.save(output_feature_path, best_video_features_np)
        print(f"最佳影片特徵已存儲為 {output_feature_path}")

        # 使用 OpenCV 將所有最佳幀拼接成新影片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width, _ = best_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))
        for frame in best_frames:
            out.write(frame)
        out.release()

        print(f"最佳影片已存檔至: {output_path}")
        print(
            f"共處理 {len(best_scores_list)} 幀，\
                平均 best_score = {np.mean(best_scores_list):.4f}\n"
        )

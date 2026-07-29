import os

import cv2
import numpy as np
import torch

from agilab.tku.visual_navigation_system.models import Encoder


class GreedyPredictor:
    """
    負責載入 Encoder 模型與最佳影片特徵庫，
    並對輸入的單張影像進行貪婪法特徵比對 (尋找最近鄰)。
    """

    def __init__(self, model_path, db_features_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 1. 載入 Encoder 模型
        self.model = Encoder().to(self.device)
        if os.path.exists(model_path):
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # 因為原本儲存的是 AutoEncoder 的權重，需要過濾出 encoder 的部分
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            if not encoder_state_dict:
                encoder_state_dict = state_dict

            self.model.load_state_dict(encoder_state_dict, strict=False)
        else:
            print(f"找不到模型權重，將使用隨機初始化權重: {model_path}")

        self.model.eval()

        # 2. 載入資料庫特徵
        self.db_features = torch.tensor(
            np.load(db_features_path), dtype=torch.float32, device=self.device
        )

    def transform(self, frame):
        """影像前處理: resize 到 192x108，並正規化成 tensor"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (192, 108))
        img_normalized = resized.transpose(2, 0, 1) / 255.0
        return (
            torch.tensor(img_normalized, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

    def predict(self, frame):
        """
        輸入一張影像，回傳:
        1. raw_db: 最相似的資料庫索引
        2. best_match_feature: 該索引的特徵向量
        3. min_dist: 與該特徵的最小歐氏距離
        """
        inp = self.transform(frame)
        with torch.no_grad():
            feat = self.model(inp)
            dists = torch.norm(self.db_features - feat, dim=1)
            raw_db = int(torch.argmin(dists).item())
            best_match_feature = self.db_features[raw_db]
            min_dist = float(torch.min(dists).item())

        return raw_db, best_match_feature, min_dist


def load_segment_info(info_path):
    """
    讀取 [name, frame_count, cum_frame] 格式的資訊檔
    返回 segment_info 列表，用於映射真實座標
    """
    segs, raw_s, act_s = [], 0, 0
    if not os.path.exists(info_path):
        return segs

    with open(info_path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip("[] \n'\"") for p in line.split(",")]
            if len(parts) < 2:
                continue
            name, length = parts[0], int(parts[1])
            is_rot = "旋轉" in name
            act_len = 1 if is_rot else length
            segs.append(
                {
                    "name": name,
                    "raw_start": raw_s,
                    "raw_end": raw_s + length - 1,
                    "actual_start": act_s,
                    "actual_length": act_len,
                    "is_rotation": is_rot,
                }
            )
            raw_s += length
            act_s += act_len
    return segs


def map_to_actual_idx(raw_idx, segs):
    """將模型預測的原始索引，轉換成去除旋轉冗餘後的實際索引"""
    for seg in segs:
        if seg["raw_start"] <= raw_idx <= seg["raw_end"]:
            return (
                seg["actual_start"]
                if seg["is_rotation"]
                else seg["actual_start"] + (raw_idx - seg["raw_start"])
            )
    return None

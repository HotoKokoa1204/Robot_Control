import os
import random
import sys

import torch
import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
from agilab.tku.visual_navigation_system.video_utils import video_to_tensor


def main():
    # 1. 載入設定檔
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = config.get("dataset_preparation")
    if not ds_config:
        raise ValueError("設定檔中找不到 dataset_preparation 區塊。")

    root_dir = ds_config["input_dir"]
    straight_video_count = ds_config["straight_video_count"]
    rotation_video_count = ds_config["rotation_video_count"]
    output_tensor_path = ds_config["output_tensor_path"]
    output_info_path = ds_config["output_info_path"]
    output_frame_count_path = ds_config["output_frame_count_path"]

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"找不到影片資料夾: {root_dir}")

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_tensor_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_info_path), exist_ok=True)

    # 2. 搜尋影片並根據路徑結構分類
    # 結構: {'路徑 (例如: 直線A)': {'順向/逆向/旋轉': [影片路徑1, 影片路徑2]}}
    train_videos = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for video_file in filenames:
            if video_file.lower().endswith(".mp4"):
                video_path = os.path.join(dirpath, video_file)
                parts = video_path.split(os.sep)

                # 這裡的邏輯與原本 1Step_Process_Video.ipynb 一致
                parent = parts[-2]
                if parent not in ("順向", "逆向"):
                    level_1, level_2 = parent, "旋轉"
                else:
                    level_1, level_2 = parts[-3], parent

                if level_1 not in train_videos:
                    train_videos[level_1] = {}
                if level_2 not in train_videos[level_1]:
                    train_videos[level_1][level_2] = []

                train_videos[level_1][level_2].append(video_path)

    # 3. 執行抽樣
    selected_files = []
    for level_1, level_2_dict in train_videos.items():
        for level_2, videos in level_2_dict.items():
            n_train = (
                rotation_video_count if level_2 == "旋轉" else straight_video_count
            )

            if len(videos) < n_train:
                print(
                    f"{level_1}/{level_2} 的影片數量不足 ({len(videos)} < {n_train})\
                        ，將使用所有可用影片。"
                )
                sampled = videos
            else:
                sampled = random.sample(videos, n_train)

            selected_files.extend(sampled)

    # 打亂全部抽出的影片順序 (可選，原程式碼是分組的，但這邊打亂有助於訓練)
    random.shuffle(selected_files)

    print(f"總共抽出 {len(selected_files)} 部影片，開始轉換並合併 Tensor...")

    # 4. 讀取並合併 Tensor
    final_merged_tensor = None

    with open(output_info_path, "w", encoding="utf-8") as f_info, open(
        output_frame_count_path, "w", encoding="utf-8"
    ) as f_frame:
        for idx, pt_file in enumerate(selected_files, start=1):
            print(f"[{idx}/{len(selected_files)}] 處理影片: {pt_file}")
            try:
                tensor = video_to_tensor(pt_file)
            except Exception as e:
                print(f"轉換失敗跳過: {e}")
                continue

            frame_count = tensor.shape[0]
            pt_filename = f"single_video_tensor_{idx}.pt"

            # 從路徑解析相對資訊 (例如: 直線A_順向 / XXX.mp4)
            parts = pt_file.split(os.sep)
            relative_info = f"{parts[-3]}_{parts[-2]} / {parts[-1]}"

            # 寫入資訊
            f_info.write(f"[{frame_count}, {pt_filename}, {relative_info}]\n")
            f_frame.write(f"[{frame_count}, {pt_filename}, {relative_info}]\n")

            # 合併
            if final_merged_tensor is None:
                final_merged_tensor = tensor
            else:
                final_merged_tensor = torch.cat([final_merged_tensor, tensor], dim=0)

    # 5. 儲存結果
    if final_merged_tensor is not None:
        print(f"\n所有影片合併完成，最終 Tensor 形狀為: {final_merged_tensor.shape}")
        print(f"正在儲存至: {output_tensor_path}")
        torch.save(final_merged_tensor, output_tensor_path)
        print("儲存成功！")
    else:
        print("合併失敗，沒有任何有效的 Tensor 產生。")


if __name__ == "__main__":
    main()

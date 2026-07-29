import argparse
import os

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="依照設定檔將多個最佳影片與特徵依序串接合併 (Concatenate)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="設定檔路徑"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"找不到設定檔: {args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    concat_cfg = config.get("concat_best_videos")
    if not concat_cfg:
        print("設定檔中找不到 'concat_best_videos' 區塊！")
        return

    video_root = concat_cfg.get("input_video_base_dir", "")
    feature_root = concat_cfg.get("input_feature_base_dir", "")
    output_dir = concat_cfg.get("output_dir", "")

    os.makedirs(output_dir, exist_ok=True)

    out_video_path = os.path.join(
        output_dir, concat_cfg.get("output_video_name", "Full_Path_Best_Video.mp4")
    )
    out_feature_path = os.path.join(
        output_dir, concat_cfg.get("output_feature_name", "Full_Path_Best_Video.npy")
    )
    out_info_path = os.path.join(
        output_dir,
        concat_cfg.get("output_info_name", "Full_path_video_information.txt"),
    )

    direction = concat_cfg.get("direction", "順向")
    line_names = concat_cfg.get("line_names", [])

    print("=" * 60)
    print(f"開始串接影片與特徵: {direction}")
    print(f"順序: {line_names}")
    print("=" * 60)

    video_writer = None
    all_features = []
    video_info = []

    for folder in line_names:
        # 如果是旋轉路段，檔名不包含方向
        if "旋轉" in folder:
            video_path = os.path.join(video_root, folder, f"{folder}.mp4")
            feature_path = os.path.join(feature_root, folder, f"{folder}.npy")
        else:
            video_path = os.path.join(video_root, folder, f"{folder}_{direction}.mp4")
            feature_path = os.path.join(
                feature_root, folder, f"{folder}_{direction}.npy"
            )

        if not os.path.exists(video_path) or not os.path.exists(feature_path):
            print(f"找不到影片或特徵，跳過: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法打開影片: {video_path}")
            continue

        # 第一支影片時初始化 VideoWriter
        if video_writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)
            frame_count += 1
        cap.release()

        video_name = os.path.basename(video_path).replace(".mp4", "")
        video_info.append([video_name, frame_count])

        features = np.load(feature_path)
        all_features.append(features)

    if not all_features:
        print("沒有成功讀取任何特徵檔案，串接失敗。")
        return

    # 將特徵陣列依照第一維度(frame數)合併
    final_features = np.concatenate(all_features, axis=0)
    np.save(out_feature_path, final_features)

    if video_writer:
        video_writer.release()

    # 輸出每個子影片長度的資訊
    with open(out_info_path, "w", encoding="utf-8") as f:
        cumulative = 0
        for name, count in video_info:
            cumulative += count
            f.write(f"[{name}, {count}, {cumulative}]\n")

    print(f"輸出影片：{out_video_path}")
    print(f"特徵檔案：{out_feature_path}")
    print(f"資訊紀錄：{out_info_path}")


if __name__ == "__main__":
    main()

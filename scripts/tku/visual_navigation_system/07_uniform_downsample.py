import argparse
import glob
import os
import sys

import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
import cv2
from agilab.tku.visual_navigation_system.video_utils import uniform_downsample_video


def get_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def process_directory(input_dir: str, output_dir: str, fixed_frames: int = None):
    video_files = glob.glob(os.path.join(input_dir, "*.*"))
    # 過濾只處理常見的影片格式
    video_files = [
        v for v in video_files if v.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    if len(video_files) == 0:
        print(f"資料夾 {input_dir} 中找不到支援的影片檔案！")
        return

    # 若未指定目標幀數，則尋找該資料夾中最短的幀數
    if fixed_frames is None:
        print(f"掃描 {len(video_files)} 支影片以尋找最短幀數...")
        video_frame_counts = {}
        for video in video_files:
            count = get_frame_count(video)
            if count > 0:
                video_frame_counts[video] = count

        if len(video_frame_counts) == 0:
            print("所有影片皆無法讀取！")
            return

        min_frame_count = min(video_frame_counts.values())
        print(f"決定目標幀數：{min_frame_count} 幀 (根據資料夾中最短的影片)")
    else:
        min_frame_count = fixed_frames
        print(f"指定目標幀數：{min_frame_count} 幀")

    os.makedirs(output_dir, exist_ok=True)

    for video in video_files:
        video_name = os.path.basename(video)
        output_path = os.path.join(output_dir, video_name)
        uniform_downsample_video(video, output_path, min_frame_count)

    print(f"資料夾處理完成！所有影片皆統一為 {min_frame_count} 幀。")


def main():
    parser = argparse.ArgumentParser(
        description="依照設定檔批次進行全資料夾的均勻下採樣"
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

    ud_config = config.get("uniform_downsampling")
    if not ud_config:
        print("設定檔中找不到 'uniform_downsampling' 區塊！")
        return

    input_base_dir = ud_config.get("input_base_dir", "")
    output_base_dir = ud_config.get("output_base_dir", "")
    straight_paths = ud_config.get("straight_paths", [])
    rotation_paths = ud_config.get("rotation_paths", [])

    print("=" * 50)
    print(f"輸入總路徑: {input_base_dir}")
    print(f"輸出總路徑: {output_base_dir}")
    print("=" * 50)

    # 1. 處理直線影片 (分順向、逆向)
    for path_name in straight_paths:
        for direction in ["順向", "逆向"]:
            in_dir = os.path.join(input_base_dir, f"直線{path_name}", direction)
            out_dir = os.path.join(output_base_dir, f"直線{path_name}", direction)

            if os.path.exists(in_dir):
                print(f"\n正在處理直線路徑: 直線{path_name}/{direction}")
                process_directory(in_dir, out_dir)
            else:
                print(f"\n找不到資料夾: {in_dir}，跳過。")

    # 2. 處理旋轉影片
    for path_name in rotation_paths:
        in_dir = os.path.join(input_base_dir, f"旋轉{path_name}")
        out_dir = os.path.join(output_base_dir, f"旋轉{path_name}")

        if os.path.exists(in_dir):
            print(f"\n正在處理旋轉路徑: 旋轉{path_name}")
            process_directory(in_dir, out_dir)
        else:
            print(f"\n找不到資料夾: {in_dir}，跳過。")

    print("\n所有路徑均勻下採樣作業完成！")


if __name__ == "__main__":
    main()

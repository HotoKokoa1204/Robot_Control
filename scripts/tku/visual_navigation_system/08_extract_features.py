import argparse
import glob
import os
import sys

import torch
import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
from agilab.tku.visual_navigation_system.evaluator import FeatureExtractor


def process_directory(input_dir: str, output_dir: str, extractor: FeatureExtractor):
    video_files = glob.glob(os.path.join(input_dir, "*.*"))
    video_files = [
        v for v in video_files if v.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    if len(video_files) == 0:
        print(f"資料夾 {input_dir} 中找不到支援的影片檔案！")
        return

    os.makedirs(output_dir, exist_ok=True)

    for video in video_files:
        video_name = os.path.basename(video).rsplit(".", 1)[0]
        output_path = os.path.join(output_dir, f"{video_name}.npy")

        # 若不想覆蓋已有特徵，可以加檢查，但為了重現原本覆寫的邏輯，直接執行
        extractor.process_video(video, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="依照設定檔批次讀取影片，透過 Encoder 抽取成 .npy 特徵檔"
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

    # 讀取設定
    ef_config = config.get("extract_features")
    if not ef_config:
        print("設定檔中找不到 'extract_features' 區塊！")
        return

    input_base_dir = ef_config.get("input_base_dir", "")
    output_base_dir = ef_config.get("output_base_dir", "")
    straight_paths = ef_config.get("straight_paths", [])
    rotation_paths = ef_config.get("rotation_paths", [])

    # 取得模型權重路徑
    encoder_weights_path = ef_config.get("encoder_weights_path", "")
    if not encoder_weights_path:
        print(
            "找不到 AutoEncoder 權重路徑，請檢查 config.yaml 的\
                extract_features.encoder_weights_path"
        )
        return

    device_str = config.get("inference", {}).get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print(f"權重路徑: {encoder_weights_path}")
    print(f"輸入總路徑: {input_base_dir}")
    print(f"輸出總路徑: {output_base_dir}")
    print("=" * 50)

    # 實例化 FeatureExtractor
    extractor = FeatureExtractor(encoder_weights_path, device)

    # 1. 處理直線影片
    for path_name in straight_paths:
        for direction in ["順向", "逆向"]:
            in_dir = os.path.join(input_base_dir, f"直線{path_name}", direction)
            out_dir = os.path.join(output_base_dir, f"直線{path_name}", direction)

            if os.path.exists(in_dir):
                print(f"\n正在處理直線路徑: 直線{path_name}/{direction}")
                process_directory(in_dir, out_dir, extractor)
            else:
                print(f"\n找不到資料夾: {in_dir}，跳過。")

    # 2. 處理旋轉影片
    for path_name in rotation_paths:
        in_dir = os.path.join(input_base_dir, f"旋轉{path_name}")
        out_dir = os.path.join(output_base_dir, f"旋轉{path_name}")

        if os.path.exists(in_dir):
            print(f"\n正在處理旋轉路徑: 旋轉{path_name}")
            process_directory(in_dir, out_dir, extractor)
        else:
            print(f"\n找不到資料夾: {in_dir}，跳過。")


if __name__ == "__main__":
    main()

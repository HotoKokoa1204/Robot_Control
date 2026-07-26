import argparse
import os
import sys

import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_navigation_system.video_utils import adjust_video_and_npy_length


def main():
    parser = argparse.ArgumentParser(
        description="同步調整單一 .mp4 與 .npy 的長度 (智慧下採樣或線性內插補幀)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="設定檔路徑"
    )
    parser.add_argument("--frames", type=int, required=True, help="目標總幀數")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"找不到設定檔: {args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 讀取設定
    adj_config = config.get("adjust_best_video_length")
    if not adj_config:
        print("設定檔中找不到 'adjust_best_video_length' 區塊！")
        return

    input_mp4 = adj_config.get("input_mp4_path", "")
    input_npy = adj_config.get("input_npy_path", "")
    output_mp4 = adj_config.get("output_mp4_path", "")
    output_npy = adj_config.get("output_npy_path", "")

    target_frame = args.frames

    print("=" * 60)
    print(f"開始執行：單一影片與特徵同步對齊至 {target_frame} 幀")
    print(f"輸入影片: {input_mp4}")
    print(f"輸入特徵: {input_npy}")
    print("=" * 60)

    adjust_video_and_npy_length(
        input_mp4, input_npy, output_mp4, output_npy, target_frame
    )

    print("\n處理完成！")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

# 確保可正確載入 core 目錄下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_navigation_system.video_utils import capture_video_segment


def main():
    parser = argparse.ArgumentParser(
        description="擷取影片的特定幀區間 (取代舊版 Capture_Video.ipynb)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="輸入影片路徑 (.mp4 或 .mov)"
    )
    parser.add_argument("--output", type=str, required=True, help="輸出影片路徑 (.mp4)")
    parser.add_argument(
        "--start", type=int, required=True, help="要保留的起始幀 (從 0 開始算)"
    )
    parser.add_argument("--end", type=int, required=True, help="要保留的結束幀 (包含)")

    args = parser.parse_args()

    print("-" * 30)
    print(f"輸入影片: {args.input}")
    print(f"輸出影片: {args.output}")
    print(f"保留區間: 第 {args.start} 幀 ~ 第 {args.end} 幀")
    print("-" * 30)

    capture_video_segment(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()

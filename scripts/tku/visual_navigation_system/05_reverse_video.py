import argparse
import os
import sys

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
from agilab.tku.visual_navigation_system.video_utils import reverse_video


def main():
    parser = argparse.ArgumentParser(description="將影片倒轉 (Reverse Video)")
    parser.add_argument(
        "--input", type=str, required=True, help="輸入影片路徑 (.mp4 或 .mov)"
    )
    parser.add_argument("--output", type=str, required=True, help="輸出影片路徑 (.mp4)")

    args = parser.parse_args()

    print("-" * 30)
    print(f"輸入影片: {args.input}")
    print(f"輸出影片: {args.output}")
    print("-" * 30)

    reverse_video(args.input, args.output)


if __name__ == "__main__":
    main()

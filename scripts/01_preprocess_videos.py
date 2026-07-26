import os
import sys

import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_navigation_system.video_utils import downsample_video


def main():
    # 1. 載入設定檔
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_config = config.get("video")
    if not video_config:
        raise ValueError("設定檔中找不到 video 區塊。")

    input_dir = video_config.get("raw_video_dir")
    output_dir = video_config.get("processed_video_dir")
    target_fps = video_config.get("target_fps", 10)
    target_res = (
        video_config.get("new_width", 192),
        video_config.get("new_height", 108),
    )

    if not input_dir or not os.path.exists(input_dir):
        print(f"找不到原始影片資料夾 {input_dir}")
        return

    print(
        f"\n來源: {input_dir}\n輸出: {output_dir}\n目標解析度: {target_res}\
        , 目標 FPS: {target_fps}\n"
    )

    # 2. 遍歷資料夾進行處理
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".mp4", ".mov")):
                input_video_path = os.path.join(root, filename)

                # 構造輸出路徑，保持相對目錄結構
                relative_path = os.path.relpath(root, input_dir)
                output_subfolder = os.path.join(output_dir, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # 強制輸出為 .mp4
                output_filename = f"{os.path.splitext(filename)[0]}.mp4"
                output_video_path = os.path.join(output_subfolder, output_filename)

                if os.path.exists(output_video_path):
                    print(f"檔案已存在，跳過處理: {output_video_path}")
                    continue

                # 呼叫 video_utils 內的函數進行降取樣與縮放
                # downsample_video 已經包含了 resize 的功能
                print(f"處理中: {filename} ...")
                downsample_video(
                    input_video_path,
                    output_video_path,
                    target_fps=target_fps,
                    target_resolution=target_res,
                )

    print("\n所有影片批次處理完成！")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
from agilab.tku.visual_navigation_system.video_selector import BestVideoSelector


def main():
    parser = argparse.ArgumentParser(
        description="依照設定檔批次合成全路徑的最佳影片與特徵檔"
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

    bv_config = config.get("make_best_video")
    if not bv_config:
        print("設定檔中找不到 'make_best_video' 區塊！")
        return

    max_videos = bv_config.get("max_videos", 20)
    feat_base_dir = bv_config.get("input_feature_base_dir", "")
    vid_base_dir = bv_config.get("input_video_base_dir", "")
    out_vid_base = bv_config.get("output_best_video_base_dir", "")
    out_feat_base = bv_config.get("output_best_feature_base_dir", "")

    straight_paths = bv_config.get("straight_paths", [])
    rotation_paths = bv_config.get("rotation_paths", [])

    print("=" * 50)
    print(f"每條路徑取前 {max_videos} 部影片比對 (其餘將被視為測試集保留)")
    print("=" * 50)

    selector = BestVideoSelector()

    # 1. 處理直線影片
    for path_name in straight_paths:
        for direction in ["順向", "逆向"]:
            # 資料夾路徑
            feat_folder = os.path.join(feat_base_dir, f"直線{path_name}", direction)
            vid_folder = os.path.join(vid_base_dir, f"直線{path_name}", direction)

            # 儲存檔名例如: 直線A/直線A_順向.mp4
            out_vid_folder = os.path.join(out_vid_base, f"直線{path_name}")
            out_feat_folder = os.path.join(out_feat_base, f"直線{path_name}")
            os.makedirs(out_vid_folder, exist_ok=True)
            os.makedirs(out_feat_folder, exist_ok=True)

            out_vid_path = os.path.join(
                out_vid_folder, f"直線{path_name}_{direction}.mp4"
            )
            out_feat_path = os.path.join(
                out_feat_folder, f"直線{path_name}_{direction}.npy"
            )

            if os.path.exists(feat_folder) and os.path.exists(vid_folder):
                print(f"\n處理直線路徑: 直線{path_name}/{direction}")
                try:
                    selector.create_best_video(
                        feature_folder=feat_folder,
                        video_folder=vid_folder,
                        output_feature_path=out_feat_path,
                        output_path=out_vid_path,
                        max_videos=max_videos,
                    )
                except Exception as e:
                    print(f"處理失敗: {e}")
            else:
                print(f"\n找不到資料夾: {feat_folder} 或 {vid_folder}，跳過。")

    # 2. 處理旋轉影片
    for path_name in rotation_paths:
        feat_folder = os.path.join(feat_base_dir, f"旋轉{path_name}")
        vid_folder = os.path.join(vid_base_dir, f"旋轉{path_name}")

        out_vid_folder = os.path.join(out_vid_base, f"旋轉{path_name}")
        out_feat_folder = os.path.join(out_feat_base, f"旋轉{path_name}")
        os.makedirs(out_vid_folder, exist_ok=True)
        os.makedirs(out_feat_folder, exist_ok=True)

        out_vid_path = os.path.join(out_vid_folder, f"旋轉{path_name}.mp4")
        out_feat_path = os.path.join(out_feat_folder, f"旋轉{path_name}.npy")

        if os.path.exists(feat_folder) and os.path.exists(vid_folder):
            print(f"\n處理旋轉路徑: 旋轉{path_name}")
            try:
                selector.create_best_video(
                    feature_folder=feat_folder,
                    video_folder=vid_folder,
                    output_feature_path=out_feat_path,
                    output_path=out_vid_path,
                    max_videos=max_videos,
                )
            except Exception as e:
                print(f"處理失敗: {e}")
        else:
            print(f"\n找不到資料夾: {feat_folder} 或 {vid_folder}，跳過。")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)
from agilab.tku.visual_navigation_system.models import AutoEncoder


def process_video(video_path, autoencoder, output_path, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法打開影片：{video_path}")
        return

    # 獲取影片資訊
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 預處理影像
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = (
            torch.tensor(input_frame)
            .float()
            .div(255)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        # 經過 Encoder 和 Decoder 還原
        with torch.no_grad():
            f_vector, reconstructed_f = autoencoder(input_tensor)
            reconstructed_f = autoencoder.f_decoder(f_vector).squeeze(0).cpu()

        # 復原到影像格式
        restored_frame = reconstructed_f.permute(1, 2, 0).numpy()  # [H, W, 3]
        restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)

        # 寫入輸出影片
        restored_frame_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)
        out.write(restored_frame_bgr)

        frame_idx += 1
        print(f"處理中：{frame_idx}/{frame_count} 幀", end="\r")

    # 釋放資源
    cap.release()
    out.release()
    print(f"\n影片已儲存至：{output_path}")


def process_npy(
    npy_path, autoencoder, output_path, device, frame_size=(192, 108), fps=10, buffer=0
):
    # 載入特徵
    features = np.load(npy_path)  # 假設 shape: [N, 512]
    print(f"特徵檔案：{npy_path}，共 {features.shape[0]} 幀")

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # 寫入 buffer 數量的全白影像
    if buffer > 0:
        white_frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
        for i in range(buffer):
            out.write(white_frame)
        print(f"已寫入 {buffer} 幀全白影像作為起始 buffer")

    # 對每個特徵進行 Decoder 還原
    for idx, feat in enumerate(features):
        feat_tensor = torch.tensor(feat).float().unsqueeze(0).to(device)  # [1, 512]

        with torch.no_grad():
            recon = autoencoder.f_decoder(feat_tensor).squeeze(0).cpu()

        # 轉回影像格式
        restored_frame = recon.permute(1, 2, 0).numpy()  # [H, W, 3]
        restored_frame = (restored_frame * 255).clip(0, 255).astype(np.uint8)
        restored_frame_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)
        out.write(restored_frame_bgr)

        print(f"處理中：{idx + 1 + buffer}/{features.shape[0] + buffer} 幀", end="\r")

    out.release()
    print(f"\n影片已儲存至：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="測試 AutoEncoder 的還原效果")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "feature"],
        default="video",
        help="選擇輸入模式：video 或 feature",
    )
    args = parser.parse_args()

    # 載入設定檔
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 讀取 config.yaml 中的設定
    device_str = config.get("inference", {}).get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model_path = config.get("inference", {}).get("autoencoder_weights", "")
    output_folder = config.get("inference", {}).get(
        "decoder_output_folder", "./Decoder_after_video"
    )

    print(f"使用的權重：{model_path}")

    # 初始化模型並加載權重
    autoencoder = AutoEncoder().to(device)
    if os.path.exists(model_path):
        autoencoder.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    else:
        print(f"找不到模型權重檔案 {model_path}")
        return

    autoencoder.eval()

    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.mode == "video":
        input_video_path = config.get("inference", {}).get("test_video_input", "")
        output_video_path = os.path.join(output_folder, "using_F_Decoder_video.mp4")

        if not os.path.exists(input_video_path):
            print(f"找不到輸入影片：{input_video_path}")
            return

        process_video(input_video_path, autoencoder, output_video_path, device)

    elif args.mode == "feature":
        input_npy_path = config.get("inference", {}).get("test_npy_input", "")
        output_video_path = os.path.join(output_folder, "using_F_Decoder_npy.mp4")

        sequence_length = config.get("inference", {}).get("sequence_length", 1)
        buffer = sequence_length - 1

        frame_size = (
            config.get("video", {}).get("new_width", 192),
            config.get("video", {}).get("new_height", 108),
        )
        fps = config.get("video", {}).get("target_fps", 10)

        if not os.path.exists(input_npy_path):
            print(f"找不到輸入特徵檔：{input_npy_path}")
            return

        process_npy(
            input_npy_path,
            autoencoder,
            output_video_path,
            device,
            frame_size=frame_size,
            fps=fps,
            buffer=buffer,
        )


if __name__ == "__main__":
    main()

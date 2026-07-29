import os
from typing import Optional, Tuple

import cv2
import torch


def downsample_video(
    input_path: str,
    output_path: str,
    target_fps: int = 10,
    target_resolution: Optional[Tuple[int, int]] = (192, 108),
) -> None:
    """
    將影片降取樣至目標 FPS (例如從 60fps 降至 10fps，\
        每 6 幀取 1 幀)，並可選地進行 Resize。

    Args:
        input_path (str): 輸入影片路徑。
        output_path (str): 輸出影片路徑。
        target_fps (int, optional): 目標 FPS。預設為 10。
        target_resolution (Tuple[int, int], optional): \
            若提供，則同時進行 Resize (width, height)。預設為 (192, 108)。
    """
    if not os.path.exists(input_path):
        print(f"找不到輸入影片 {input_path}")
        return

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法打開影片: {input_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # 計算需要跳過的幀數間隔 (如果 original_fps 為 0 則避免除以 0 錯誤)
    if original_fps == 0 or target_fps == 0:
        print(f"無效的 FPS 資訊 (Original: {original_fps}, Target: {target_fps})")
        cap.release()
        return

    frame_interval = max(1, int(original_fps / target_fps))

    # 若未提供 target_resolution，則使用原影片大小
    if target_resolution is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_resolution = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_resolution)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每 frame_interval 個 frame 取一張
        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, target_resolution)
            out.write(resized_frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"影片降取樣處理完成: {output_path}")


def video_to_tensor(video_path: str, resample_factor: float = 1.0) -> torch.Tensor:
    """
    將影片讀取並轉換為 PyTorch Tensor [N, H, W, C]，並正規化至 [0, 1]。

    Args:
        video_path (str): 影片檔案路徑。
        resample_factor (float): 重取樣因子 (原始程式碼預設 1.0)。

    Returns:
        torch.Tensor: 形狀為 [N, H, W, C] 的張量。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到影片: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法打開影片: {video_path}")

    frames = []
    resampled_count = 1
    current_frame = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if round(resampled_count * resample_factor) == current_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame_rgb).float() / 255.0
            frames.append(frame_tensor)
            resampled_count += 1

        current_frame += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"影片沒有包含任何有效的 Frame: {video_path}")

    return torch.stack(frames)


def capture_video_segment(
    input_path: str, output_path: str, start_frame: int, end_frame: int
) -> None:
    """
    擷取影片的特定幀數區間並儲存為新影片。

    Args:
        input_path (str): 輸入影片路徑。
        output_path (str): 輸出影片路徑。
        start_frame (int): 起始幀 (從 0 開始計算，包含)。
        end_frame (int): 結束幀 (從 0 開始計算，包含)。
    """
    if not os.path.exists(input_path):
        print(f"找不到輸入影片 {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法打開影片: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"總幀數 {total_frames}, FPS {fps}, 尺寸 {width}x{height}")

    if start_frame > end_frame:
        print(f"起始幀 ({start_frame}) 不能大於結束幀 ({end_frame})。")
        cap.release()
        return
    if start_frame >= total_frames:
        print(f"起始幀 ({start_frame}) 已超出影片總長度 ({total_frames})。")
        cap.release()
        return
    if end_frame >= total_frames:
        print(f"結束幀 ({end_frame}) 超出影片總長度 ({total_frames})，將擷取至片尾。")
        end_frame = total_frames - 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break

        if current_frame >= start_frame:
            out.write(frame)
            frames_written += 1

        current_frame += 1

    cap.release()
    out.release()
    print(f"總共擷取了 {frames_written} 幀，已儲存至: {output_path}")


def reverse_video(input_path: str, output_path: str) -> None:
    """
    將影片倒轉並儲存為新影片。

    Args:
        input_path (str): 輸入影片路徑。
        output_path (str): 輸出影片路徑。
    """
    if not os.path.exists(input_path):
        print(f"找不到輸入影片 {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法打開影片: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"總幀數 {total_frames}, FPS {fps}, 尺寸 {width}x{height}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in reversed(frames):
        out.write(frame)

    out.release()
    print(f"倒轉完成！已儲存至: {output_path}")


def uniform_downsample_video(
    input_path: str, output_path: str, target_frame_count: int
) -> None:
    """
    將單一影片均勻取樣 (Uniform Downsampling) 至目標幀數。

    Args:
        input_path (str): 輸入影片路徑。
        output_path (str): 輸出影片路徑。
        target_frame_count (int): 目標幀數。
    """
    import numpy as np

    if not os.path.exists(input_path):
        print(f"找不到輸入影片 {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"無法打開影片: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        print(f"影片總幀數為 0，無法處理: {input_path}")
        cap.release()
        return

    if target_frame_count > total_frames:
        print(
            f"目標幀數 ({target_frame_count}) 大於影片總長度\
                ({total_frames})，將自動重複部分影格。"
        )

    # 計算均勻取樣的幀索引
    indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    idx_pointer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 允許重複取樣 (若 target_frame_count > total_frames)
        while idx_pointer < len(indices) and current_frame == indices[idx_pointer]:
            out.write(frame)
            idx_pointer += 1

        current_frame += 1

        if idx_pointer >= len(indices):
            break

    cap.release()
    out.release()


def adjust_video_and_npy_length(
    input_mp4_path: str,
    input_npy_path: str,
    output_mp4_path: str,
    output_npy_path: str,
    target_frame: int,
):
    """
    調整 .mp4 影片與對應的 .npy 特徵檔長度。
    若目標幀數較小，執行均勻下採樣；若目標幀數較大，執行線性內插補幀。
    """
    import os

    import cv2
    import numpy as np

    if not os.path.exists(input_mp4_path):
        print(f"找不到影片檔案：{input_mp4_path}")
        return
    if not os.path.exists(input_npy_path):
        print(f"找不到 npy 檔案：{input_npy_path}")
        return

    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    try:
        npy_data = np.load(input_npy_path)
    except Exception as e:
        print(f"讀取 .npy 檔案時發生錯誤：{e}")
        return

    cap = cv2.VideoCapture(input_mp4_path)
    if not cap.isOpened():
        print("無法開啟影片檔案。")
        return

    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_frame_count != npy_data.shape[0]:
        print("影片的總影格數與 .npy 檔案的第一個維度不相符！")
        cap.release()
        return

    if target_frame == original_frame_count:
        print(f"目標影格數與原始影格數相同 ({target_frame})，直接複製檔案。")
        import shutil

        cap.release()
        shutil.copy2(input_mp4_path, output_mp4_path)
        shutil.copy2(input_npy_path, output_npy_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if target_frame < original_frame_count:
        # 執行均勻下採樣
        indices_to_keep = np.linspace(
            0, original_frame_count - 1, target_frame, dtype=int
        )

        # 處理 .npy
        new_npy_data = npy_data[indices_to_keep]
        np.save(output_npy_path, new_npy_data)

        # 處理 .mp4
        # 若輸入與輸出同名，使用暫存檔案避免邊讀邊寫時破壞原檔
        import shutil

        temp_mp4_path = output_mp4_path + ".tmp.mp4"
        out = cv2.VideoWriter(temp_mp4_path, fourcc, fps, (frame_width, frame_height))
        current_frame_index = 0
        idx_pointer = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            while (
                idx_pointer < len(indices_to_keep)
                and current_frame_index == indices_to_keep[idx_pointer]
            ):
                out.write(frame)
                idx_pointer += 1

            current_frame_index += 1
            if idx_pointer >= len(indices_to_keep):
                break

        cap.release()
        out.release()

        # 覆寫原檔案
        shutil.move(temp_mp4_path, output_mp4_path)

        print(f"下採樣至 {target_frame} 幀：{output_mp4_path}")

    else:
        original_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame)

        cap.release()

        original_frames_arr = np.array(original_frames, dtype=np.float64)

        target_indices_float = np.linspace(0, original_frame_count - 1, target_frame)
        idx1 = np.floor(target_indices_float).astype(int)
        idx2 = np.ceil(target_indices_float).astype(int)
        idx2[idx2 >= original_frame_count] = original_frame_count - 1

        weight2 = target_indices_float - idx1
        weight1 = 1.0 - weight2

        # 處理 .npy
        w1_npy = weight1[:, np.newaxis]
        w2_npy = weight2[:, np.newaxis]
        new_npy_data = npy_data[idx1] * w1_npy + npy_data[idx2] * w2_npy
        np.save(output_npy_path, new_npy_data)

        # 處理 .mp4
        w1_vid = weight1[:, np.newaxis, np.newaxis, np.newaxis]
        w2_vid = weight2[:, np.newaxis, np.newaxis, np.newaxis]
        new_frames_arr = (
            original_frames_arr[idx1] * w1_vid + original_frames_arr[idx2] * w2_vid
        )
        new_frames_arr = new_frames_arr.astype(np.uint8)

        out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (frame_width, frame_height))
        for new_frame in new_frames_arr:
            out.write(new_frame)
        out.release()
        print(f"已成功內插補幀至 {target_frame} 幀：{output_mp4_path}")

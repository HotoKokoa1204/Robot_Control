import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_navigation_system.lstm_model import LocationLSTM
from visual_navigation_system.models import AutoEncoder
from visual_navigation_system.predictor import load_segment_info, map_to_actual_idx


def main():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    if not os.path.exists(config_path):
        print(f"找不到設定檔: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config.get("lstm")
    if not cfg:
        print("設定檔中找不到 'lstm' 區塊！")
        return

    # ---------- 1. 載入設定與段落資訊 ----------
    best_info_path = cfg.get("best_info_path")
    input_info_path = cfg.get("test_info_path")

    best_segs = load_segment_info(best_info_path)
    in_segs = load_segment_info(input_info_path)

    if not best_segs or not in_segs:
        print("找不到輸入或最佳影片資訊檔。請確認路徑。")
        return

    # 計算對齊後的「實際」總幀數與誤差容忍值
    best_act_total = sum(seg["actual_length"] for seg in best_segs)
    error_percent = config.get("greedy_prediction", {}).get(
        "error_tolerance_percent", 0.05
    )
    threshold = error_percent * best_act_total
    print(
        f"最佳影片總幀數(去旋轉): {best_act_total},\
            允許誤差幀數(+-{error_percent * 100}%): {threshold:.2f}"
    )

    # ---------- 2. 地圖繪圖設定 ----------
    map_image_path = cfg.get("map_image_path")
    map_img = cv2.imread(map_image_path)
    if map_img is None:
        print(f"無法讀取地圖圖片: {map_image_path}")
        return
    h, w = map_img.shape[:2]

    pts = [(int(p[0]), int(p[1])) for p in cfg.get("map_points", [])]
    indices = cfg.get("static_path_indices", [])
    if not pts or not indices:
        print("未在 yaml 中設定地圖點位 (map_points) 或路線對應 (static_path_indices)")
        return

    static_path = [pts[i] for i in indices]
    direction_vectors = [
        (
            static_path[i + 1][0] - static_path[i][0],
            static_path[i + 1][1] - static_path[i][1],
        )
        for i in range(len(static_path) - 1)
    ]

    best_frame_recorder, best_cum_frame = [], []
    with open(best_info_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip("'\"[]") for p in line.split(",")]
            if len(parts) < 3:
                continue
            best_frame_recorder.append(int(parts[1]))
            best_cum_frame.append(int(parts[2]))

    def calculate_drawing_position(raw_db_idx):
        if raw_db_idx is None or raw_db_idx < 0:
            return static_path[0]
        try:
            video_num = next(i for i, c in enumerate(best_cum_frame) if raw_db_idx < c)
        except StopIteration:
            return static_path[-1]

        dx, dy = direction_vectors[video_num]
        x0, y0 = static_path[video_num]
        prev_cum = best_cum_frame[video_num - 1] if video_num > 0 else 0
        rate = (
            (raw_db_idx - prev_cum) / best_frame_recorder[video_num]
            if best_frame_recorder[video_num] > 0
            else 0
        )
        return x0 + rate * dx, y0 + rate * dy

    def get_xy_from_actual(act_idx, segs):
        target_seg = next(
            (
                s
                for s in segs
                if s["actual_start"] <= act_idx < s["actual_start"] + s["actual_length"]
            ),
            None,
        )
        if not target_seg:
            return static_path[0] if act_idx < 0 else static_path[-1]

        if target_seg["actual_length"] > 1:
            relative_progress = (act_idx - target_seg["actual_start"]) / (
                target_seg["actual_length"] - 1
            )
        else:
            relative_progress = 0.0

        raw_len = target_seg["raw_end"] - target_seg["raw_start"]
        interp_raw_idx = target_seg["raw_start"] + relative_progress * raw_len
        return calculate_drawing_position(interp_raw_idx)

    # ---------- 3. 初始化模型預測器 ----------
    device_str = cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    encoder = AutoEncoder().to(device)
    encoder_path = cfg.get("encoder_model_path")
    if os.path.exists(encoder_path):
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device, weights_only=False)
        )
    encoder.eval()

    lstm_model = LocationLSTM(
        input_size=512,
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        output_size=512,
    ).to(device)
    lstm_path = cfg.get("lstm_weights_path")
    if os.path.exists(lstm_path):
        lstm_model.load_state_dict(
            torch.load(lstm_path, map_location=device, weights_only=False)
        )
    lstm_model.eval()

    db_features_path = cfg.get("best_features_path")
    db_features = torch.tensor(
        np.load(db_features_path), dtype=torch.float32, device=device
    )

    def transform(f):
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192, 108)).transpose(2, 0, 1) / 255.0
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    # 動態投票限制
    enable_voting_mechanism = cfg.get("enable_voting_mechanism", False)
    voting_window_size = cfg.get("voting_window_size", 15)
    voting_majority = voting_window_size // 2 + 1
    distance_error_margin = cfg.get("distance_error_margin", 50)
    prediction_history = deque(maxlen=voting_window_size)

    sequence_len = cfg.get("sequence_length", 9)
    feature_sequence = deque(maxlen=sequence_len)

    # ---------- 4. 第一階段：LSTM 推理並記錄結果 ----------
    test_video_path = cfg.get("test_video_path")
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"無法打開測試影片: {test_video_path}")
        return

    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    results_data = []
    correct_predictions = 0
    predictions_made = 0
    final_raw_db_pred = None

    with torch.no_grad():
        for idx in range(total_in):
            ret, frame = cap.read()
            if not ret:
                break

            # 提取當前幀特徵並加入滑動窗口
            current_feat, _ = encoder(transform(frame))
            feature_sequence.append(current_feat)

            # 序列 0~8 幀的輸入預測的是第 9 幀；最後一幀沒有下一幀可供評估，因此不評估
            if len(feature_sequence) == sequence_len and idx + 1 < total_in:
                predictions_made += 1

                # 此次 LSTM 輸出的目標是輸入窗口之後的下一幀，而非窗口末幀
                predicted_frame_idx = idx + 1

                seq_tensor = torch.cat(list(feature_sequence), dim=0).unsqueeze(0)
                predicted_feat = lstm_model(seq_tensor)

                dists = torch.norm(db_features - predicted_feat, dim=1)
                raw_db_predicted = int(torch.argmin(dists).item())

                final_raw_db_pred = raw_db_predicted

                if enable_voting_mechanism:
                    if len(prediction_history) == voting_window_size:
                        distortion_votes = 0
                        for i, past_pred in enumerate(reversed(prediction_history)):
                            time_gap = i + 1
                            error = abs(raw_db_predicted - past_pred)
                            dynamic_threshold = time_gap + distance_error_margin
                            if error > dynamic_threshold:
                                distortion_votes += 1

                        if distortion_votes >= voting_majority:
                            final_raw_db_pred = prediction_history[-1]

                    prediction_history.append(final_raw_db_pred)

                # 準確率計算
                current_frame_x_idx = predicted_frame_idx
                act_in_ground_truth = map_to_actual_idx(current_frame_x_idx, in_segs)
                act_best_predicted = map_to_actual_idx(final_raw_db_pred, best_segs)

                in_seg = next(
                    (
                        s
                        for s in in_segs
                        if s["raw_start"] <= current_frame_x_idx <= s["raw_end"]
                    ),
                    None,
                )
                best_seg = (
                    next((s for s in best_segs if s["name"] == in_seg["name"]), None)
                    if in_seg
                    else None
                )

                if (
                    in_seg
                    and best_seg
                    and act_in_ground_truth is not None
                    and act_best_predicted is not None
                ):
                    in_s, in_l = in_seg["actual_start"], in_seg["actual_length"]
                    out_s, out_l = best_seg["actual_start"], best_seg["actual_length"]

                    act_inxscale_ground_truth = out_s
                    if in_l > 1:
                        rel = (act_in_ground_truth - in_s) / (in_l - 1)
                        act_inxscale_ground_truth = rel * (out_l - 1) + out_s

                    error = abs(act_inxscale_ground_truth - act_best_predicted)

                    if error <= threshold:
                        correct_predictions += 1

                    results_data.append(
                        {
                            "idx": predicted_frame_idx,
                            "raw_db": final_raw_db_pred,
                            "act_inxscale": act_inxscale_ground_truth,
                            "predicted_feature": predicted_feat.squeeze(0)
                            .cpu()
                            .numpy(),
                            "error": error,
                        }
                    )

            pred_str = (
                f"預測 {final_raw_db_pred}"
                if final_raw_db_pred is not None
                else "序列未滿"
            )
            print(f"frame {idx + 1}/{total_in} | {pred_str}", end="\r")

    cap.release()
    final_accuracy = (
        correct_predictions / predictions_made if predictions_made > 0 else 0
    )
    print(f"\n準確率: {correct_predictions}/{predictions_made} = {final_accuracy:.2%}")

    if not results_data:
        print("沒有產生任何預測結果！")
        return

    # ---------- 5. 第二階段：繪製地圖動畫 ----------
    out_video = cfg.get("output_video_path", "Map_LSTM.mp4")
    fourcc = cv2.Videowriter_fourcc(*"mp4v")
    writer = cv2.Videowriter(out_video, fourcc, fps, (w, h))

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2

    for res in results_data:
        idx = res["idx"]
        raw_db = res["raw_db"]
        act_inxscale = res["act_inxscale"]

        canvas = map_img.copy()

        x_pre, y_pre = calculate_drawing_position(raw_db)
        x_act, y_act = get_xy_from_actual(act_inxscale, best_segs)

        # 繪製容錯區間
        lo_act = max(0, act_inxscale - threshold)
        hi_act = min(best_act_total - 1, act_inxscale + threshold)

        tolerance_points = [
            get_xy_from_actual(p, best_segs)
            for p in np.linspace(lo_act, hi_act, num=20)
        ]
        pts_cv2 = np.array(
            [[round(p[0]), round(h - p[1])] for p in tolerance_points], dtype=np.int32
        )
        cv2.polylines(
            canvas,
            [pts_cv2],
            isClosed=False,
            color=(0, 255, 0),
            thickness=12,
            lineType=cv2.LINE_AA,
        )

        # 繪製預測點與實際點
        cv2.circle(
            canvas,
            (int(round(x_act)), int(round(h - y_act))),
            15,
            (255, 0, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            (int(round(x_pre)), int(round(h - y_pre))),
            10,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        # 圖例與文字
        legend_x, legend_y = 10, 10
        items = [
            ("Predicted", (0, 0, 255), "circle"),
            ("Actual", (255, 0, 0), "circle"),
            ("Tolerance window", (0, 255, 0), "line"),
        ]
        max_w = max(cv2.getTextSize(t, font, fs, th)[0][0] for t, _, _ in items)
        txt_h = cv2.getTextSize(items[0][0], font, fs, th)[0][1]
        pad = 10
        box_w, box_h = 35 + max_w + pad * 2, (txt_h + pad) * len(items) + pad
        cv2.rectangle(
            canvas,
            (legend_x - pad, legend_y - pad),
            (legend_x + box_w, legend_y + box_h),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (legend_x - pad, legend_y - pad),
            (legend_x + box_w, legend_y + box_h),
            (0, 0, 0),
            1,
        )

        y0 = legend_y + pad
        for text, color, shape in items:
            if shape == "circle":
                cv2.circle(canvas, (legend_x + 15, y0 + txt_h // 2), 8, color, -1)
            else:
                cv2.line(
                    canvas,
                    (legend_x + 5, y0 + txt_h // 2),
                    (legend_x + 25, y0 + txt_h // 2),
                    color,
                    6,
                    cv2.LINE_AA,
                )
            text_x = legend_x + 35
            cv2.putText(
                canvas,
                text,
                (text_x, y0 + txt_h - 2),
                font,
                fs,
                (0, 0, 0),
                th,
                cv2.LINE_AA,
            )
            y0 += txt_h + pad

        # 幀數
        frame_text = f"Frame: {idx + 1} / {total_in}"
        cv2.putText(
            canvas,
            frame_text,
            (1000, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # 準確率 (固定顯示在右下角，與原本 Notebook 完全相同)
        acc_text = f"Accuracy: {final_accuracy:.2%}"
        (aw, ah), bs = cv2.getTextSize(acc_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        x0, y0_acc = w - aw - 20, h - 20
        cv2.rectangle(
            canvas,
            (x0 - 5, y0_acc - ah - 5),
            (x0 + aw + 5, y0_acc + bs),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            acc_text,
            (x0, y0_acc),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        writer.write(canvas)
        print(f"正在寫入地圖影片... 預測結果幀數 {idx + 1}", end="\r")

    writer.release()

    # 儲存特徵向量
    out_features = cfg.get("output_features_path", "lstm_predicted_features.npy")
    predicted_features = np.array([res["predicted_feature"] for res in results_data])
    np.save(out_features, predicted_features)

    print(f"\n預測特徵向量儲存至: {out_features}")
    print(f"預測結果地圖影片儲存至: {out_video}")


if __name__ == "__main__":
    main()

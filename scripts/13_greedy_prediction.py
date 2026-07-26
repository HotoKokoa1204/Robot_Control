import os
import sys

import cv2
import numpy as np
import yaml

# 確保可正確載入 core 目錄下的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_navigation_system.predictor import (
    GreedyPredictor,
    load_segment_info,
    map_to_actual_idx,
)


def main():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    if not os.path.exists(config_path):
        print(f"找不到設定檔: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config.get("greedy_prediction")
    if not cfg:
        print("設定檔中找不到 'greedy_prediction' 區塊！")
        return

    # ---------- 1. 載入設定與段落資訊 ----------
    best_info_path = cfg.get("best_info_path")
    input_info_path = cfg.get("input_info_path")

    best_segs = load_segment_info(best_info_path)
    in_segs = load_segment_info(input_info_path)

    if not best_segs or not in_segs:
        print("找不到輸入或最佳影片資訊檔。請確認路徑。")
        return

    # 計算對齊後的「實際」總幀數與誤差容忍值
    best_act_total = sum(seg["actual_length"] for seg in best_segs)
    error_percent = cfg.get("error_tolerance_percent", 0.05)
    threshold = error_percent * best_act_total
    print(
        f"最佳影片總幀數(去旋轉): {best_act_total}, \
            允許誤差幀數(+-{error_percent * 100}%): {threshold:.2f}"
    )

    # ---------- 2. 地圖繪圖設定 ----------
    map_image_path = cfg.get("map_image_path")
    map_img = cv2.imread(map_image_path)
    if map_img is None:
        print(f"無法讀取地圖圖片: {map_image_path}")
        return
    h, w = map_img.shape[:2]

    # 確保座標轉為整數，避免 yaml 解析為字串
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

    # 讀取最佳影片原幀累積資訊，用於畫圖映射
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

    # ---------- 3. 初始化預測器與輸出 ----------
    predictor = GreedyPredictor(
        model_path=cfg.get("model_path"), db_features_path=cfg.get("db_features_path")
    )

    test_video_path = cfg.get("test_video_path")
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"無法打開測試影片: {test_video_path}")
        return

    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_video = cfg.get("output_video_path", "Map.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    results_data = []
    correct = 0

    # ---------- 4. 第一階段：推理並記錄結果 ----------
    for idx in range(total_in):
        ret, frame = cap.read()
        if not ret:
            break

        # 預測最相似的資料庫索引
        raw_db, best_match_feature, min_dist = predictor.predict(frame)

        # 對應到實際空間 (去除旋轉後)
        act_in = map_to_actual_idx(idx, in_segs)
        act_best = map_to_actual_idx(raw_db, best_segs)
        if act_in is None or act_best is None:
            continue

        in_seg = next(
            (s for s in in_segs if s["raw_start"] <= idx <= s["raw_end"]), None
        )
        best_seg = (
            next((s for s in best_segs if s["name"] == in_seg["name"]), None)
            if in_seg
            else None
        )
        if not in_seg or not best_seg:
            continue

        in_s, in_l = in_seg["actual_start"], in_seg["actual_length"]
        out_s, out_l = best_seg["actual_start"], best_seg["actual_length"]

        act_inxscale = out_s
        if in_l > 1:
            rel = (act_in - in_s) / (in_l - 1)
            act_inxscale = rel * (out_l - 1) + out_s

        # 計算誤差是否在全域閾值內
        error = abs(act_inxscale - act_best)
        is_correct = error <= threshold
        if is_correct:
            correct += 1

        results_data.append(
            {
                "idx": idx,
                "raw_db": raw_db,
                "act_inxscale": act_inxscale,
                "predicted_feature": best_match_feature.cpu().numpy(),
                "min_dist": min_dist,
                "error": error,
            }
        )

        print(
            f"處理中... 幀 {idx + 1}/{total_in} | 預測 {raw_db:03d}\
                (dist: {min_dist:.2f}) | error: {error:.2f}",
            end="\r",
        )

    cap.release()
    final_accuracy = correct / total_in if total_in > 0 else 0
    print(f"\n準確率: {correct}/{total_in} = {final_accuracy:.2%}")

    # ---------- 5. 第二階段：繪製地圖動畫 ----------
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
        print(f"正在寫入地圖影片... 幀 {idx + 1}/{total_in}", end="\r")

    writer.release()

    # 儲存特徵向量
    out_features = cfg.get("output_features_path", "greedy_predicted_features.npy")
    predicted_features = np.array([res["predicted_feature"] for res in results_data])
    np.save(out_features, predicted_features)

    print(f"預測特徵向量儲存至: {out_features}")
    print(f"預測結果地圖影片儲存至: {out_video}")


if __name__ == "__main__":
    main()

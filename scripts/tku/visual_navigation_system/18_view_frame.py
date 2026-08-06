import sys

import cv2
from PyQt5.QtCore import QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

STYLESHEET = """
QWidget {
    background-color: #2E2F30;
    color: #E0E0E0;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QGroupBox {
    background-color: #3C3D3E;
    border: 1px solid #505050;
    border-radius: 5px;
    margin-top: 1ex;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    background-color: #3C3D3E;
}
QLabel {
    font-size: 14px;
}
QLabel#VideoLabel {
    border: 2px solid #505050;
    background-color: black;
}
QPushButton {
    background-color: #505152;
    border: 1px solid #606162;
    padding: 8px;
    border-radius: 4px;
    min-width: 40px;
}
QPushButton:hover {
    background-color: #606162;
    border: 1px solid #707172;
}
QPushButton:pressed {
    background-color: #202122;
}
QPushButton:disabled {
    background-color: #404142;
    border: 1px solid #505152;
}
QSlider::groove:horizontal {
    border: 1px solid #505152;
    background: #404142;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #0078D7;
    border: 1px solid #0078D7;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
QListWidget {
    background-color: #3C3D3E;
    border: 1px solid #505050;
    font-size: 14px;
}
QListWidget::item:hover {
    background-color: #505152;
}
QListWidget::item:selected {
    background-color: #0078D7;
}
QScrollArea {
    border: 1px solid #505050;
    background-color: #3C3D3E;
}
"""


class TrimSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_start_frame = None
        self.trim_end_frame = None

    def set_trim_range(self, start, end):
        self.trim_start_frame = start
        self.trim_end_frame = end
        self.update()

    def paint_event(self, event):
        super().paint_event(event)
        if (
            self.trim_start_frame is None
            or self.trim_end_frame is None
            or self.maximum() == 0
        ):
            return
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        start_pos = (
            self.trim_start_frame / self.maximum()
        ) * groove_rect.width() + groove_rect.x()
        end_pos = (
            self.trim_end_frame / self.maximum()
        ) * groove_rect.width() + groove_rect.x()
        highlight_rect = QRect(
            int(start_pos),
            groove_rect.y(),
            int(end_pos - start_pos),
            groove_rect.height(),
        )
        painter.fillRect(highlight_rect, QColor(64, 224, 208, 128))


class ThumbnailView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedHeight(100)
        self.scroll_widget = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(4)
        self.setWidget(self.scroll_widget)

    def clear_thumbnails(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def populate_thumbnails(self, video_path, num_thumbnails=25):
        self.clear_thumbnails()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return
        interval = total_frames // num_thumbnails if num_thumbnails > 0 else 0

        for i in range(num_thumbnails):
            frame_index = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                thumbnail = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                rgb_image = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image)
                thumb_label = QLabel()
                thumb_label.setPixmap(pixmap)
                thumb_label.setFixedSize(160, 90)
                thumb_label.setStyleSheet("border: 1px solid #606162;")
                self.scroll_layout.addWidget(thumb_label)
            QApplication.processEvents()

        self.scroll_layout.addStretch()
        cap.release()


class VideoPlayerWidget(QWidget):
    def __init__(self, is_main_player=False):
        super().__init__()
        self.video_capture = None
        self.original_video_path = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.is_main_player = is_main_player
        self.recorded_frames_list = None
        self.is_playing = False
        self.playback_timer = QTimer(self)
        self.playback_fps = 10
        self.trim_start_frame = None
        self.trim_end_frame = None

        main_layout = QVBoxLayout(self)
        self.video_label = QLabel("請點擊下方按鈕開啟影片檔案")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label = QLabel("影格: N/A / N/A")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_slider = (
            TrimSlider(Qt.Horizontal) if self.is_main_player else QSlider(Qt.Horizontal)
        )

        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.frame_info_label)
        main_layout.addWidget(self.frame_slider)
        if self.is_main_player:
            self.thumbnail_view = ThumbnailView()
            main_layout.addWidget(self.thumbnail_view)

        controls_layout = QHBoxLayout()
        nav_controls_layout = QHBoxLayout()
        trim_controls_layout = QHBoxLayout()
        style = self.style()
        self.open_button = QPushButton()
        self.open_button.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        self.open_button.setToolTip("開啟影片 (Ctrl+O)")
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.setToolTip("播放/暫停 (Space)")
        self.prev_frame_button = QPushButton()
        self.prev_frame_button.setIcon(style.standardIcon(QStyle.SP_MediaSeekBackward))
        self.prev_frame_button.setToolTip("上一幀 (左方向鍵)")
        self.next_frame_button = QPushButton()
        self.next_frame_button.setIcon(style.standardIcon(QStyle.SP_MediaSeekForward))
        self.next_frame_button.setToolTip("下一幀 (右方向鍵)")
        nav_controls_layout.addWidget(self.open_button)
        nav_controls_layout.addWidget(self.play_pause_button)
        nav_controls_layout.addWidget(self.prev_frame_button)
        nav_controls_layout.addWidget(self.next_frame_button)

        if self.is_main_player:
            self.record_frame_button = QPushButton()
            self.record_frame_button.setIcon(
                style.standardIcon(QStyle.SP_DialogSaveButton)
            )
            self.record_frame_button.setToolTip("紀錄此影格 (R)")
            self.set_trim_start_button = QPushButton("[")
            self.set_trim_start_button.setToolTip("設定剪輯起點")
            self.set_trim_end_button = QPushButton("]")
            self.set_trim_end_button.setToolTip("設定剪輯終點")
            self.export_button = QPushButton()
            self.export_button.setIcon(style.standardIcon(QStyle.SP_DriveFDIcon))
            self.export_button.setToolTip("匯出剪輯後的影片")
            trim_controls_layout.addWidget(self.record_frame_button)
            trim_controls_layout.addWidget(self.set_trim_start_button)
            trim_controls_layout.addWidget(self.set_trim_end_button)
            trim_controls_layout.addWidget(self.export_button)

        controls_layout.addLayout(nav_controls_layout)
        controls_layout.addStretch()
        if self.is_main_player:
            controls_layout.addLayout(trim_controls_layout)
        main_layout.addLayout(controls_layout)

        self.playback_timer.timeout.connect(self.advance_frame_for_playback)
        self.frame_slider.valueChanged.connect(self.set_frame_by_slider)
        self.open_button.clicked.connect(self.open_video_file)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.prev_frame_button.clicked.connect(self.handle_manual_prev_frame)
        self.next_frame_button.clicked.connect(self.handle_manual_next_frame)

        if self.is_main_player:
            self.record_frame_button.clicked.connect(self.record_frame)
            self.set_trim_start_button.clicked.connect(self.set_trim_start)
            self.set_trim_end_button.clicked.connect(self.set_trim_end)
            self.export_button.clicked.connect(self.export_trimmed_video)
        self.update_button_states()

    def open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選取影片檔案", "", "影片檔案 (*.mp4 *.avi)"
        )
        if file_path:
            self.original_video_path = file_path
            self.video_capture = cv2.VideoCapture(file_path)
            if not self.video_capture.isOpened():
                self.video_label.setText("無法開啟影片")
                return
            self.stop_playback()
            self.trim_start_frame = None
            self.trim_end_frame = None
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_number = 0
            self.frame_slider.setRange(
                0, self.total_frames - 1 if self.total_frames > 0 else 0
            )
            if self.is_main_player:
                self.frame_slider.set_trim_range(None, None)
                self.thumbnail_view.populate_thumbnails(self.original_video_path)
            self.show_frame()
            self.update_button_states()

    def toggle_playback(self):
        if not self.video_capture:
            return
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if not self.video_capture:
            return
        if self.trim_start_frame is not None and self.trim_end_frame is not None:
            if (
                self.current_frame_number < self.trim_start_frame
                or self.current_frame_number >= self.trim_end_frame
            ):
                self.current_frame_number = self.trim_start_frame
                self.show_frame()
        elif self.current_frame_number >= self.total_frames - 1:
            self.current_frame_number = 0
            self.show_frame()
        self.is_playing = True
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.playback_timer.start(1000 // self.playback_fps)

    def stop_playback(self):
        self.is_playing = False
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playback_timer.stop()

    def advance_frame_for_playback(self):
        end_frame = (
            self.trim_end_frame
            if self.trim_end_frame is not None
            else self.total_frames - 1
        )
        if self.current_frame_number >= end_frame:
            self.stop_playback()
            if (
                self.trim_end_frame is not None
                and self.current_frame_number > self.trim_end_frame
            ):
                self.current_frame_number = self.trim_end_frame
                self.show_frame()
            return
        self.show_next_frame()

    def handle_manual_next_frame(self):
        if self.is_playing:
            self.stop_playback()
        self.show_next_frame()

    def handle_manual_prev_frame(self):
        if self.is_playing:
            self.stop_playback()
        self.show_previous_frame()

    def show_next_frame(self):
        if self.current_frame_number < self.total_frames - 1:
            self.current_frame_number += 1
            self.show_frame()

    def show_previous_frame(self):
        if self.current_frame_number > 0:
            self.current_frame_number -= 1
            self.show_frame()

    def set_frame_by_slider(self, frame_number):
        if self.video_capture:
            if self.is_playing:
                self.stop_playback()
            self.current_frame_number = frame_number
            self.show_frame()

    def show_frame(self):
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.video_capture.read()
            if ret:
                # --- 修改處：固定處理的解析度 ---
                frame = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                # 顯示時再放大以填滿標籤，方便觀看
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_label.setPixmap(pixmap)
            self.update_ui_elements()

    def update_ui_elements(self):
        display_frame = self.current_frame_number
        display_total = self.total_frames - 1 if self.total_frames > 0 else 0
        self.frame_info_label.setText(f"影格: {display_frame} / {display_total}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_number)
        self.frame_slider.blockSignals(False)

    def key_press_event(self, event):
        if not self.video_capture:
            super().key_press_event(event)
            return
        key = event.key()
        if key == Qt.Key_Right:
            self.handle_manual_next_frame()
        elif key == Qt.Key_Left:
            self.handle_manual_prev_frame()
        elif key == Qt.Key_Space:
            self.toggle_playback()
        elif key == Qt.Key_R and self.is_main_player:
            self.record_frame()
        else:
            super().key_press_event(event)

    def set_trim_start(self):
        if not self.video_capture:
            return
        self.trim_start_frame = self.current_frame_number
        if (
            self.trim_end_frame is not None
            and self.trim_start_frame > self.trim_end_frame
        ):
            self.trim_end_frame, self.trim_start_frame = (
                self.trim_start_frame,
                self.trim_end_frame,
            )
        self.frame_slider.set_trim_range(self.trim_start_frame, self.trim_end_frame)
        self.update_button_states()

    def set_trim_end(self):
        if not self.video_capture:
            return
        self.trim_end_frame = self.current_frame_number
        if (
            self.trim_start_frame is not None
            and self.trim_start_frame > self.trim_end_frame
        ):
            self.trim_start_frame, self.trim_end_frame = (
                self.trim_end_frame,
                self.trim_start_frame,
            )
        self.frame_slider.set_trim_range(self.trim_start_frame, self.trim_end_frame)
        self.update_button_states()

    def export_trimmed_video(self):
        if self.trim_start_frame is None or self.trim_end_frame is None:
            QMessageBox.warning(self, "錯誤", "請先設定剪輯的起點和終點。")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "儲存剪輯影片", "", "MP4 檔案 (*.mp4)"
        )
        if not save_path:
            return

        # --- 修改處：固定匯出的規格 ---
        export_fps = 10.0
        export_width = 192
        export_height = 108

        cap = cv2.VideoCapture(self.original_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            save_path, fourcc, export_fps, (export_width, export_height)
        )

        total_trim_frames = self.trim_end_frame - self.trim_start_frame + 1
        progress = QProgressDialog(
            "正在匯出影片...", "取消", 0, total_trim_frames, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("匯出中")
        progress.show()

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start_frame)
        for i in range(total_trim_frames):
            if progress.wasCanceled():
                break
            ret, frame = cap.read()
            if not ret:
                break

            # --- 修改處：在寫入前縮放每一幀 ---
            resized_frame = cv2.resize(
                frame, (export_width, export_height), interpolation=cv2.INTER_AREA
            )
            out.write(resized_frame)

            progress.setValue(i + 1)
            QApplication.processEvents()

        cap.release()
        out.release()
        if progress.wasCanceled():
            QMessageBox.information(self, "完成", "匯出已取消。")
        else:
            progress.setValue(total_trim_frames)
            QMessageBox.information(self, "完成", f"影片已成功匯出至:\n{save_path}")

    def record_frame(self):
        if (
            self.is_main_player
            and self.recorded_frames_list is not None
            and self.video_capture
        ):
            frame_to_record = self.current_frame_number
            item_text = f"影格編號: {frame_to_record}"
            existing_items = [
                self.recorded_frames_list.item(i).text()
                for i in range(self.recorded_frames_list.count())
            ]
            if item_text not in existing_items:
                self.recorded_frames_list.addItem(QListWidgetItem(item_text))
                self.recorded_frames_list.scrollToBottom()

    def update_button_states(self):
        is_video_loaded = (
            self.video_capture is not None and self.video_capture.isOpened()
        )
        self.play_pause_button.setEnabled(is_video_loaded)
        self.prev_frame_button.setEnabled(is_video_loaded)
        self.next_frame_button.setEnabled(is_video_loaded)
        self.frame_slider.setEnabled(is_video_loaded)
        if self.is_main_player:
            self.record_frame_button.setEnabled(is_video_loaded)
            self.set_trim_start_button.setEnabled(is_video_loaded)
            self.set_trim_end_button.setEnabled(is_video_loaded)
            can_export = (
                is_video_loaded
                and self.trim_start_frame is not None
                and self.trim_end_frame is not None
            )
            self.export_button.setEnabled(can_export)

    def set_recorded_frames_list(self, list_widget):
        self.recorded_frames_list = list_widget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多功能影片處理工具")
        self.setGeometry(100, 100, 1800, 850)
        self.setStyleSheet(STYLESHEET)
        main_player_group = QGroupBox("主影片 (含剪輯與縮圖)")
        self.main_video_player = VideoPlayerWidget(is_main_player=True)
        main_player_layout = QVBoxLayout()
        main_player_layout.addWidget(self.main_video_player)
        main_player_group.setLayout(main_player_layout)
        comp_player_group = QGroupBox("比較影片")
        self.comparison_video_player = VideoPlayerWidget()
        comp_player_layout = QVBoxLayout()
        comp_player_layout.addWidget(self.comparison_video_player)
        comp_player_group.setLayout(comp_player_layout)
        record_list_group = QGroupBox("紀錄的影格")
        self.recorded_frames_list = QListWidget()
        record_list_layout = QVBoxLayout()
        record_list_layout.addWidget(self.recorded_frames_list)
        record_list_group.setLayout(record_list_layout)
        self.main_video_player.set_recorded_frames_list(self.recorded_frames_list)
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(main_player_group, 3)
        main_layout.addWidget(record_list_group, 1)
        main_layout.addWidget(comp_player_group, 3)
        self.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

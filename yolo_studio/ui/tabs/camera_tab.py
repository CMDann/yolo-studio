"""Camera tab for real-time webcam inference in YOLO Studio."""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.models.database import CameraSession, TrainingRun, get_session, utc_now
from sqlalchemy.orm import joinedload

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class SavedRunInfo:
    """Saved training-run payload used by model selector."""

    id: int
    name: str
    architecture: str
    weights_path: str
    class_names: list[str]


@dataclass(slots=True)
class DetectionRecord:
    """Single detection record emitted by the camera worker."""

    frame_number: int
    timestamp_ms: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class CameraStreamWorker(QThread):
    """QThread worker that streams webcam frames and runs inference."""

    frame_ready = pyqtSignal(QImage)
    frame_meta = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        device_index: int,
        weights_path: str,
        class_names: list[str],
        class_colors: list[tuple[int, int, int]],
        conf: float,
        iou: float,
        output_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._device_index = device_index
        self._weights_path = weights_path
        self._class_names = class_names
        self._class_colors = class_colors or [(0, 212, 170)]
        self._conf = conf
        self._iou = iou
        self._output_dir = output_dir

        self._running = False
        self._record_enabled = False
        self._snapshot_requested = 0
        self._lock = threading.Lock()

    def set_recording(self, enabled: bool) -> None:
        """Enable/disable saving annotated frames."""

        with self._lock:
            self._record_enabled = enabled

    def request_snapshot(self) -> None:
        """Request a snapshot save on the next frame."""

        with self._lock:
            self._snapshot_requested += 1

    def stop(self) -> None:
        """Stop the streaming loop."""

        self._running = False

    def run(self) -> None:
        """Stream frames, run inference, and emit annotated images."""

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            self.error.emit(f"Ultralytics is unavailable: {exc}")
            return

        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            self.error.emit(f"Unable to open camera device {self._device_index}.")
            return

        try:
            model = YOLO(self._weights_path)
        except Exception as exc:
            capture.release()
            self.error.emit(f"Unable to load model: {exc}")
            return

        self._running = True
        frame_number = 0
        target_frame_time = 1.0 / 25.0

        while self._running:
            loop_start = time.time()
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_number += 1
            timestamp_ms = int(time.time() * 1000)
            detections: list[DetectionRecord] = []

            try:
                results = model.predict(source=frame, conf=self._conf, iou=self._iou, verbose=False)
                if results:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy
                        confs = boxes.conf
                        clss = boxes.cls

                        if hasattr(xyxy, "cpu"):
                            xyxy = xyxy.cpu().numpy()
                        if hasattr(confs, "cpu"):
                            confs = confs.cpu().numpy()
                        if hasattr(clss, "cpu"):
                            clss = clss.cpu().numpy()

                        for coords, conf, cls_idx in zip(xyxy, confs, clss):
                            x1, y1, x2, y2 = [int(v) for v in coords]
                            class_index = int(cls_idx)
                            class_name = (
                                self._class_names[class_index]
                                if 0 <= class_index < len(self._class_names)
                                else str(class_index)
                            )
                            color = self._class_colors[class_index % len(self._class_colors)]

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{class_name} {float(conf):.2f}"
                            (text_w, text_h), baseline = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                1,
                            )
                            text_x = max(0, x1)
                            text_y = max(0, y1 - text_h - baseline - 4)
                            cv2.rectangle(
                                frame,
                                (text_x, text_y),
                                (text_x + text_w + 6, text_y + text_h + baseline + 4),
                                color,
                                -1,
                            )
                            cv2.putText(
                                frame,
                                label,
                                (text_x + 3, text_y + text_h + 1),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )

                            detections.append(
                                DetectionRecord(
                                    frame_number=frame_number,
                                    timestamp_ms=timestamp_ms,
                                    class_name=class_name,
                                    confidence=float(conf),
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                )
                            )
            except Exception as exc:
                LOGGER.exception("Inference failed.")
                self.status.emit(f"Inference error: {exc}")

            save_paths: list[Path] = []
            snapshot_count = 0
            record_enabled = False

            with self._lock:
                record_enabled = self._record_enabled
                if self._snapshot_requested > 0:
                    snapshot_count = self._snapshot_requested
                    self._snapshot_requested = 0

            if record_enabled or snapshot_count > 0:
                frames_dir = self._output_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)

                if record_enabled:
                    frame_path = frames_dir / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    save_paths.append(frame_path)

                for idx in range(snapshot_count):
                    snap_path = frames_dir / f"snapshot_{frame_number:06d}_{timestamp_ms}_{idx}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    save_paths.append(snap_path)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = rgb.shape
            bytes_per_line = width * 3
            qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()

            self.frame_ready.emit(qimage)
            self.frame_meta.emit(
                {
                    "frame_number": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "detections": detections,
                    "saved_files": [str(path) for path in save_paths],
                }
            )

            elapsed = time.time() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        capture.release()


class CameraTab(QWidget):
    """Tab that manages real-time camera inference sessions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_project_id: int | None = None
        self._project_root = PROJECT_ROOT
        self._output_root = self._project_root / "output" / "camera_sessions"
        self._saved_runs: dict[int, SavedRunInfo] = {}
        self._device_indices: list[int] = []

        self._stream_worker: CameraStreamWorker | None = None
        self._active_session_id: int | None = None
        self._active_output_dir: Path | None = None

        self._detection_log: list[DetectionRecord] = []
        self._class_counts: dict[str, int] = {}
        self._confidence_sum = 0.0
        self._frame_count = 0
        self._detection_count = 0
        self._saved_files: list[str] = []
        self._csv_saved = False

        self._device_combo: QComboBox
        self._model_combo: QComboBox
        self._conf_slider: QSlider
        self._iou_slider: QSlider
        self._conf_value_label: QLabel
        self._iou_value_label: QLabel
        self._preview_label: QLabel
        self._start_button: QPushButton
        self._record_toggle: QCheckBox
        self._save_csv_button: QPushButton
        self._snapshot_button: QPushButton
        self._frame_count_label: QLabel
        self._detection_count_label: QLabel

        self._build_ui()
        self.refresh_devices()
        self.refresh_saved_models()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        self._project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._output_root = self._project_root / "output" / "camera_sessions"
        self.refresh_saved_models()

    def _build_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        controls = QWidget(self)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        setup_group = QGroupBox("Camera Setup", controls)
        setup_layout = QVBoxLayout()
        setup_layout.setContentsMargins(8, 8, 8, 8)
        setup_layout.setSpacing(8)

        device_row = QWidget(setup_group)
        device_layout = QHBoxLayout()
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(8)

        self._device_combo = QComboBox(device_row)
        refresh_button = QPushButton("Refresh", device_row)
        refresh_button.setProperty("secondary", True)
        refresh_button.clicked.connect(self.refresh_devices)

        device_layout.addWidget(self._device_combo, stretch=1)
        device_layout.addWidget(refresh_button)
        device_row.setLayout(device_layout)

        self._model_combo = QComboBox(setup_group)

        form = QFormLayout()
        form.addRow("Camera Device", device_row)
        form.addRow("Model", self._model_combo)

        setup_layout.addLayout(form)
        setup_group.setLayout(setup_layout)

        threshold_group = QGroupBox("Thresholds", controls)
        threshold_layout = QVBoxLayout()
        threshold_layout.setContentsMargins(8, 8, 8, 8)
        threshold_layout.setSpacing(8)

        conf_row = QWidget(threshold_group)
        conf_layout = QHBoxLayout()
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(8)

        self._conf_slider = QSlider(Qt.Orientation.Horizontal, conf_row)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(25)
        self._conf_slider.valueChanged.connect(self._update_threshold_labels)

        self._conf_value_label = QLabel("0.25", conf_row)
        self._conf_value_label.setObjectName("metricValue")

        conf_layout.addWidget(self._conf_slider, stretch=1)
        conf_layout.addWidget(self._conf_value_label)
        conf_row.setLayout(conf_layout)

        iou_row = QWidget(threshold_group)
        iou_layout = QHBoxLayout()
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.setSpacing(8)

        self._iou_slider = QSlider(Qt.Orientation.Horizontal, iou_row)
        self._iou_slider.setRange(1, 99)
        self._iou_slider.setValue(45)
        self._iou_slider.valueChanged.connect(self._update_threshold_labels)

        self._iou_value_label = QLabel("0.45", iou_row)
        self._iou_value_label.setObjectName("metricValue")

        iou_layout.addWidget(self._iou_slider, stretch=1)
        iou_layout.addWidget(self._iou_value_label)
        iou_row.setLayout(iou_layout)

        threshold_layout.addWidget(QLabel("Confidence", threshold_group))
        threshold_layout.addWidget(conf_row)
        threshold_layout.addWidget(QLabel("IoU", threshold_group))
        threshold_layout.addWidget(iou_row)
        threshold_group.setLayout(threshold_layout)

        control_group = QGroupBox("Session Controls", controls)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(8)

        self._start_button = QPushButton("Start Stream", control_group)
        self._start_button.clicked.connect(self._toggle_stream)

        self._record_toggle = QCheckBox("Record", control_group)
        self._record_toggle.toggled.connect(self._toggle_recording)

        buttons_row = QWidget(control_group)
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)

        self._snapshot_button = QPushButton("Snapshot", buttons_row)
        self._snapshot_button.setProperty("secondary", True)
        self._snapshot_button.clicked.connect(self._snapshot_frame)

        self._save_csv_button = QPushButton("Save CSV", buttons_row)
        self._save_csv_button.setProperty("secondary", True)
        self._save_csv_button.clicked.connect(self._save_csv)

        buttons_layout.addWidget(self._snapshot_button)
        buttons_layout.addWidget(self._save_csv_button)
        buttons_row.setLayout(buttons_layout)

        stats_frame = QFrame(control_group)
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(4)

        self._frame_count_label = QLabel("Frames: 0", stats_frame)
        self._frame_count_label.setProperty("role", "subtle")
        self._detection_count_label = QLabel("Detections: 0", stats_frame)
        self._detection_count_label.setProperty("role", "subtle")

        stats_layout.addWidget(self._frame_count_label)
        stats_layout.addWidget(self._detection_count_label)
        stats_frame.setLayout(stats_layout)

        control_layout.addWidget(self._start_button)
        control_layout.addWidget(self._record_toggle)
        control_layout.addWidget(buttons_row)
        control_layout.addWidget(stats_frame)
        control_layout.addStretch(1)
        control_group.setLayout(control_layout)

        controls_layout.addWidget(setup_group)
        controls_layout.addWidget(threshold_group)
        controls_layout.addWidget(control_group)
        controls_layout.addStretch(1)
        controls.setLayout(controls_layout)

        preview = QGroupBox("Live Preview", self)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(8)

        self._preview_label = QLabel("Start a stream to see camera output.", preview)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(360)
        self._preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self._preview_label.setScaledContents(True)

        preview_layout.addWidget(self._preview_label)
        preview.setLayout(preview_layout)

        layout.addWidget(controls, stretch=1)
        layout.addWidget(preview, stretch=3)
        self.setLayout(layout)

    def refresh_devices(self) -> None:
        """Enumerate available camera devices via OpenCV."""

        self._device_combo.clear()
        self._device_indices.clear()

        indices = []
        consecutive_misses = 0
        max_probes = 12
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY

        for idx in range(max_probes):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                indices.append(idx)
                consecutive_misses = 0
            else:
                consecutive_misses += 1
            cap.release()

            if indices and consecutive_misses >= 3:
                break

        if not indices:
            self._device_combo.addItem("No devices detected", None)
            return

        for idx in indices:
            self._device_combo.addItem(f"Camera {idx}", idx)
        self._device_indices = indices

    def refresh_saved_models(self) -> None:
        """Reload saved model list from TrainingRun table."""

        self._saved_runs.clear()
        self._model_combo.clear()

        session = get_session()
        try:
            query = (
                session.query(TrainingRun)
                .options(joinedload(TrainingRun.dataset))
                .filter(TrainingRun.is_saved.is_(True))
            )
            if self._active_project_id is not None:
                query = query.filter(TrainingRun.project_id == self._active_project_id)
            runs = query.order_by(TrainingRun.completed_at.desc(), TrainingRun.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load saved models.")
            runs = []
        finally:
            session.close()

        if not runs:
            self._model_combo.addItem("No saved models", None)
            return

        for run in runs:
            class_names = run.dataset.class_names if run.dataset is not None else []
            info = SavedRunInfo(
                id=run.id,
                name=run.name,
                architecture=run.model_architecture,
                weights_path=str(run.weights_path or ""),
                class_names=class_names,
            )
            self._saved_runs[run.id] = info
            label = f"{run.name} ({run.model_architecture})"
            self._model_combo.addItem(label, run.id)

    def _update_threshold_labels(self) -> None:
        self._conf_value_label.setText(f"{self._conf_slider.value() / 100.0:.2f}")
        self._iou_value_label.setText(f"{self._iou_slider.value() / 100.0:.2f}")

    def _toggle_recording(self, enabled: bool) -> None:
        if self._stream_worker is not None:
            self._stream_worker.set_recording(enabled)

    def _toggle_stream(self) -> None:
        if self._stream_worker is not None and self._stream_worker.isRunning():
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self) -> None:
        if self._stream_worker is not None and self._stream_worker.isRunning():
            QMessageBox.information(self, "Camera", "A camera stream is already running.")
            return

        device_index = self._device_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "Camera", "Select a camera device.")
            return

        run_id = self._model_combo.currentData()
        if run_id is None:
            QMessageBox.warning(self, "Camera", "Select a saved model.")
            return

        run_info = self._saved_runs.get(int(run_id))
        if run_info is None:
            QMessageBox.warning(self, "Camera", "Selected model is no longer available.")
            return

        weights_path = Path(run_info.weights_path)
        if not weights_path.exists():
            QMessageBox.warning(self, "Camera", "Selected weights file does not exist.")
            return

        session = get_session()
        try:
            record = CameraSession(
                model_id=int(run_id),
                device_index=int(device_index),
                started_at=utc_now(),
                output_dir="",
                frame_count=0,
                detection_count=0,
                project_id=self._active_project_id,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session_id = record.id
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Camera", f"Could not create session: {exc}")
            return
        finally:
            session.close()

        output_dir = self._output_root / str(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        session = get_session()
        try:
            db_record = session.get(CameraSession, session_id)
            if db_record is not None:
                db_record.output_dir = str(output_dir.resolve())
                session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to update camera session output dir.")
        finally:
            session.close()

        self._reset_session_stats()
        self._active_session_id = session_id
        self._active_output_dir = output_dir

        class_colors = _generate_class_colors(run_info.class_names)

        self._stream_worker = CameraStreamWorker(
            device_index=int(device_index),
            weights_path=str(weights_path),
            class_names=run_info.class_names,
            class_colors=class_colors,
            conf=self._conf_slider.value() / 100.0,
            iou=self._iou_slider.value() / 100.0,
            output_dir=output_dir,
        )
        self._stream_worker.frame_ready.connect(self._update_frame)
        self._stream_worker.frame_meta.connect(self._consume_frame_meta)
        self._stream_worker.status.connect(self._update_status)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_worker.finished.connect(self._on_stream_finished)
        self._stream_worker.set_recording(self._record_toggle.isChecked())
        self._stream_worker.start()

        self._start_button.setText("Stop Stream")
        self._model_combo.setEnabled(False)
        self._device_combo.setEnabled(False)

    def _stop_stream(self) -> None:
        if self._stream_worker is None:
            return

        self._stream_worker.stop()

    def _update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self._preview_label.setPixmap(pixmap)

    def _consume_frame_meta(self, payload: dict[str, Any]) -> None:
        frame_number = int(payload.get("frame_number", 0))
        detections = payload.get("detections") or []
        saved_files = payload.get("saved_files") or []

        self._frame_count = max(self._frame_count, frame_number)
        self._frame_count_label.setText(f"Frames: {self._frame_count}")

        for detection in detections:
            if isinstance(detection, DetectionRecord):
                record = detection
            else:
                record = DetectionRecord(**detection)
            self._detection_log.append(record)
            self._detection_count += 1
            self._confidence_sum += record.confidence
            self._class_counts[record.class_name] = self._class_counts.get(record.class_name, 0) + 1

        if detections:
            self._detection_count_label.setText(f"Detections: {self._detection_count}")

        for path in saved_files:
            self._saved_files.append(path)

    def _update_status(self, message: str) -> None:
        self._preview_label.setText(message)

    def _on_stream_error(self, message: str) -> None:
        QMessageBox.critical(self, "Camera Error", message)
        self._stop_stream()

    def _on_stream_finished(self) -> None:
        if self._stream_worker is not None:
            self._stream_worker.deleteLater()
            self._stream_worker = None

        self._start_button.setText("Start Stream")
        self._model_combo.setEnabled(True)
        self._device_combo.setEnabled(True)

        self._finalize_session()

    def _snapshot_frame(self) -> None:
        if self._stream_worker is None or not self._stream_worker.isRunning():
            QMessageBox.information(self, "Snapshot", "Start a stream before taking a snapshot.")
            return

        self._stream_worker.request_snapshot()

    def _save_csv(self) -> None:
        if self._active_output_dir is None:
            QMessageBox.information(self, "Save CSV", "Start a session before saving CSV output.")
            return

        csv_path = self._active_output_dir / "detections.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_number", "timestamp_ms", "class_name", "confidence", "x1", "y1", "x2", "y2"])
                for record in self._detection_log:
                    writer.writerow(
                        [
                            record.frame_number,
                            record.timestamp_ms,
                            record.class_name,
                            f"{record.confidence:.4f}",
                            record.x1,
                            record.y1,
                            record.x2,
                            record.y2,
                        ]
                    )
            self._csv_saved = True
            QMessageBox.information(self, "Save CSV", f"Detections saved to:\n{csv_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save CSV", f"Could not write CSV: {exc}")

    def _reset_session_stats(self) -> None:
        self._detection_log = []
        self._class_counts = {}
        self._confidence_sum = 0.0
        self._frame_count = 0
        self._detection_count = 0
        self._saved_files = []
        self._csv_saved = False
        self._frame_count_label.setText("Frames: 0")
        self._detection_count_label.setText("Detections: 0")

    def _finalize_session(self) -> None:
        if self._active_session_id is None:
            return

        session = get_session()
        try:
            record = session.get(CameraSession, self._active_session_id)
            if record is not None:
                record.ended_at = utc_now()
                record.frame_count = self._frame_count
                record.detection_count = self._detection_count
                session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to finalize camera session.")
        finally:
            session.close()

        avg_conf = (self._confidence_sum / self._detection_count) if self._detection_count else 0.0
        class_lines = [f"{name}: {count}" for name, count in sorted(self._class_counts.items())]
        class_summary = "\n".join(class_lines) if class_lines else "-"
        files_saved = len(self._saved_files) + (1 if self._csv_saved else 0)

        output_dir = str(self._active_output_dir.resolve()) if self._active_output_dir else "-"

        QMessageBox.information(
            self,
            "Camera Session Summary",
            "\n".join(
                [
                    f"Total frames: {self._frame_count}",
                    f"Total detections: {self._detection_count}",
                    f"Detections per class:\n{class_summary}",
                    f"Average confidence: {avg_conf:.3f}",
                    f"Files saved: {files_saved}",
                    f"Output dir: {output_dir}",
                ]
            ),
        )

        self._active_session_id = None
        self._active_output_dir = None


def _generate_class_colors(class_names: list[str]) -> list[tuple[int, int, int]]:
    """Generate deterministic BGR colors per class index."""

    if not class_names:
        return [(0, 212, 170)]

    colors: list[tuple[int, int, int]] = []
    total = len(class_names)

    for idx in range(total):
        hue = idx / max(1, total)
        r, g, b = _hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))

    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB (0-1 range)."""

    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


__all__ = ["CameraTab"]

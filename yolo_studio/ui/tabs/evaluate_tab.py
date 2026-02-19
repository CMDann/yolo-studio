"""Evaluate tab for offline model testing and analytics."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from core.models.database import Dataset, RemoteDevice, RemoteTestResult, TrainingRun, get_session
from sqlalchemy.orm import joinedload

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class SavedRunInfo:
    id: int
    name: str
    architecture: str
    weights_path: str
    class_names: list[str]


@dataclass(slots=True)
class DatasetInfo:
    id: int
    name: str
    local_path: str
    class_names: list[str]


@dataclass(slots=True)
class Prediction:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class EvaluationWorker(QThread):
    """QThread worker that runs offline evaluation."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        weights_path: str,
        source_type: str,
        source_path: str,
        class_names: list[str],
        conf: float,
        iou: float,
        imgsz: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._weights_path = weights_path
        self._source_type = source_type
        self._source_path = source_path
        self._class_names = class_names
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz

    def run(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            self.error.emit(f"Ultralytics is unavailable: {exc}")
            return

        model = YOLO(self._weights_path)

        if self._source_type == "dataset":
            self._run_dataset_eval(model)
        else:
            self._run_folder_eval(model)

    def _run_dataset_eval(self, model: Any) -> None:
        self.progress.emit(0, 0, "Running dataset evaluation...")

        try:
            result = model.val(
                data=self._source_path,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                verbose=False,
                plots=False,
            )
        except Exception as exc:
            self.error.emit(f"Evaluation failed: {exc}")
            return

        payload = _extract_val_payload(result, self._class_names)
        payload["source_type"] = "dataset"
        payload["source_path"] = self._source_path
        self.progress.emit(1, 1, "Evaluation complete")
        self.finished.emit(payload)

    def _run_folder_eval(self, model: Any) -> None:
        folder = Path(self._source_path)
        images = _collect_images(folder)
        total = len(images)
        if total == 0:
            self.error.emit("No images found in selected folder.")
            return

        gt_by_image: list[list[Prediction]] = []
        pred_by_image: list[list[Prediction]] = []
        speed_ms: list[float] = []

        for idx, image_path in enumerate(images, start=1):
            self.progress.emit(idx, total, f"Processing {image_path.name}")

            label_path = image_path.with_suffix(".txt")
            gt_boxes = []
            if label_path.exists():
                gt_boxes = _load_yolo_labels(label_path, image_path)

            start = time.time()
            try:
                results = model.predict(
                    source=str(image_path),
                    conf=self._conf,
                    iou=self._iou,
                    imgsz=self._imgsz,
                    verbose=False,
                )
            except Exception as exc:
                self.error.emit(f"Prediction failed on {image_path.name}: {exc}")
                return
            elapsed_ms = (time.time() - start) * 1000.0

            preds = []
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
                        x1, y1, x2, y2 = coords
                        preds.append(
                            Prediction(
                                cls=int(cls_idx),
                                conf=float(conf),
                                x1=float(x1),
                                y1=float(y1),
                                x2=float(x2),
                                y2=float(y2),
                            )
                        )

            gt_by_image.append(gt_boxes)
            pred_by_image.append(preds)
            speed_ms.append(elapsed_ms)

        payload = _compute_folder_metrics(
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            speed_ms=speed_ms,
            class_names=self._class_names,
            iou_threshold=self._iou,
        )
        payload["source_type"] = "folder"
        payload["source_path"] = str(folder.resolve())
        self.finished.emit(payload)


class EvaluateTab(QWidget):
    """Tab that manages offline evaluation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_project_id: int | None = None
        self._project_root = PROJECT_ROOT
        self._output_root = self._project_root / "output" / "evaluations"
        self._saved_runs: dict[int, SavedRunInfo] = {}
        self._datasets: dict[int, DatasetInfo] = {}
        self._worker: EvaluationWorker | None = None
        self._latest_payload: dict[str, Any] | None = None

        self._model_combo: QComboBox
        self._source_combo: QComboBox
        self._dataset_combo: QComboBox
        self._folder_input: QLineEdit
        self._folder_button: QPushButton
        self._conf_slider: QSlider
        self._iou_slider: QSlider
        self._imgsz_spin: QSpinBox
        self._conf_value_label: QLabel
        self._iou_value_label: QLabel
        self._run_button: QPushButton
        self._export_button: QPushButton
        self._status_label: QLabel
        self._progress_label: QLabel

        self._map50_label: QLabel
        self._map_label: QLabel
        self._precision_label: QLabel
        self._recall_label: QLabel
        self._f1_label: QLabel

        self._confusion_raw_canvas: FigureCanvas
        self._confusion_norm_canvas: FigureCanvas
        self._pr_canvas: FigureCanvas
        self._f1_canvas: FigureCanvas
        self._p_canvas: FigureCanvas
        self._r_canvas: FigureCanvas
        self._class_dist_plot: pg.PlotWidget
        self._speed_plot: pg.PlotWidget

        self._build_ui()
        self.refresh_saved_models()
        self.refresh_datasets()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        self._project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._output_root = self._project_root / "output" / "evaluations"
        self.refresh_saved_models()
        self.refresh_datasets()

    def _build_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        controls = QWidget(self)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        selector_group = QGroupBox("Evaluation Setup", controls)
        selector_layout = QVBoxLayout()
        selector_layout.setContentsMargins(8, 8, 8, 8)
        selector_layout.setSpacing(8)

        self._model_combo = QComboBox(selector_group)
        self._source_combo = QComboBox(selector_group)
        self._source_combo.addItem("Dataset", "dataset")
        self._source_combo.addItem("Image Folder", "folder")
        self._source_combo.currentIndexChanged.connect(self._update_source_visibility)

        self._dataset_combo = QComboBox(selector_group)

        folder_row = QWidget(selector_group)
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(8)

        self._folder_input = QLineEdit(folder_row)
        self._folder_input.setPlaceholderText("Select a folder with images")

        self._folder_button = QPushButton("Browse", folder_row)
        self._folder_button.setProperty("secondary", True)
        self._folder_button.clicked.connect(self._browse_folder)

        folder_layout.addWidget(self._folder_input, stretch=1)
        folder_layout.addWidget(self._folder_button)
        folder_row.setLayout(folder_layout)

        form = QFormLayout()
        form.addRow("Model", self._model_combo)
        form.addRow("Input Source", self._source_combo)
        form.addRow("Dataset", self._dataset_combo)
        form.addRow("Folder", folder_row)

        selector_layout.addLayout(form)
        selector_group.setLayout(selector_layout)

        settings_group = QGroupBox("Inference Settings", controls)
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(8)

        conf_row = QWidget(settings_group)
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

        iou_row = QWidget(settings_group)
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

        imgsz_row = QWidget(settings_group)
        imgsz_layout = QHBoxLayout()
        imgsz_layout.setContentsMargins(0, 0, 0, 0)
        imgsz_layout.setSpacing(8)

        self._imgsz_spin = QSpinBox(imgsz_row)
        self._imgsz_spin.setRange(160, 4096)
        self._imgsz_spin.setValue(640)
        self._imgsz_spin.setSingleStep(32)

        imgsz_layout.addWidget(self._imgsz_spin)
        imgsz_layout.addStretch(1)
        imgsz_row.setLayout(imgsz_layout)

        settings_layout.addWidget(QLabel("Confidence", settings_group))
        settings_layout.addWidget(conf_row)
        settings_layout.addWidget(QLabel("IoU", settings_group))
        settings_layout.addWidget(iou_row)
        settings_layout.addWidget(QLabel("Image Size", settings_group))
        settings_layout.addWidget(imgsz_row)
        settings_group.setLayout(settings_layout)

        control_group = QGroupBox("Controls", controls)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(8)

        self._run_button = QPushButton("Run Evaluation", control_group)
        self._run_button.clicked.connect(self._start_evaluation)

        self._export_button = QPushButton("Export Report", control_group)
        self._export_button.setProperty("secondary", True)
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_report)

        self._status_label = QLabel("Idle", control_group)
        self._status_label.setProperty("role", "subtle")
        self._progress_label = QLabel("-", control_group)
        self._progress_label.setProperty("role", "subtle")

        control_layout.addWidget(self._run_button)
        control_layout.addWidget(self._export_button)
        control_layout.addWidget(self._status_label)
        control_layout.addWidget(self._progress_label)
        control_layout.addStretch(1)
        control_group.setLayout(control_layout)

        controls_layout.addWidget(selector_group)
        controls_layout.addWidget(settings_group)
        controls_layout.addWidget(control_group)
        controls_layout.addStretch(1)
        controls.setLayout(controls_layout)

        results_panel = QWidget(self)
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        metrics_row = QWidget(results_panel)
        metrics_layout = QHBoxLayout()
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(10)

        map50_card, self._map50_label = _metric_card("mAP50", metrics_row)
        map_card, self._map_label = _metric_card("mAP50-95", metrics_row)
        precision_card, self._precision_label = _metric_card("Precision", metrics_row)
        recall_card, self._recall_label = _metric_card("Recall", metrics_row)
        f1_card, self._f1_label = _metric_card("F1", metrics_row)

        metrics_layout.addWidget(map50_card)
        metrics_layout.addWidget(map_card)
        metrics_layout.addWidget(precision_card)
        metrics_layout.addWidget(recall_card)
        metrics_layout.addWidget(f1_card)
        metrics_layout.addStretch(1)
        metrics_row.setLayout(metrics_layout)

        tabs = QTabWidget(results_panel)
        tabs.setDocumentMode(True)

        self._confusion_raw_canvas = _create_canvas()
        self._confusion_norm_canvas = _create_canvas()
        self._pr_canvas = _create_canvas()
        self._f1_canvas = _create_canvas()
        self._p_canvas = _create_canvas()
        self._r_canvas = _create_canvas()

        self._class_dist_plot = _create_plot_widget("Class Distribution")
        self._speed_plot = _create_plot_widget("Inference Speed (ms)")

        for canvas in (
            self._confusion_raw_canvas,
            self._confusion_norm_canvas,
            self._pr_canvas,
            self._f1_canvas,
            self._p_canvas,
            self._r_canvas,
        ):
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumHeight(260)

        for plot in (self._class_dist_plot, self._speed_plot):
            plot.setMinimumHeight(260)

        tabs.addTab(self._wrap_canvas(self._confusion_raw_canvas), "Confusion (Raw)")
        tabs.addTab(self._wrap_canvas(self._confusion_norm_canvas), "Confusion (Normalized)")
        tabs.addTab(self._wrap_canvas(self._pr_canvas), "Precision-Recall")
        tabs.addTab(self._wrap_canvas(self._f1_canvas), "F1-Confidence")
        tabs.addTab(self._wrap_canvas(self._p_canvas), "Precision-Confidence")
        tabs.addTab(self._wrap_canvas(self._r_canvas), "Recall-Confidence")
        tabs.addTab(self._class_dist_plot, "Class Distribution")
        tabs.addTab(self._speed_plot, "Speed Histogram")

        results_layout.addWidget(metrics_row)
        results_layout.addWidget(tabs, stretch=1)
        results_panel.setLayout(results_layout)

        layout.addWidget(controls, stretch=1)
        layout.addWidget(results_panel, stretch=3)
        self.setLayout(layout)

        self._update_source_visibility()

    def refresh_saved_models(self) -> None:
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
            self._model_combo.addItem(f"{run.name} ({run.model_architecture})", run.id)

    def refresh_datasets(self) -> None:
        self._datasets.clear()
        self._dataset_combo.clear()

        session = get_session()
        try:
            query = session.query(Dataset)
            if self._active_project_id is not None:
                query = query.filter(Dataset.project_id == self._active_project_id)
            records = query.order_by(Dataset.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load datasets.")
            records = []
        finally:
            session.close()

        if not records:
            self._dataset_combo.addItem("No datasets", None)
            return

        for record in records:
            info = DatasetInfo(
                id=record.id,
                name=record.name,
                local_path=str(record.local_path),
                class_names=record.class_names or [],
            )
            self._datasets[record.id] = info
            self._dataset_combo.addItem(record.name, record.id)

    def _update_source_visibility(self) -> None:
        source = self._source_combo.currentData()
        is_dataset = source == "dataset"
        self._dataset_combo.setEnabled(is_dataset)
        self._folder_input.setEnabled(not is_dataset)
        self._folder_button.setEnabled(not is_dataset)

    def _browse_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if selected:
            self._folder_input.setText(str(Path(selected).resolve()))

    def _update_threshold_labels(self) -> None:
        self._conf_value_label.setText(f"{self._conf_slider.value() / 100.0:.2f}")
        self._iou_value_label.setText(f"{self._iou_slider.value() / 100.0:.2f}")

    def _start_evaluation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Evaluate", "An evaluation is already running.")
            return

        run_id = self._model_combo.currentData()
        if run_id is None:
            QMessageBox.warning(self, "Evaluate", "Select a saved model.")
            return

        run_info = self._saved_runs.get(int(run_id))
        if run_info is None:
            QMessageBox.warning(self, "Evaluate", "Selected model is no longer available.")
            return

        weights_path = Path(run_info.weights_path)
        if not weights_path.exists():
            QMessageBox.warning(self, "Evaluate", "Selected weights file does not exist.")
            return

        source_type = self._source_combo.currentData()
        if source_type == "dataset":
            dataset_id = self._dataset_combo.currentData()
            if dataset_id is None:
                QMessageBox.warning(self, "Evaluate", "Select a dataset.")
                return

            dataset_info = self._datasets.get(int(dataset_id))
            if dataset_info is None:
                QMessageBox.warning(self, "Evaluate", "Dataset is no longer available.")
                return

            dataset_yaml = _find_dataset_yaml(Path(dataset_info.local_path))
            if dataset_yaml is None:
                QMessageBox.warning(self, "Evaluate", "Dataset data.yaml was not found.")
                return

            source_path = str(dataset_yaml)
            class_names = dataset_info.class_names or run_info.class_names
        else:
            source_path = self._folder_input.text().strip()
            if not source_path:
                QMessageBox.warning(self, "Evaluate", "Select an image folder.")
                return

            if not Path(source_path).exists():
                QMessageBox.warning(self, "Evaluate", "Image folder does not exist.")
                return

            class_names = run_info.class_names

        self._status_label.setText("Running evaluation...")
        self._progress_label.setText("-")
        self._export_button.setEnabled(False)
        self._clear_plots()

        self._worker = EvaluationWorker(
            weights_path=str(weights_path),
            source_type=str(source_type),
            source_path=str(source_path),
            class_names=class_names,
            conf=self._conf_slider.value() / 100.0,
            iou=self._iou_slider.value() / 100.0,
            imgsz=int(self._imgsz_spin.value()),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            self._progress_label.setText(f"{current}/{total} - {message}")
        else:
            self._progress_label.setText(message)

    def _on_error(self, message: str) -> None:
        self._status_label.setText("Evaluation failed")
        QMessageBox.critical(self, "Evaluate", message)
        self._worker = None

    def _on_finished(self, payload: dict[str, Any]) -> None:
        self._worker = None
        self._latest_payload = payload
        self._status_label.setText("Evaluation complete")
        self._export_button.setEnabled(True)

        run_id = self._model_combo.currentData()
        if run_id is not None:
            payload["training_run_id"] = int(run_id)

        self._render_payload(payload)
        self._persist_results(payload)
        self._persist_payload_snapshot(payload)

    def _render_payload(self, payload: dict[str, Any]) -> None:
        metrics = payload.get("metrics") or {}

        self._map50_label.setText(_fmt_metric(metrics.get("map50")))
        self._map_label.setText(_fmt_metric(metrics.get("map50_95")))
        self._precision_label.setText(_fmt_metric(metrics.get("precision")))
        self._recall_label.setText(_fmt_metric(metrics.get("recall")))
        self._f1_label.setText(_fmt_metric(metrics.get("f1")))

        confusion = payload.get("confusion_matrix")
        classes = payload.get("class_names") or []
        if confusion is not None:
            _plot_confusion(self._confusion_raw_canvas, confusion, classes, normalize=False)
            _plot_confusion(self._confusion_norm_canvas, confusion, classes, normalize=True)
        else:
            _plot_empty(self._confusion_raw_canvas, "No confusion matrix")
            _plot_empty(self._confusion_norm_canvas, "No confusion matrix")

        curves = payload.get("curves") or {}
        _plot_pr_curves(self._pr_canvas, curves.get("pr") or {}, classes)
        _plot_conf_curve(self._f1_canvas, curves.get("f1") or {}, classes, "F1")
        _plot_conf_curve(self._p_canvas, curves.get("p") or {}, classes, "Precision")
        _plot_conf_curve(self._r_canvas, curves.get("r") or {}, classes, "Recall")

        class_dist = payload.get("class_distribution") or {}
        _plot_class_distribution(self._class_dist_plot, class_dist, classes)

        speeds = payload.get("speed_ms") or []
        _plot_speed_histogram(self._speed_plot, speeds)

    def _persist_results(self, payload: dict[str, Any]) -> None:
        run_id = self._model_combo.currentData()
        if run_id is None:
            return

        source_type = str(payload.get("source_type") or "")
        source_path = str(payload.get("source_path") or "")

        device_id = _ensure_local_device()
        if device_id is None:
            return

        metrics = payload.get("metrics") or {}
        num_images = int(payload.get("num_images") or 0)

        session = get_session()
        try:
            record = RemoteTestResult(
                device_id=device_id,
                training_run_id=int(run_id),
                run_at=datetime.now(timezone.utc),
                test_dataset_path=source_path or "-",
                source_type=source_type,
                source_path=source_path,
                num_images_tested=num_images,
                map50=_to_float(metrics.get("map50")),
                map50_95=_to_float(metrics.get("map50_95")),
                precision=_to_float(metrics.get("precision")),
                recall=_to_float(metrics.get("recall")),
                output_images_dir=None,
                notes="Offline evaluation",
            )
            session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to save evaluation results.")
        finally:
            session.close()

    def _persist_payload_snapshot(self, payload: dict[str, Any]) -> None:
        run_id = payload.get("training_run_id")
        if run_id is None:
            return

        self._output_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = self._output_root / f"run_{int(run_id)}.json"
        try:
            snapshot_path.write_text(
                json.dumps(payload, indent=2, default=_json_fallback),
                encoding="utf-8",
            )
        except Exception:
            LOGGER.exception("Failed to write evaluation snapshot: %s", snapshot_path)

    def _export_report(self) -> None:
        if not self._latest_payload:
            QMessageBox.information(self, "Export", "No evaluation results to export.")
            return

        self._output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_pdf = self._output_root / f"evaluation_{timestamp}.pdf"

        pdf_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Evaluation Report",
            str(default_pdf),
            "PDF Files (*.pdf)",
        )
        if not pdf_path_str:
            return

        pdf_path = Path(pdf_path_str)
        json_path = pdf_path.with_suffix(".json")

        try:
            _export_pdf(pdf_path, self._latest_payload)
            json_path.write_text(json.dumps(self._latest_payload, indent=2), encoding="utf-8")
            QMessageBox.information(self, "Export", f"Report saved to:\n{pdf_path}\n{json_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export", f"Export failed: {exc}")

    def _clear_plots(self) -> None:
        _plot_empty(self._confusion_raw_canvas, "Waiting for results")
        _plot_empty(self._confusion_norm_canvas, "Waiting for results")
        _plot_empty(self._pr_canvas, "Waiting for results")
        _plot_empty(self._f1_canvas, "Waiting for results")
        _plot_empty(self._p_canvas, "Waiting for results")
        _plot_empty(self._r_canvas, "Waiting for results")
        self._class_dist_plot.clear()
        self._speed_plot.clear()
        for label in (
            self._map50_label,
            self._map_label,
            self._precision_label,
            self._recall_label,
            self._f1_label,
        ):
            label.setText("-")

    def _wrap_canvas(self, canvas: FigureCanvas) -> QWidget:
        wrapper = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        wrapper.setLayout(layout)
        return wrapper


def _metric_card(title: str, parent: QWidget) -> tuple[QFrame, QLabel]:
    container = QFrame(parent)
    container.setFrameShape(QFrame.Shape.StyledPanel)
    layout = QVBoxLayout()
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(2)

    label = QLabel("-", container)
    label.setObjectName("metricValue")

    title_label = QLabel(title, container)
    title_label.setProperty("role", "subtle")

    layout.addWidget(label)
    layout.addWidget(title_label)
    container.setLayout(layout)
    return container, label


def _create_canvas() -> FigureCanvas:
    fig = Figure(figsize=(4, 3), tight_layout=True)
    canvas = FigureCanvas(fig)
    return canvas


def _create_plot_widget(title: str) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground("w")
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setTitle(title)
    return plot


def _plot_empty(canvas: FigureCanvas, message: str) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    canvas.draw()


def _plot_confusion(canvas: FigureCanvas, matrix: np.ndarray, classes: list[str], normalize: bool) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)

    data = matrix.copy().astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = data / row_sums

    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    size = data.shape[0]
    tick_labels = classes + ["bg"]
    if len(tick_labels) != size:
        tick_labels = [str(idx) for idx in range(size - 1)] + ["bg"]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Normalized" if normalize else "Raw")

    canvas.draw()


def _plot_pr_curves(canvas: FigureCanvas, pr_curves: dict[int, dict[str, list[float]]], classes: list[str]) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)

    if not pr_curves:
        ax.text(0.5, 0.5, "No PR curves", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw()
        return

    for cls_idx, curve in pr_curves.items():
        recall = curve.get("recall", [])
        precision = curve.get("precision", [])
        try:
            cls_index = int(cls_idx)
        except Exception:
            cls_index = -1
        label = classes[cls_index] if 0 <= cls_index < len(classes) else str(cls_idx)
        ax.plot(recall, precision, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(fontsize=7, ncol=2, loc="best")
    canvas.draw()


def _plot_conf_curve(canvas: FigureCanvas, curves: dict[str, list[float]], classes: list[str], title: str) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)

    conf = curves.get("conf", [])
    values = curves.get("value", [])
    if not conf or not values:
        ax.text(0.5, 0.5, f"No {title} curve", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw()
        return

    ax.plot(conf, values, label=title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel(title)
    ax.set_title(f"{title} vs Confidence")
    ax.legend(fontsize=8)
    canvas.draw()


def _plot_class_distribution(plot: pg.PlotWidget, dist: dict[str, list[int]], classes: list[str]) -> None:
    plot.clear()
    gt = dist.get("gt") or []
    pred = dist.get("pred") or []
    if not gt and not pred:
        return

    count = max(len(gt), len(pred), len(classes))
    if count == 0:
        return
    x = np.arange(count)
    width = 0.4
    if gt:
        bar_gt = pg.BarGraphItem(
            x=x - width / 2,
            height=gt + [0] * (count - len(gt)),
            width=width,
            brush=pg.mkBrush(0, 123, 255, 180),
        )
        plot.addItem(bar_gt)
    if pred:
        bar_pred = pg.BarGraphItem(
            x=x + width / 2,
            height=pred + [0] * (count - len(pred)),
            width=width,
            brush=pg.mkBrush(255, 99, 71, 180),
        )
        plot.addItem(bar_pred)

    ticks = [(idx, classes[idx] if idx < len(classes) else str(idx)) for idx in range(count)]
    plot.getAxis("bottom").setTicks([ticks])


def _plot_speed_histogram(plot: pg.PlotWidget, speeds: list[float]) -> None:
    plot.clear()
    if not speeds:
        return

    hist, edges = np.histogram(speeds, bins=min(20, max(5, len(speeds) // 4)))
    x = edges[:-1]
    width = np.diff(edges)
    bar = pg.BarGraphItem(x=x, height=hist, width=width, brush=pg.mkBrush(34, 197, 94, 180))
    plot.addItem(bar)


def _extract_val_payload(result: Any, class_names: list[str]) -> dict[str, Any]:
    metrics = {
        "map50": _safe_get(result, ["box.map50", "metrics.box.map50"]),
        "map50_95": _safe_get(result, ["box.map", "metrics.box.map"]),
        "precision": _safe_get(result, ["box.p", "metrics.box.p"]),
        "recall": _safe_get(result, ["box.r", "metrics.box.r"]),
        "f1": _safe_get(result, ["box.f1", "metrics.box.f1"]),
    }

    names = class_names
    result_names = _safe_get(result, ["names", "model.names"])
    if isinstance(result_names, dict):
        names = [result_names[idx] for idx in sorted(result_names.keys())]
    elif isinstance(result_names, list):
        names = [str(n) for n in result_names]

    confusion = None
    confusion_obj = _safe_get(result, ["confusion_matrix", "metrics.confusion_matrix"])
    if confusion_obj is not None:
        matrix = getattr(confusion_obj, "matrix", None) or getattr(confusion_obj, "confusion_matrix", None)
        if matrix is not None:
            confusion = np.array(matrix)

    curves = {}
    pr_curve = _safe_get(result, ["box.pr_curve", "metrics.box.pr_curve"])
    pr_recall = _safe_get(result, ["box.pr", "metrics.box.pr"])
    if pr_curve is not None and pr_recall is not None:
        pr_payload = {}
        for idx in range(len(pr_curve)):
            pr_payload[idx] = {
                "precision": pr_curve[idx].tolist() if hasattr(pr_curve[idx], "tolist") else list(pr_curve[idx]),
                "recall": pr_recall.tolist() if hasattr(pr_recall, "tolist") else list(pr_recall),
            }
        curves["pr"] = pr_payload

    conf = _safe_get(result, ["box.conf", "metrics.box.conf"])
    f1_curve = _safe_get(result, ["box.f1_curve", "metrics.box.f1_curve"])
    p_curve = _safe_get(result, ["box.p_curve", "metrics.box.p_curve"])
    r_curve = _safe_get(result, ["box.r_curve", "metrics.box.r_curve"])

    if conf is not None:
        conf_list = conf.tolist() if hasattr(conf, "tolist") else list(conf)
        if f1_curve is not None:
            curves["f1"] = {"conf": conf_list, "value": _tolist(f1_curve)}
        if p_curve is not None:
            curves["p"] = {"conf": conf_list, "value": _tolist(p_curve)}
        if r_curve is not None:
            curves["r"] = {"conf": conf_list, "value": _tolist(r_curve)}

    speed = getattr(result, "speed", {}) or {}
    num_images = int(_safe_get(result, ["seen", "box.seen"], default=0) or 0)
    speed_ms = []
    if speed and num_images:
        inference_ms = float(speed.get("inference", 0.0))
        speed_ms = [inference_ms] * num_images

    class_dist = _derive_class_distribution(confusion, len(names)) if confusion is not None else {}

    per_class_map = None
    ap = _safe_get(result, ["box.ap", "metrics.box.ap"])
    if ap is not None:
        ap_values = ap.tolist() if hasattr(ap, "tolist") else list(ap)
        per_class_map = {idx: float(val) for idx, val in enumerate(ap_values)}

    return {
        "metrics": _clean_metrics(metrics),
        "confusion_matrix": confusion,
        "curves": curves,
        "class_distribution": class_dist,
        "speed_ms": speed_ms,
        "class_names": names,
        "num_images": num_images,
        "per_class_map": per_class_map,
    }


def _compute_folder_metrics(
    gt_by_image: list[list[Prediction]],
    pred_by_image: list[list[Prediction]],
    speed_ms: list[float],
    class_names: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    gt_counts = [0] * num_classes
    pred_counts = [0] * num_classes

    for gt_boxes in gt_by_image:
        for box in gt_boxes:
            if 0 <= box.cls < num_classes:
                gt_counts[box.cls] += 1

    for preds in pred_by_image:
        for pred in preds:
            if 0 <= pred.cls < num_classes:
                pred_counts[pred.cls] += 1

    for gt_boxes, preds in zip(gt_by_image, pred_by_image):
        matches = _match_detections(gt_boxes, preds, iou_threshold)
        for gt_idx, pred_idx in matches["matches"]:
            gt_cls = gt_boxes[gt_idx].cls
            pred_cls = preds[pred_idx].cls
            confusion[gt_cls, pred_cls] += 1
        for pred_idx in matches["unmatched_preds"]:
            pred_cls = preds[pred_idx].cls
            confusion[num_classes, pred_cls] += 1
        for gt_idx in matches["unmatched_gts"]:
            gt_cls = gt_boxes[gt_idx].cls
            confusion[gt_cls, num_classes] += 1

    pr_curves = _compute_pr_curves(gt_by_image, pred_by_image, num_classes, iou_threshold)
    f1_curve = _aggregate_curve(pr_curves, "f1")
    p_curve = _aggregate_curve(pr_curves, "precision")
    r_curve = _aggregate_curve(pr_curves, "recall")

    metrics = _aggregate_metrics(confusion)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "curves": {
            "pr": pr_curves,
            "f1": f1_curve,
            "p": p_curve,
            "r": r_curve,
        },
        "class_distribution": {"gt": gt_counts, "pred": pred_counts},
        "speed_ms": speed_ms,
        "class_names": class_names,
        "num_images": len(gt_by_image),
    }


def _match_detections(gt_boxes: list[Prediction], preds: list[Prediction], iou_threshold: float) -> dict[str, Any]:
    matches = []
    unmatched_preds = list(range(len(preds)))
    unmatched_gts = list(range(len(gt_boxes)))

    if not gt_boxes or not preds:
        return {"matches": matches, "unmatched_preds": unmatched_preds, "unmatched_gts": unmatched_gts}

    preds_sorted = sorted(enumerate(preds), key=lambda item: item[1].conf, reverse=True)
    used_gts = set()

    for pred_idx, pred in preds_sorted:
        best_iou = 0.0
        best_gt = None
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gts:
                continue
            iou = _box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_gt is not None and best_iou >= iou_threshold:
            matches.append((best_gt, pred_idx))
            used_gts.add(best_gt)

    unmatched_preds = [idx for idx in range(len(preds)) if idx not in {p for _, p in matches}]
    unmatched_gts = [idx for idx in range(len(gt_boxes)) if idx not in {g for g, _ in matches}]

    return {"matches": matches, "unmatched_preds": unmatched_preds, "unmatched_gts": unmatched_gts}


def _compute_pr_curves(
    gt_by_image: list[list[Prediction]],
    pred_by_image: list[list[Prediction]],
    num_classes: int,
    iou_threshold: float,
) -> dict[int, dict[str, list[float]]]:
    curves: dict[int, dict[str, list[float]]] = {}

    for cls_idx in range(num_classes):
        scores = []
        matches = []
        n_gt = 0

        for gt_boxes, preds in zip(gt_by_image, pred_by_image):
            gt_cls = [g for g in gt_boxes if g.cls == cls_idx]
            n_gt += len(gt_cls)
            preds_cls = [p for p in preds if p.cls == cls_idx]
            preds_cls_sorted = sorted(preds_cls, key=lambda p: p.conf, reverse=True)

            used = set()
            for pred in preds_cls_sorted:
                best_iou = 0.0
                best_gt = None
                for gt_idx, gt in enumerate(gt_cls):
                    if gt_idx in used:
                        continue
                    iou = _box_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx

                if best_gt is not None and best_iou >= iou_threshold:
                    matches.append(1)
                    used.add(best_gt)
                else:
                    matches.append(0)

                scores.append(pred.conf)

        if not scores:
            curves[cls_idx] = {"precision": [], "recall": [], "conf": [], "f1": []}
            continue

        order = np.argsort(-np.array(scores))
        scores_sorted = np.array(scores)[order]
        matches_sorted = np.array(matches)[order]

        tp = np.cumsum(matches_sorted)
        fp = np.cumsum(1 - matches_sorted)
        precision = tp / np.maximum(tp + fp, 1e-9)
        recall = tp / max(n_gt, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

        curves[cls_idx] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "conf": scores_sorted.tolist(),
            "f1": f1.tolist(),
        }

    return curves


def _aggregate_curve(curves: dict[int, dict[str, list[float]]], key: str) -> dict[str, list[float]]:
    if not curves:
        return {"conf": [], "value": []}

    all_conf = []
    all_values = []
    for curve in curves.values():
        conf = curve.get("conf") or []
        values = curve.get(key) or []
        if conf and values:
            all_conf.append(conf)
            all_values.append(values)

    if not all_conf:
        return {"conf": [], "value": []}

    min_len = min(len(c) for c in all_conf)
    conf = np.array([c[:min_len] for c in all_conf]).mean(axis=0)
    values = np.array([v[:min_len] for v in all_values]).mean(axis=0)
    return {"conf": conf.tolist(), "value": values.tolist()}


def _aggregate_metrics(confusion: np.ndarray) -> dict[str, float]:
    num_classes = confusion.shape[0] - 1

    tp = np.diag(confusion)[:num_classes]
    fp = confusion[num_classes, :num_classes]
    fn = confusion[:num_classes, num_classes]

    precision = tp.sum() / max((tp + fp).sum(), 1)
    recall = tp.sum() / max((tp + fn).sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": 0.0,
        "map50_95": 0.0,
    }


def _derive_class_distribution(confusion: np.ndarray, num_classes: int) -> dict[str, list[int]]:
    gt = confusion[:num_classes, :].sum(axis=1).tolist()
    pred = confusion[:, :num_classes].sum(axis=0).tolist()
    return {"gt": [int(v) for v in gt], "pred": [int(v) for v in pred]}


def _safe_get(obj: Any, paths: list[str], default: Any | None = None) -> Any:
    for path in paths:
        current = obj
        ok = True
        for part in path.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                ok = False
                break
        if ok:
            return current
    return default


def _tolist(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return []


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned = {}
    for key, value in metrics.items():
        cleaned[key] = _to_float(value)
    return cleaned


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def _collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
    filtered = [
        path
        for path in images
        if not path.name.startswith("._") and not path.name.startswith(".")
    ]
    return sorted(filtered)


def _load_yolo_labels(label_path: Path, image_path: Path) -> list[Prediction]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    height, width = image.shape[:2]
    labels = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_idx = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        x2 = (cx + w / 2.0) * width
        y2 = (cy + h / 2.0) * height
        labels.append(Prediction(cls_idx, 1.0, x1, y1, x2, y2))
    return labels


def _box_iou(a: Prediction, b: Prediction) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _find_dataset_yaml(dataset_root: Path) -> Path | None:
    candidates = [dataset_root / "data.yaml", dataset_root / "dataset.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in dataset_root.glob("*.yaml"):
        if candidate.exists():
            return candidate

    for candidate in dataset_root.glob("*.yml"):
        if candidate.exists():
            return candidate

    return None


def _ensure_local_device() -> int | None:
    session = get_session()
    try:
        device = (
            session.query(RemoteDevice)
            .filter(RemoteDevice.name == "Local Evaluation")
            .order_by(RemoteDevice.id.asc())
            .first()
        )
        if device is not None:
            return int(device.id)

        device = RemoteDevice(
            name="Local Evaluation",
            host="localhost",
            port=0,
            auth_token="local",
        )
        session.add(device)
        session.commit()
        session.refresh(device)
        return int(device.id)
    except Exception:
        session.rollback()
        LOGGER.exception("Unable to create Local Evaluation device.")
        return None
    finally:
        session.close()


def _export_pdf(path: Path, payload: dict[str, Any]) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        fig = _build_summary_figure(payload)
        pdf.savefig(fig)

        fig = _build_confusion_figure(payload.get("confusion_matrix"), payload.get("class_names") or [])
        pdf.savefig(fig)

        fig = _build_pr_figure(payload.get("curves") or {}, payload.get("class_names") or [])
        pdf.savefig(fig)

        fig = _build_curve_figure(payload.get("curves") or {}, "f1", "F1")
        pdf.savefig(fig)

        fig = _build_curve_figure(payload.get("curves") or {}, "p", "Precision")
        pdf.savefig(fig)

        fig = _build_curve_figure(payload.get("curves") or {}, "r", "Recall")
        pdf.savefig(fig)


def _build_summary_figure(payload: dict[str, Any]) -> Figure:
    fig = Figure(figsize=(8, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.axis("off")

    metrics = payload.get("metrics") or {}
    lines = [
        f"Source: {payload.get('source_type')} - {payload.get('source_path')}",
        f"mAP50: {metrics.get('map50', 0):.3f}",
        f"mAP50-95: {metrics.get('map50_95', 0):.3f}",
        f"Precision: {metrics.get('precision', 0):.3f}",
        f"Recall: {metrics.get('recall', 0):.3f}",
        f"F1: {metrics.get('f1', 0):.3f}",
        f"Num images: {payload.get('num_images', 0)}",
    ]

    ax.text(0.01, 0.95, "\n".join(lines), va="top")
    return fig


def _build_confusion_figure(matrix: Any, classes: list[str]) -> Figure:
    fig = Figure(figsize=(6, 5), tight_layout=True)
    ax = fig.add_subplot(111)
    if matrix is None:
        ax.text(0.5, 0.5, "No confusion matrix", ha="center", va="center")
        ax.axis("off")
        return fig

    data = np.array(matrix)
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_labels = classes + ["bg"]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    return fig


def _build_pr_figure(curves: dict[str, Any], classes: list[str]) -> Figure:
    fig = Figure(figsize=(6, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    pr_curves = curves.get("pr") or {}
    if not pr_curves:
        ax.text(0.5, 0.5, "No PR curves", ha="center", va="center")
        ax.axis("off")
        return fig

    for cls_idx, curve in pr_curves.items():
        recall = curve.get("recall", [])
        precision = curve.get("precision", [])
        try:
            cls_index = int(cls_idx)
        except Exception:
            cls_index = -1
        label = classes[cls_index] if 0 <= cls_index < len(classes) else str(cls_idx)
        ax.plot(recall, precision, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(fontsize=7, ncol=2, loc="best")
    return fig


def _build_curve_figure(curves: dict[str, Any], key: str, title: str) -> Figure:
    fig = Figure(figsize=(6, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    curve = curves.get(key) or {}
    conf = curve.get("conf", [])
    values = curve.get("value", [])
    if not conf or not values:
        ax.text(0.5, 0.5, f"No {title} curve", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.plot(conf, values, label=title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel(title)
    ax.set_title(f"{title} vs Confidence")
    ax.legend(fontsize=8)
    return fig


def _json_fallback(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


__all__ = ["EvaluateTab"]

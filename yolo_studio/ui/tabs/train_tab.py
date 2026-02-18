"""Training tab UI for configuring and running YOLO training sessions."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.models.database import Dataset, TrainingRun, get_session
from ui.widgets.file_drop_zone import FileDropZone
from ui.widgets.log_panel import LogPanel
from ui.widgets.metric_chart import MetricChart

try:
    from core.workers.trainer import YOLOTrainer
except Exception:  # pragma: no cover - populated in step 6
    YOLOTrainer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

MODEL_ARCHITECTURES: tuple[str, ...] = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov11l",
    "yolov11x",
)


@dataclass(slots=True)
class DatasetSummary:
    """Lightweight dataset payload used for training-tab dropdowns."""

    id: int
    name: str
    source: str
    local_path: str
    class_names: list[str]
    num_images: int
    num_classes: int
    tags: list[str]


class TrainTab(QWidget):
    """Tab that manages YOLO training setup, execution, and run persistence."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the training tab and all controls.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._project_root = Path(__file__).resolve().parents[2]
        self._datasets: dict[int, DatasetSummary] = {}
        self._trainer: Any = None
        self._split_paths: dict[str, str] = {}
        self._generated_data_yaml_path: str | None = None
        self._latest_training_result: dict[str, Any] | None = None
        self._best_map50: float | None = None
        self._best_map50_95: float | None = None

        self._dataset_combo: QComboBox
        self._dataset_meta_label: QLabel

        self._train_drop_zone: FileDropZone
        self._val_drop_zone: FileDropZone
        self._test_drop_zone: FileDropZone

        self._data_yaml_path_input: QLineEdit

        self._model_combo: QComboBox
        self._epochs_spin: QSpinBox
        self._batch_size_spin: QSpinBox
        self._image_size_spin: QSpinBox
        self._learning_rate_spin: QDoubleSpinBox
        self._optimizer_combo: QComboBox
        self._warmup_epochs_spin: QDoubleSpinBox
        self._weight_decay_spin: QDoubleSpinBox

        self._mosaic_checkbox: QCheckBox
        self._mixup_checkbox: QCheckBox
        self._copy_paste_checkbox: QCheckBox
        self._hsv_checkbox: QCheckBox
        self._flip_checkbox: QCheckBox

        self._use_pretrained_checkbox: QCheckBox
        self._use_custom_weights_checkbox: QCheckBox
        self._custom_weights_input: QLineEdit
        self._browse_custom_weights_button: QPushButton

        self._run_name_input: QLineEdit
        self._notes_input: QTextEdit

        self._start_button: QPushButton
        self._stop_button: QPushButton
        self._save_run_button: QPushButton

        self._metric_chart: MetricChart
        self._epoch_progress_bar: QProgressBar
        self._epoch_progress_label: QLabel
        self._best_map_label: QLabel
        self._log_panel: LogPanel

        self._build_ui()
        self.refresh_datasets()

    def refresh_datasets(self) -> None:
        """Reload dataset options from SQLite and refresh the metadata card."""

        selected_id = self._dataset_combo.currentData()
        self._datasets.clear()
        self._dataset_combo.clear()

        session = get_session()
        try:
            records = session.query(Dataset).order_by(Dataset.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load datasets for training tab.")
            records = []
        finally:
            session.close()

        for record in records:
            summary = DatasetSummary(
                id=record.id,
                name=record.name,
                source=record.source.value,
                local_path=record.local_path,
                class_names=record.class_names or [],
                num_images=record.num_images or 0,
                num_classes=record.num_classes or 0,
                tags=record.tags or [],
            )
            self._datasets[summary.id] = summary
            self._dataset_combo.addItem(f"{summary.name} (#{summary.id})", summary.id)

        if self._dataset_combo.count() == 0:
            self._dataset_meta_label.setText("No datasets found in the library. Add one from the Dataset tab.")
            self._dataset_meta_label.setProperty("role", "warning")
            self._dataset_meta_label.style().unpolish(self._dataset_meta_label)
            self._dataset_meta_label.style().polish(self._dataset_meta_label)
            return

        if selected_id is not None:
            index = self._dataset_combo.findData(selected_id)
            if index >= 0:
                self._dataset_combo.setCurrentIndex(index)

        self._on_dataset_changed(self._dataset_combo.currentIndex())

    def _build_ui(self) -> None:
        """Compose left configuration and right monitoring panels."""

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([780, 820])

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(splitter)
        self.setLayout(root_layout)

    def _build_left_panel(self) -> QWidget:
        """Create the configuration form panel with scrolling support."""

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(scroll)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)

        layout.addWidget(self._build_model_dataset_group())
        layout.addWidget(self._build_dataset_drop_group())
        layout.addWidget(self._build_hyperparameter_group())
        layout.addWidget(self._build_augmentation_group())
        layout.addWidget(self._build_pretrained_group())
        layout.addWidget(self._build_run_metadata_group())
        layout.addWidget(self._build_control_group())
        layout.addStretch(1)

        container.setLayout(layout)
        scroll.setWidget(container)
        return scroll

    def _build_model_dataset_group(self) -> QGroupBox:
        """Create model architecture and dataset selection controls."""

        group = QGroupBox("Model + Dataset")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setSpacing(10)

        self._model_combo = QComboBox(group)
        self._model_combo.addItems(list(MODEL_ARCHITECTURES))
        self._model_combo.setCurrentText("yolov8n")

        dataset_row = QWidget(group)
        dataset_row_layout = QHBoxLayout()
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(8)

        self._dataset_combo = QComboBox(dataset_row)
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        refresh_button = QPushButton("Refresh", dataset_row)
        refresh_button.setProperty("secondary", True)
        refresh_button.clicked.connect(self.refresh_datasets)

        dataset_row_layout.addWidget(self._dataset_combo, stretch=1)
        dataset_row_layout.addWidget(refresh_button)
        dataset_row.setLayout(dataset_row_layout)

        self._dataset_meta_label = QLabel("Select a dataset to view metadata.", group)
        self._dataset_meta_label.setWordWrap(True)
        self._dataset_meta_label.setObjectName("metricValue")

        form.addRow("Architecture", self._model_combo)
        form.addRow("Dataset", dataset_row)
        form.addRow("Dataset Card", self._dataset_meta_label)

        group.setLayout(form)
        return group

    def _build_dataset_drop_group(self) -> QGroupBox:
        """Create train/val/test drop zones and generated data.yaml controls."""

        group = QGroupBox("Dataset Inputs")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self._train_drop_zone = FileDropZone("Train Images Folder", "Drop train folder (images+labels)", group)
        self._val_drop_zone = FileDropZone("Val Images Folder", "Drop val folder (images+labels)", group)
        self._test_drop_zone = FileDropZone("Test Images Folder", "Drop test folder (optional)", group)

        self._train_drop_zone.path_selected.connect(lambda path: self._on_split_path_changed("train", path))
        self._val_drop_zone.path_selected.connect(lambda path: self._on_split_path_changed("val", path))
        self._test_drop_zone.path_selected.connect(lambda path: self._on_split_path_changed("test", path))

        self._data_yaml_path_input = QLineEdit(group)
        self._data_yaml_path_input.setReadOnly(True)
        self._data_yaml_path_input.setPlaceholderText("Auto-generated data.yaml path")
        self._data_yaml_path_input.setObjectName("codeField")

        yaml_row = QWidget(group)
        yaml_row_layout = QHBoxLayout()
        yaml_row_layout.setContentsMargins(0, 0, 0, 0)
        yaml_row_layout.setSpacing(8)

        browse_yaml_button = QPushButton("Use Existing data.yaml", yaml_row)
        browse_yaml_button.setProperty("secondary", True)
        browse_yaml_button.clicked.connect(self._select_existing_data_yaml)

        clear_yaml_button = QPushButton("Clear", yaml_row)
        clear_yaml_button.clicked.connect(self._clear_data_yaml)

        yaml_row_layout.addWidget(self._data_yaml_path_input, stretch=1)
        yaml_row_layout.addWidget(browse_yaml_button)
        yaml_row_layout.addWidget(clear_yaml_button)
        yaml_row.setLayout(yaml_row_layout)

        layout.addWidget(self._train_drop_zone)
        layout.addWidget(self._val_drop_zone)
        layout.addWidget(self._test_drop_zone)
        layout.addWidget(QLabel("data.yaml", group))
        layout.addWidget(yaml_row)

        group.setLayout(layout)
        return group

    def _build_hyperparameter_group(self) -> QGroupBox:
        """Create core hyperparameter fields for YOLO training."""

        group = QGroupBox("Hyperparameters")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        self._epochs_spin = QSpinBox(group)
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(100)

        self._batch_size_spin = QSpinBox(group)
        self._batch_size_spin.setRange(1, 1024)
        self._batch_size_spin.setValue(16)

        self._image_size_spin = QSpinBox(group)
        self._image_size_spin.setRange(64, 4096)
        self._image_size_spin.setSingleStep(32)
        self._image_size_spin.setValue(640)

        self._learning_rate_spin = QDoubleSpinBox(group)
        self._learning_rate_spin.setRange(0.000001, 1.0)
        self._learning_rate_spin.setDecimals(6)
        self._learning_rate_spin.setSingleStep(0.0001)
        self._learning_rate_spin.setValue(0.01)

        self._optimizer_combo = QComboBox(group)
        self._optimizer_combo.addItems(["SGD", "Adam", "AdamW", "RMSProp"])

        self._warmup_epochs_spin = QDoubleSpinBox(group)
        self._warmup_epochs_spin.setRange(0.0, 20.0)
        self._warmup_epochs_spin.setDecimals(1)
        self._warmup_epochs_spin.setSingleStep(0.5)
        self._warmup_epochs_spin.setValue(3.0)

        self._weight_decay_spin = QDoubleSpinBox(group)
        self._weight_decay_spin.setRange(0.0, 0.1)
        self._weight_decay_spin.setDecimals(6)
        self._weight_decay_spin.setSingleStep(0.0001)
        self._weight_decay_spin.setValue(0.0005)

        grid.addWidget(QLabel("Epochs", group), 0, 0)
        grid.addWidget(self._epochs_spin, 0, 1)
        grid.addWidget(QLabel("Batch Size", group), 0, 2)
        grid.addWidget(self._batch_size_spin, 0, 3)

        grid.addWidget(QLabel("Image Size", group), 1, 0)
        grid.addWidget(self._image_size_spin, 1, 1)
        grid.addWidget(QLabel("Learning Rate", group), 1, 2)
        grid.addWidget(self._learning_rate_spin, 1, 3)

        grid.addWidget(QLabel("Optimizer", group), 2, 0)
        grid.addWidget(self._optimizer_combo, 2, 1)
        grid.addWidget(QLabel("Warmup Epochs", group), 2, 2)
        grid.addWidget(self._warmup_epochs_spin, 2, 3)

        grid.addWidget(QLabel("Weight Decay", group), 3, 0)
        grid.addWidget(self._weight_decay_spin, 3, 1)

        group.setLayout(grid)
        return group

    def _build_augmentation_group(self) -> QGroupBox:
        """Create augmentation feature toggles."""

        group = QGroupBox("Augmentation")
        row = QHBoxLayout()
        row.setSpacing(14)

        self._mosaic_checkbox = QCheckBox("Mosaic", group)
        self._mosaic_checkbox.setChecked(True)

        self._mixup_checkbox = QCheckBox("Mixup", group)
        self._mixup_checkbox.setChecked(False)

        self._copy_paste_checkbox = QCheckBox("Copy Paste", group)
        self._copy_paste_checkbox.setChecked(False)

        self._hsv_checkbox = QCheckBox("HSV", group)
        self._hsv_checkbox.setChecked(True)

        self._flip_checkbox = QCheckBox("Flip", group)
        self._flip_checkbox.setChecked(True)

        row.addWidget(self._mosaic_checkbox)
        row.addWidget(self._mixup_checkbox)
        row.addWidget(self._copy_paste_checkbox)
        row.addWidget(self._hsv_checkbox)
        row.addWidget(self._flip_checkbox)
        row.addStretch(1)

        group.setLayout(row)
        return group

    def _build_pretrained_group(self) -> QGroupBox:
        """Create controls for default/custom pretrained weight behavior."""

        group = QGroupBox("Pretrained Weights")
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self._use_pretrained_checkbox = QCheckBox("Use pretrained initialization", group)
        self._use_pretrained_checkbox.setChecked(True)
        self._use_pretrained_checkbox.toggled.connect(self._on_pretrained_mode_changed)

        self._use_custom_weights_checkbox = QCheckBox("Use custom .pt file", group)
        self._use_custom_weights_checkbox.setChecked(False)
        self._use_custom_weights_checkbox.toggled.connect(self._on_pretrained_mode_changed)

        custom_row = QWidget(group)
        custom_row_layout = QHBoxLayout()
        custom_row_layout.setContentsMargins(0, 0, 0, 0)
        custom_row_layout.setSpacing(8)

        self._custom_weights_input = QLineEdit(custom_row)
        self._custom_weights_input.setPlaceholderText("Path to custom pretrained weights")
        self._custom_weights_input.setObjectName("codeField")

        self._browse_custom_weights_button = QPushButton("Browse .pt", custom_row)
        self._browse_custom_weights_button.setProperty("secondary", True)
        self._browse_custom_weights_button.clicked.connect(self._select_custom_weights)

        custom_row_layout.addWidget(self._custom_weights_input, stretch=1)
        custom_row_layout.addWidget(self._browse_custom_weights_button)
        custom_row.setLayout(custom_row_layout)

        layout.addWidget(self._use_pretrained_checkbox)
        layout.addWidget(self._use_custom_weights_checkbox)
        layout.addWidget(custom_row)

        group.setLayout(layout)
        self._on_pretrained_mode_changed()
        return group

    def _build_run_metadata_group(self) -> QGroupBox:
        """Create training-run metadata fields."""

        group = QGroupBox("Run Details")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self._run_name_input = QLineEdit(timestamp, group)
        self._notes_input = QTextEdit(group)
        self._notes_input.setPlaceholderText("Optional notes for this training run")
        self._notes_input.setMinimumHeight(90)

        form.addRow("Run Name", self._run_name_input)
        form.addRow("Notes", self._notes_input)

        group.setLayout(form)
        return group

    def _build_control_group(self) -> QGroupBox:
        """Create start/stop controls for launching worker thread training."""

        group = QGroupBox("Controls")
        row = QHBoxLayout()
        row.setSpacing(10)

        self._start_button = QPushButton("Start Training", group)
        self._start_button.clicked.connect(self._start_training)

        self._stop_button = QPushButton("Stop Training", group)
        self._stop_button.setProperty("danger", True)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._stop_training)

        row.addWidget(self._start_button)
        row.addWidget(self._stop_button)
        row.addStretch(1)

        group.setLayout(row)
        return group

    def _build_right_panel(self) -> QWidget:
        """Create live training monitoring panel with metrics and logs."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        monitor_group = QGroupBox("Live Monitoring", panel)
        monitor_layout = QVBoxLayout()
        monitor_layout.setSpacing(10)

        self._metric_chart = MetricChart(monitor_group)
        self._metric_chart.setMinimumHeight(320)

        progress_row = QWidget(monitor_group)
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)

        self._epoch_progress_label = QLabel("Epoch: 0/0", progress_row)
        self._epoch_progress_label.setObjectName("metricValue")

        self._best_map_label = QLabel("Best mAP50: - | mAP50-95: -", progress_row)
        self._best_map_label.setObjectName("metricBadge")
        self._best_map_label.setProperty("role", "success")

        progress_layout.addWidget(self._epoch_progress_label)
        progress_layout.addWidget(self._best_map_label)
        progress_layout.addStretch(1)
        progress_row.setLayout(progress_layout)

        self._epoch_progress_bar = QProgressBar(monitor_group)
        self._epoch_progress_bar.setRange(0, 100)
        self._epoch_progress_bar.setValue(0)

        self._log_panel = LogPanel(monitor_group)

        self._save_run_button = QPushButton("Save This Run", monitor_group)
        self._save_run_button.setProperty("secondary", True)
        self._save_run_button.setEnabled(False)
        self._save_run_button.clicked.connect(self._save_current_run)

        monitor_layout.addWidget(self._metric_chart)
        monitor_layout.addWidget(progress_row)
        monitor_layout.addWidget(self._epoch_progress_bar)
        monitor_layout.addWidget(self._log_panel, stretch=1)
        monitor_layout.addWidget(self._save_run_button, alignment=Qt.AlignmentFlag.AlignRight)

        monitor_group.setLayout(monitor_layout)

        layout.addWidget(monitor_group)
        panel.setLayout(layout)
        return panel

    def _on_dataset_changed(self, _index: int) -> None:
        """Update dataset card and inferred data.yaml path on dropdown change.

        Args:
            _index: Current combo index (unused).
        """

        dataset_id = self._dataset_combo.currentData()
        if dataset_id is None or dataset_id not in self._datasets:
            self._dataset_meta_label.setText("No dataset selected.")
            self._dataset_meta_label.setProperty("role", "warning")
            self._refresh_label_style(self._dataset_meta_label)
            return

        dataset = self._datasets[dataset_id]
        tags = ", ".join(dataset.tags) if dataset.tags else "-"
        classes = ", ".join(dataset.class_names[:8])
        if len(dataset.class_names) > 8:
            classes += ", ..."

        self._dataset_meta_label.setText(
            f"Source: {dataset.source}\n"
            f"Images: {dataset.num_images} | Classes: {dataset.num_classes}\n"
            f"Path: {dataset.local_path}\n"
            f"Class Names: {classes or '-'}\n"
            f"Tags: {tags}"
        )
        self._dataset_meta_label.setProperty("role", "")
        self._refresh_label_style(self._dataset_meta_label)

        candidate_yaml = Path(dataset.local_path) / "data.yaml"
        if candidate_yaml.exists() and not self._generated_data_yaml_path:
            self._set_data_yaml_path(str(candidate_yaml.resolve()))

    def _on_split_path_changed(self, split_name: str, path: str) -> None:
        """Record split-folder updates and regenerate data.yaml when possible.

        Args:
            split_name: One of train/val/test.
            path: Absolute directory path.
        """

        self._split_paths[split_name] = path
        self._try_generate_data_yaml()

    def _try_generate_data_yaml(self) -> None:
        """Generate a temporary data.yaml when train/val folders are configured."""

        train_path = self._split_paths.get("train")
        val_path = self._split_paths.get("val")
        if not train_path or not val_path:
            return

        run_slug = self._slugify(self._run_name_input.text().strip() or "run")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        generated_dir = self._project_root / "datasets" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = generated_dir / f"{run_slug}_{timestamp}_data.yaml"

        selected_dataset = self._datasets.get(self._dataset_combo.currentData())
        names = self._resolve_generated_class_names(selected_dataset)

        dump_payload: dict[str, Any] = {
            "path": "",
            "train": str(Path(train_path).resolve()),
            "val": str(Path(val_path).resolve()),
            "names": names,
            "nc": len(names),
        }

        test_path = self._split_paths.get("test")
        if test_path:
            dump_payload["test"] = str(Path(test_path).resolve())

        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dump_payload, handle, sort_keys=False)

        self._generated_data_yaml_path = str(yaml_path)
        self._set_data_yaml_path(self._generated_data_yaml_path)

        self._append_log(
            f"Generated data.yaml at {self._generated_data_yaml_path} "
            f"({len(names)} classes)"
        )

    def _resolve_generated_class_names(self, selected_dataset: DatasetSummary | None) -> list[str]:
        """Resolve class names for auto-generated data.yaml.

        Priority:
            1) Selected dataset metadata class names.
            2) Nearby YAML files (dataset/split-parent locations).
            3) Class IDs inferred from split label files.

        Args:
            selected_dataset: Currently selected dataset summary, if any.

        Returns:
            list[str]: Normalized class names.
        """

        if selected_dataset is not None:
            names = self._normalize_class_names(selected_dataset.class_names)
            if names:
                return names

        for yaml_path in self._candidate_data_yaml_paths(selected_dataset):
            names = self._load_class_names_from_yaml(yaml_path)
            if names:
                return names

        return self._infer_class_names_from_split_labels()

    def _candidate_data_yaml_paths(self, selected_dataset: DatasetSummary | None) -> list[Path]:
        """Build candidate YAML paths for class-name discovery.

        Args:
            selected_dataset: Currently selected dataset summary.

        Returns:
            list[Path]: Ordered candidate YAML files.
        """

        candidates: list[Path] = []
        seen: set[Path] = set()

        def add_directory(directory: Path) -> None:
            for file_name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
                candidate = (directory / file_name).resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        if selected_dataset and selected_dataset.local_path:
            add_directory(Path(selected_dataset.local_path).resolve())

        for key in ("train", "val", "test"):
            raw_path = self._split_paths.get(key)
            if not raw_path:
                continue

            split_path = Path(raw_path).resolve()
            add_directory(split_path)
            add_directory(split_path.parent)
            add_directory(split_path.parent.parent)

        return candidates

    def _infer_class_names_from_split_labels(self) -> list[str]:
        """Infer synthetic class names from label IDs inside split folders.

        Returns:
            list[str]: Synthetic names like `class_0`, or empty when none found.
        """

        label_roots: list[Path] = []
        seen: set[Path] = set()

        def add_root(path: Path) -> None:
            if not path.exists() or not path.is_dir():
                return
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            label_roots.append(resolved)

        for key in ("train", "val", "test"):
            raw_path = self._split_paths.get(key)
            if not raw_path:
                continue

            split_path = Path(raw_path).resolve()
            add_root(split_path)

            labels_child = split_path / "labels"
            if labels_child.exists():
                add_root(labels_child)

            if split_path.name.lower() == "images":
                sibling_labels = split_path.parent / "labels"
                if sibling_labels.exists():
                    add_root(sibling_labels)

        max_class_id = -1
        for label_root in label_roots:
            for label_path in label_root.rglob("*.txt"):
                try:
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_id = int(float(parts[0]))
                        if class_id >= 0:
                            max_class_id = max(max_class_id, class_id)
                except Exception:
                    continue

        if max_class_id < 0:
            return []

        return [f"class_{index}" for index in range(max_class_id + 1)]

    def _set_data_yaml_path(self, path: str) -> None:
        """Apply an explicit data.yaml path to the read-only field.

        Args:
            path: Absolute path to a YOLO-format data.yaml file.
        """

        self._data_yaml_path_input.setText(path)

    def _select_existing_data_yaml(self) -> None:
        """Allow the user to manually select a data.yaml file."""

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select data.yaml",
            str(self._project_root),
            "YAML Files (*.yaml *.yml)",
        )
        if not selected:
            return

        self._generated_data_yaml_path = None
        self._set_data_yaml_path(str(Path(selected).resolve()))

    def _clear_data_yaml(self) -> None:
        """Clear selected/generated data.yaml reference."""

        self._generated_data_yaml_path = None
        self._data_yaml_path_input.clear()

    def _on_pretrained_mode_changed(self, _checked: bool | None = None) -> None:
        """Enable or disable custom-weight controls based on toggles."""

        has_pretrained = self._use_pretrained_checkbox.isChecked()
        self._use_custom_weights_checkbox.setEnabled(has_pretrained)

        if not has_pretrained:
            self._use_custom_weights_checkbox.setChecked(False)

        allow_custom = self._use_pretrained_checkbox.isChecked() and self._use_custom_weights_checkbox.isChecked()
        self._custom_weights_input.setEnabled(allow_custom)
        self._browse_custom_weights_button.setEnabled(allow_custom)

    def _select_custom_weights(self) -> None:
        """Open a .pt file picker for custom pretrained weights."""

        if not self._custom_weights_input.isEnabled():
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pretrained Weights",
            str(self._project_root),
            "PyTorch Weights (*.pt)",
        )
        if selected:
            self._custom_weights_input.setText(str(Path(selected).resolve()))

    def _start_training(self) -> None:
        """Collect config and launch YOLO training in a worker thread."""

        if YOLOTrainer is None:
            QMessageBox.critical(
                self,
                "Trainer Unavailable",
                "YOLOTrainer is not implemented yet. Step 6 will provide the training backend.",
            )
            return

        if self._trainer is not None and self._trainer.isRunning():
            QMessageBox.information(self, "Training Running", "A training run is already in progress.")
            return

        config = self._collect_training_config()
        if config is None:
            return

        self._latest_training_result = None
        self._best_map50 = None
        self._best_map50_95 = None
        self._save_run_button.setEnabled(False)
        self._metric_chart.reset()
        self._log_panel.clear()
        self._epoch_progress_bar.setValue(0)
        self._epoch_progress_label.setText("Epoch: 0/0")
        self._best_map_label.setText("Best mAP50: - | mAP50-95: -")

        try:
            self._trainer = YOLOTrainer(config)
        except Exception as exc:
            self._append_log(f"[ERROR] Failed to initialize trainer: {exc}")
            QMessageBox.critical(self, "Trainer Initialization Error", str(exc))
            return
        self._trainer.progress.connect(self._on_progress)
        self._trainer.status.connect(self._on_status)
        self._trainer.finished.connect(self._on_training_finished)
        self._trainer.error.connect(self._on_training_error)

        if hasattr(self._trainer, "metrics"):
            self._trainer.metrics.connect(self._on_metrics)

        if hasattr(self._trainer, "log"):
            self._trainer.log.connect(self._append_log)

        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)

        self._append_log("Starting training run...")
        self._trainer.start()

    def _stop_training(self) -> None:
        """Request graceful early stop from the running trainer worker."""

        if self._trainer is None or not self._trainer.isRunning():
            return

        if hasattr(self._trainer, "request_stop"):
            self._trainer.request_stop()
            self._append_log("Stop requested. Waiting for current epoch to complete...")
            self._stop_button.setEnabled(False)
            return

        self._append_log("Trainer does not support graceful stop. Ignoring stop request.")

    def _collect_training_config(self) -> dict[str, Any] | None:
        """Build and validate the training configuration payload.

        Returns:
            dict[str, Any] | None: Valid training config, or None if validation fails.
        """

        run_name = self._run_name_input.text().strip()
        if not run_name:
            QMessageBox.warning(self, "Validation Error", "Training run name is required.")
            return None

        dataset_id = self._dataset_combo.currentData()
        if dataset_id is None:
            QMessageBox.warning(self, "Validation Error", "Please select a dataset.")
            return None

        data_yaml_path = self._data_yaml_path_input.text().strip()
        if not data_yaml_path:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please select or generate a data.yaml file before starting training.",
            )
            return None

        if not Path(data_yaml_path).exists():
            QMessageBox.warning(self, "Validation Error", "Configured data.yaml path does not exist.")
            return None

        class_count = self._read_class_count_from_data_yaml(Path(data_yaml_path))
        if class_count <= 0:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Configured data.yaml defines zero classes. "
                "Regenerate data.yaml from split folders or choose a YAML with valid names/nc.",
            )
            return None

        use_pretrained = self._use_pretrained_checkbox.isChecked()
        use_custom_weights = self._use_custom_weights_checkbox.isChecked() and use_pretrained
        custom_weights_path = self._custom_weights_input.text().strip()

        if use_custom_weights and not custom_weights_path:
            QMessageBox.warning(self, "Validation Error", "Custom .pt path is enabled but not set.")
            return None

        if custom_weights_path and not Path(custom_weights_path).exists():
            QMessageBox.warning(self, "Validation Error", "Selected custom .pt file does not exist.")
            return None

        config: dict[str, Any] = {
            "run_name": run_name,
            "notes": self._notes_input.toPlainText().strip(),
            "dataset_id": int(dataset_id),
            "model_architecture": self._model_combo.currentText(),
            "data_yaml_path": data_yaml_path,
            "epochs": int(self._epochs_spin.value()),
            "batch_size": int(self._batch_size_spin.value()),
            "image_size": int(self._image_size_spin.value()),
            "learning_rate": float(self._learning_rate_spin.value()),
            "optimizer": self._optimizer_combo.currentText(),
            "warmup_epochs": float(self._warmup_epochs_spin.value()),
            "weight_decay": float(self._weight_decay_spin.value()),
            "augmentation": {
                "mosaic": self._mosaic_checkbox.isChecked(),
                "mixup": self._mixup_checkbox.isChecked(),
                "copy_paste": self._copy_paste_checkbox.isChecked(),
                "hsv": self._hsv_checkbox.isChecked(),
                "flip": self._flip_checkbox.isChecked(),
            },
            "use_pretrained": use_pretrained,
            "custom_weights_path": custom_weights_path if use_custom_weights else None,
            "output_root": str((self._project_root / "runs" / "train").resolve()),
        }

        return config

    def _on_progress(self, progress: int) -> None:
        """Update training progress bar.

        Args:
            progress: Integer percentage from 0-100.
        """

        self._epoch_progress_bar.setValue(max(0, min(progress, 100)))

    def _on_status(self, status: str) -> None:
        """Append status updates from the trainer thread.

        Args:
            status: Status line emitted by trainer.
        """

        self._append_log(f"[STATUS] {status}")

    def _on_metrics(self, payload: dict[str, Any]) -> None:
        """Update chart, progress labels, and best mAP badges from epoch metrics.

        Args:
            payload: Metric message emitted by trainer.
        """

        epoch = int(payload.get("epoch", 0))
        total_epochs = int(payload.get("total_epochs", 0))

        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            cast_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            self._metric_chart.append_epoch_metrics(epoch, cast_metrics)

            map50 = cast_metrics.get("metrics/mAP50(B)")
            map50_95 = cast_metrics.get("metrics/mAP50-95(B)")
            if map50 is not None or map50_95 is not None:
                if map50 is not None:
                    self._best_map50 = map50 if self._best_map50 is None else max(self._best_map50, map50)
                if map50_95 is not None:
                    self._best_map50_95 = (
                        map50_95 if self._best_map50_95 is None else max(self._best_map50_95, map50_95)
                    )
                self._update_best_map_badge()

        if total_epochs > 0 and epoch > 0:
            percentage = int((epoch / total_epochs) * 100)
            self._epoch_progress_bar.setValue(max(0, min(percentage, 100)))
            self._epoch_progress_label.setText(f"Epoch: {epoch}/{total_epochs}")

    def _on_training_finished(self, result: dict[str, Any]) -> None:
        """Handle successful trainer completion payload.

        Args:
            result: Final result dictionary emitted by trainer.
        """

        self._latest_training_result = result

        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._epoch_progress_bar.setValue(100)

        weights_path = result.get("weights_path")
        run_id = result.get("training_run_id")

        if weights_path and Path(str(weights_path)).exists() and run_id is not None:
            self._save_run_button.setEnabled(True)

        self._append_log("Training finished.")
        if result:
            self._append_log(f"Result: {result}")

        self._trainer = None

    def _on_training_error(self, message: str) -> None:
        """Handle training worker errors and reset run controls.

        Args:
            message: Error text emitted by trainer.
        """

        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Training Error", message)
        self._trainer = None

    def _save_current_run(self) -> None:
        """Mark a completed run as saved and copy weights to managed storage."""

        if not self._latest_training_result:
            QMessageBox.information(self, "No Run", "No completed run is available to save.")
            return

        run_id = self._latest_training_result.get("training_run_id")
        weights_path = self._latest_training_result.get("weights_path")

        if run_id is None or not weights_path:
            QMessageBox.warning(self, "Missing Metadata", "Training run metadata is incomplete.")
            return

        source_weights = Path(str(weights_path))
        if not source_weights.exists():
            QMessageBox.warning(self, "Missing Weights", "Trained weights file no longer exists.")
            return

        saved_models_dir = self._project_root / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)

        safe_run_name = self._slugify(self._run_name_input.text().strip() or "saved_model")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = saved_models_dir / f"{safe_run_name}_{timestamp}.pt"

        shutil.copy2(source_weights, target_path)

        existing_notes = self._notes_input.toPlainText().strip()
        notes, accepted = QInputDialog.getMultiLineText(
            self,
            "Save This Run",
            "Optional notes for this saved model:",
            existing_notes,
        )

        if not accepted:
            notes = existing_notes

        session = get_session()
        try:
            run = session.get(TrainingRun, int(run_id))
            if run is None:
                raise ValueError(f"TrainingRun #{run_id} not found")

            run.is_saved = True
            run.weights_path = str(target_path.resolve())
            run.completed_at = run.completed_at or datetime.now(timezone.utc)
            if notes.strip():
                run.notes = notes.strip()

            session.commit()
        except Exception as exc:
            session.rollback()
            target_path.unlink(missing_ok=True)
            QMessageBox.critical(self, "Save Failed", f"Could not mark run as saved: {exc}")
            return
        finally:
            session.close()

        self._latest_training_result["weights_path"] = str(target_path.resolve())
        self._append_log(f"Saved model copied to {target_path.resolve()}")
        QMessageBox.information(self, "Run Saved", f"Saved model: {target_path.name}")

    def _append_log(self, message: str) -> None:
        """Append one line to the tab-local log panel.

        Args:
            message: Log text.
        """

        self._log_panel.append_message(message)

    def _update_best_map_badge(self) -> None:
        """Refresh the best-metric badge using tracked maxima."""

        if self._best_map50 is None and self._best_map50_95 is None:
            self._best_map_label.setText("Best mAP50: - | mAP50-95: -")
            return

        map50_text = f"{self._best_map50:.4f}" if self._best_map50 is not None else "-"
        map50_95_text = f"{self._best_map50_95:.4f}" if self._best_map50_95 is not None else "-"
        self._best_map_label.setText(f"Best mAP50: {map50_text} | mAP50-95: {map50_95_text}")

    @staticmethod
    def _slugify(value: str) -> str:
        """Create a filesystem-safe slug from arbitrary text.

        Args:
            value: Arbitrary user-provided text.

        Returns:
            str: Safe slug for filenames.
        """

        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
        return cleaned or "run"

    @staticmethod
    def _load_yaml_payload(yaml_path: Path) -> dict[str, Any]:
        """Safely load a YAML dictionary payload.

        Args:
            yaml_path: YAML file to parse.

        Returns:
            dict[str, Any]: Parsed mapping, or empty dict when invalid.
        """

        if not yaml_path.exists():
            return {}

        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            LOGGER.exception("Could not parse YAML: %s", yaml_path)
            return {}

        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _load_class_names_from_yaml(cls, yaml_path: Path) -> list[str]:
        """Extract class names from a YOLO data.yaml file.

        Args:
            yaml_path: YAML file path.

        Returns:
            list[str]: Parsed class names.
        """

        payload = cls._load_yaml_payload(yaml_path)
        if not payload:
            return []

        names = payload.get("names")
        if isinstance(names, list):
            return cls._normalize_class_names(names)

        if isinstance(names, dict):
            ordered_keys = sorted(
                names.keys(),
                key=lambda raw: (0, int(raw)) if str(raw).isdigit() else (1, str(raw)),
            )
            return cls._normalize_class_names([names[key] for key in ordered_keys])

        return []

    @classmethod
    def _read_class_count_from_data_yaml(cls, yaml_path: Path) -> int:
        """Resolve class count from a YOLO data.yaml file.

        Args:
            yaml_path: YAML file path.

        Returns:
            int: Number of classes, or zero when unavailable.
        """

        names = cls._load_class_names_from_yaml(yaml_path)
        if names:
            return len(names)

        payload = cls._load_yaml_payload(yaml_path)
        nc = payload.get("nc")
        if isinstance(nc, (int, float)):
            return max(0, int(nc))
        if isinstance(nc, str) and nc.strip().isdigit():
            return max(0, int(nc.strip()))

        return 0

    @staticmethod
    def _normalize_class_names(raw_names: list[Any]) -> list[str]:
        """Normalize and deduplicate class names while preserving order.

        Args:
            raw_names: Candidate class names.

        Returns:
            list[str]: Cleaned class-name list.
        """

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_name in raw_names:
            name = str(raw_name).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    @staticmethod
    def _refresh_label_style(label: QLabel) -> None:
        """Re-polish a label so dynamic property styles are applied.

        Args:
            label: Target label widget.
        """

        label.style().unpolish(label)
        label.style().polish(label)


__all__ = ["TrainTab"]

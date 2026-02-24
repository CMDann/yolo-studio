"""Dataset management tab for browsing, building, and saving YOLO datasets/models."""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.models.database import Dataset, DatasetSource, RemoteDevice, TrainingRun, get_session
from sqlalchemy.orm import joinedload

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None


LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DatasetRow:
    """Projection object for displaying datasets in the table."""

    id: int
    name: str
    source: str
    num_classes: int
    num_images: int
    created_at: str
    tags: str
    local_path: str
    description: str
    class_names: list[str]


class ImageDropListWidget(QListWidget):
    """List widget that accepts image files via drag-and-drop."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the drop-enabled list widget.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event: Any) -> None:
        """Accept drag events that include local file URLs.

        Args:
            event: Drag-enter event.
        """

        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        """Allow moving accepted drag items over the widget.

        Args:
            event: Drag-move event.
        """

        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: Any) -> None:
        """Emit resolved dropped image files.

        Args:
            event: Drop event.
        """

        resolved: list[str] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                resolved.extend(_collect_images_from_directory(path))
            elif _is_image_file(path):
                resolved.append(str(path.resolve()))

        if resolved:
            self.files_dropped.emit(_dedupe_preserve_order(resolved))
            event.acceptProposedAction()
            return

        event.ignore()


class DatasetFormDialog(QDialog):
    """Dialog for creating or editing dataset metadata records."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        initial: dict[str, str] | None = None,
        allow_source_change: bool = False,
    ) -> None:
        """Initialize the dataset metadata dialog.

        Args:
            title: Dialog title text.
            parent: Optional parent widget.
            initial: Optional initial field values.
            allow_source_change: Whether source text is editable.
        """

        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 360)

        values = initial or {}

        self._name_input = QLineEdit(values.get("name", ""), self)
        self._source_input = QLineEdit(values.get("source", DatasetSource.MANUAL.value), self)
        self._source_input.setReadOnly(not allow_source_change)
        self._path_input = QLineEdit(values.get("local_path", ""), self)
        self._description_input = QTextEdit(self)
        self._description_input.setPlainText(values.get("description", ""))
        self._tags_input = QLineEdit(values.get("tags", ""), self)

        browse_button = QPushButton("Browse", self)
        browse_button.setProperty("secondary", True)
        browse_button.clicked.connect(self._browse_path)

        path_row = QWidget(self)
        path_row_layout = QHBoxLayout()
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.setSpacing(6)
        path_row_layout.addWidget(self._path_input, stretch=1)
        path_row_layout.addWidget(browse_button)
        path_row.setLayout(path_row_layout)

        form = QFormLayout()
        form.addRow("Name", self._name_input)
        form.addRow("Source", self._source_input)
        form.addRow("Local Path", path_row)
        form.addRow("Tags (comma-separated)", self._tags_input)
        form.addRow("Description", self._description_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def values(self) -> dict[str, Any]:
        """Return normalized dialog field values.

        Returns:
            dict[str, Any]: Normalized metadata fields.
        """

        tags = [part.strip() for part in self._tags_input.text().split(",") if part.strip()]
        return {
            "name": self._name_input.text().strip(),
            "source": self._source_input.text().strip() or DatasetSource.MANUAL.value,
            "local_path": self._path_input.text().strip(),
            "description": self._description_input.toPlainText().strip(),
            "tags": tags,
        }

    def _browse_path(self) -> None:
        """Open folder picker for dataset path."""

        selected = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if selected:
            self._path_input.setText(str(Path(selected).resolve()))


class DatasetTab(QWidget):
    """Main tab for dataset library and saved model management."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the dataset tab and compose all sub-panels.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._active_project_id: int | None = None
        self._project_root = PROJECT_ROOT
        self._dataset_library_root = self._project_root / "datasets" / "library"
        self._builder_preview_root = self._project_root / "datasets" / "builder_preview"

        self._dataset_rows: dict[int, DatasetRow] = {}
        self._builder_images: list[str] = []
        self._builder_yaml_path: str | None = None
        self._saved_run_rows: dict[int, dict[str, Any]] = {}

        self._dataset_table: QTableWidget
        self._dataset_detail: QTextEdit

        self._builder_image_list: ImageDropListWidget
        self._builder_preview_label: QLabel
        self._builder_preview_meta_label: QLabel
        self._builder_class_list: QListWidget
        self._builder_train_slider: QSlider
        self._builder_val_slider: QSlider
        self._builder_split_label: QLabel
        self._builder_name_input: QLineEdit
        self._builder_tags_input: QLineEdit
        self._builder_description_input: QTextEdit
        self._builder_yaml_display: QLineEdit

        self._labeling_root_input: QLineEdit
        self._labeling_status_label: QLabel
        self._labeling_tool_input: QLineEdit

        self._saved_models_table: QTableWidget

        self._build_ui()
        self.refresh_datasets()
        self.refresh_saved_models()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        if project_root:
            self._project_root = Path(project_root)
        else:
            self._project_root = PROJECT_ROOT
        self._dataset_library_root = self._project_root / "datasets" / "library"
        self._builder_preview_root = self._project_root / "datasets" / "builder_preview"
        self.refresh_datasets()
        self.refresh_saved_models()

    def refresh_datasets(self) -> None:
        """Reload datasets from SQLite and repopulate the library table."""

        self._dataset_rows.clear()
        self._dataset_table.setRowCount(0)

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

        for record in records:
            created = record.created_at.isoformat() if record.created_at else "-"
            row = DatasetRow(
                id=record.id,
                name=record.name,
                source=record.source.value,
                num_classes=record.num_classes or 0,
                num_images=record.num_images or 0,
                created_at=created,
                tags=", ".join(record.tags or []),
                local_path=record.local_path,
                description=record.description or "",
                class_names=record.class_names or [],
            )
            self._add_dataset_row(row)

        if self._dataset_table.rowCount() > 0:
            self._dataset_table.selectRow(0)
            self._update_dataset_detail()
        else:
            self._dataset_detail.setPlainText("No datasets registered.")

    def refresh_saved_models(self) -> None:
        """Reload saved-model rows from SQLite and repopulate the table."""

        self._saved_run_rows.clear()
        self._saved_models_table.setRowCount(0)

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

        for run in runs:
            dataset_name = run.dataset.name if run.dataset is not None else "-"
            date_saved = run.completed_at.isoformat() if run.completed_at else "-"

            self._saved_run_rows[run.id] = {
                "id": run.id,
                "name": run.name,
                "architecture": run.model_architecture,
                "dataset": dataset_name,
                "map50": run.best_map50,
                "map50_95": run.best_map50_95,
                "date_saved": date_saved,
                "notes": run.notes or "",
                "weights_path": run.weights_path,
            }

            row = self._saved_models_table.rowCount()
            self._saved_models_table.insertRow(row)

            values = [
                run.name,
                run.model_architecture,
                dataset_name,
                _fmt_metric(run.best_map50),
                _fmt_metric(run.best_map50_95),
                date_saved,
                (run.notes or "").replace("\n", " "),
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, run.id)
                self._saved_models_table.setItem(row, col, item)

        self._saved_models_table.resizeColumnsToContents()

    def _build_ui(self) -> None:
        """Create complete dataset tab layout."""

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_library_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([620, 1040])

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(splitter)
        self.setLayout(root_layout)

    def _build_library_panel(self) -> QWidget:
        """Create the dataset library browser panel."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QWidget(panel)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)

        new_button = QPushButton("New Dataset", toolbar)
        new_button.clicked.connect(self._create_manual_dataset)

        import_button = QPushButton("Import from Folder", toolbar)
        import_button.setProperty("secondary", True)
        import_button.clicked.connect(self._import_dataset_folder)

        refresh_button = QPushButton("Refresh", toolbar)
        refresh_button.setProperty("secondary", True)
        refresh_button.clicked.connect(self.refresh_datasets)

        toolbar_layout.addWidget(new_button)
        toolbar_layout.addWidget(import_button)
        toolbar_layout.addWidget(refresh_button)
        toolbar_layout.addStretch(1)
        toolbar.setLayout(toolbar_layout)

        self._dataset_table = QTableWidget(0, 6, panel)
        self._dataset_table.setHorizontalHeaderLabels(
            ["Name", "Source", "Classes", "Images", "Created", "Tags"]
        )
        self._dataset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._dataset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._dataset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dataset_table.setAlternatingRowColors(True)
        self._dataset_table.verticalHeader().setVisible(False)
        self._dataset_table.itemSelectionChanged.connect(self._update_dataset_detail)
        self._dataset_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._dataset_table.customContextMenuRequested.connect(self._open_dataset_context_menu)

        detail_group = QGroupBox("Dataset Details", panel)
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(6, 6, 6, 6)

        self._dataset_detail = QTextEdit(detail_group)
        self._dataset_detail.setReadOnly(True)
        self._dataset_detail.setObjectName("codeField")

        detail_layout.addWidget(self._dataset_detail)
        detail_group.setLayout(detail_layout)

        layout.addWidget(toolbar)
        layout.addWidget(self._dataset_table, stretch=3)
        layout.addWidget(detail_group, stretch=2)
        panel.setLayout(layout)

        return panel

    def _build_right_panel(self) -> QWidget:
        """Create right-side tab container for builder and saved models."""

        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_builder_panel(), "Dataset Builder")
        tabs.addTab(self._build_labeling_panel(), "Labeling")
        tabs.addTab(self._build_saved_models_panel(), "Saved Models")
        return tabs

    def _build_labeling_panel(self) -> QWidget:
        """Create labeling helper panel for creating label folders."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        setup_group = QGroupBox("Labeling Setup", panel)
        setup_layout = QFormLayout()

        root_row = QWidget(setup_group)
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(6)

        self._labeling_root_input = QLineEdit(root_row)
        self._labeling_root_input.setPlaceholderText("Select dataset root (contains images/ or train/val/test)")

        browse_button = QPushButton("Browse", root_row)
        browse_button.setProperty("secondary", True)
        browse_button.clicked.connect(self._browse_labeling_root)

        scan_button = QPushButton("Scan", root_row)
        scan_button.clicked.connect(self._scan_labeling_root)

        root_layout.addWidget(self._labeling_root_input, stretch=1)
        root_layout.addWidget(browse_button)
        root_layout.addWidget(scan_button)
        root_row.setLayout(root_layout)

        self._labeling_status_label = QLabel("Select a dataset folder to begin.", setup_group)
        self._labeling_status_label.setProperty("role", "subtle")

        setup_layout.addRow("Dataset Root", root_row)
        setup_layout.addRow(self._labeling_status_label)
        setup_group.setLayout(setup_layout)

        tool_group = QGroupBox("Annotation Tool", panel)
        tool_layout = QFormLayout()

        tool_row = QWidget(tool_group)
        tool_row_layout = QHBoxLayout()
        tool_row_layout.setContentsMargins(0, 0, 0, 0)
        tool_row_layout.setSpacing(6)

        self._labeling_tool_input = QLineEdit(tool_row)
        self._labeling_tool_input.setPlaceholderText("Path to annotation tool executable")

        tool_browse_button = QPushButton("Browse", tool_row)
        tool_browse_button.setProperty("secondary", True)
        tool_browse_button.clicked.connect(self._browse_labeling_tool)

        tool_row_layout.addWidget(self._labeling_tool_input, stretch=1)
        tool_row_layout.addWidget(tool_browse_button)
        tool_row.setLayout(tool_row_layout)

        tool_layout.addRow("Tool Path", tool_row)
        tool_group.setLayout(tool_layout)

        action_group = QGroupBox("Actions", panel)
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(6, 6, 6, 6)
        action_layout.setSpacing(6)

        create_button = QPushButton("Create Label Folders", action_group)
        create_button.clicked.connect(self._create_label_folders)

        empty_button = QPushButton("Create Empty Labels", action_group)
        empty_button.setProperty("secondary", True)
        empty_button.clicked.connect(self._create_empty_label_files)

        open_button = QPushButton("Open Labels Folder", action_group)
        open_button.setProperty("secondary", True)
        open_button.clicked.connect(self._open_labels_folder)

        launch_button = QPushButton("Launch Tool", action_group)
        launch_button.clicked.connect(self._launch_labeling_tool)

        action_layout.addWidget(create_button)
        action_layout.addWidget(empty_button)
        action_layout.addWidget(open_button)
        action_layout.addWidget(launch_button)
        action_layout.addStretch(1)
        action_group.setLayout(action_layout)

        layout.addWidget(setup_group)
        layout.addWidget(tool_group)
        layout.addWidget(action_group)
        layout.addStretch(1)
        panel.setLayout(layout)
        return panel

    def _build_builder_panel(self) -> QWidget:
        """Create dataset builder panel with preview and class manager."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_row = QWidget(panel)
        top_row_layout = QHBoxLayout()
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(6)

        add_images_button = QPushButton("Add Images", top_row)
        add_images_button.clicked.connect(self._add_builder_images)

        clear_images_button = QPushButton("Clear", top_row)
        clear_images_button.setProperty("secondary", True)
        clear_images_button.clicked.connect(self._clear_builder_images)

        top_row_layout.addWidget(add_images_button)
        top_row_layout.addWidget(clear_images_button)
        top_row_layout.addStretch(1)
        top_row.setLayout(top_row_layout)

        content_splitter = QSplitter(Qt.Orientation.Horizontal, panel)
        content_splitter.setChildrenCollapsible(False)

        left_group = QGroupBox("Image Files", panel)
        left_layout = QVBoxLayout()
        self._builder_image_list = ImageDropListWidget(left_group)
        self._builder_image_list.files_dropped.connect(self._add_builder_images_from_paths)
        self._builder_image_list.itemSelectionChanged.connect(self._update_builder_preview)
        self._builder_image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._builder_image_list.setAlternatingRowColors(True)
        self._builder_image_list.setToolTip("Drop image files or folders here.")
        left_layout.addWidget(self._builder_image_list)
        left_group.setLayout(left_layout)

        right_container = QWidget(panel)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        preview_group = QGroupBox("Annotation Preview", right_container)
        preview_layout = QVBoxLayout()

        self._builder_preview_label = QLabel("Select an image to preview annotations.", preview_group)
        self._builder_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._builder_preview_label.setMinimumHeight(260)
        self._builder_preview_label.setFrameShape(QFrame.Shape.StyledPanel)

        self._builder_preview_meta_label = QLabel("-", preview_group)
        self._builder_preview_meta_label.setProperty("role", "subtle")

        preview_layout.addWidget(self._builder_preview_label)
        preview_layout.addWidget(self._builder_preview_meta_label)
        preview_group.setLayout(preview_layout)

        class_group = QGroupBox("Class Labels", right_container)
        class_layout = QGridLayout()
        class_layout.setHorizontalSpacing(8)
        class_layout.setVerticalSpacing(8)

        self._builder_class_list = QListWidget(class_group)
        self._builder_class_list.setAlternatingRowColors(True)

        class_add_btn = QPushButton("Add", class_group)
        class_add_btn.clicked.connect(self._add_builder_class)

        class_remove_btn = QPushButton("Remove", class_group)
        class_remove_btn.setProperty("secondary", True)
        class_remove_btn.clicked.connect(self._remove_builder_class)

        class_rename_btn = QPushButton("Rename", class_group)
        class_rename_btn.setProperty("secondary", True)
        class_rename_btn.clicked.connect(self._rename_builder_class)

        class_layout.addWidget(self._builder_class_list, 0, 0, 3, 1)
        class_layout.addWidget(class_add_btn, 0, 1)
        class_layout.addWidget(class_remove_btn, 1, 1)
        class_layout.addWidget(class_rename_btn, 2, 1)
        class_group.setLayout(class_layout)

        split_group = QGroupBox("Train/Val/Test Split", right_container)
        split_layout = QGridLayout()

        self._builder_train_slider = QSlider(Qt.Orientation.Horizontal, split_group)
        self._builder_train_slider.setRange(50, 90)
        self._builder_train_slider.setValue(70)
        self._builder_train_slider.valueChanged.connect(self._on_split_slider_changed)

        self._builder_val_slider = QSlider(Qt.Orientation.Horizontal, split_group)
        self._builder_val_slider.setRange(5, 40)
        self._builder_val_slider.setValue(20)
        self._builder_val_slider.valueChanged.connect(self._on_split_slider_changed)

        self._builder_split_label = QLabel("Train 70% / Val 20% / Test 10%", split_group)
        self._builder_split_label.setObjectName("metricValue")

        split_layout.addWidget(QLabel("Train", split_group), 0, 0)
        split_layout.addWidget(self._builder_train_slider, 0, 1)
        split_layout.addWidget(QLabel("Val", split_group), 1, 0)
        split_layout.addWidget(self._builder_val_slider, 1, 1)
        split_layout.addWidget(self._builder_split_label, 2, 0, 1, 2)
        split_group.setLayout(split_layout)

        meta_group = QGroupBox("Builder Metadata", right_container)
        meta_layout = QFormLayout()

        self._builder_name_input = QLineEdit("", meta_group)
        self._builder_tags_input = QLineEdit("", meta_group)
        self._builder_description_input = QTextEdit(meta_group)
        self._builder_description_input.setMinimumHeight(70)

        self._builder_yaml_display = QLineEdit("", meta_group)
        self._builder_yaml_display.setObjectName("codeField")
        self._builder_yaml_display.setReadOnly(True)

        yaml_buttons = QWidget(meta_group)
        yaml_buttons_layout = QHBoxLayout()
        yaml_buttons_layout.setContentsMargins(0, 0, 0, 0)
        yaml_buttons_layout.setSpacing(6)

        gen_yaml_button = QPushButton("Generate data.yaml", yaml_buttons)
        gen_yaml_button.clicked.connect(self._generate_builder_yaml)

        save_button = QPushButton("Save to Dataset Library", yaml_buttons)
        save_button.setProperty("secondary", True)
        save_button.clicked.connect(self._save_builder_dataset)

        yaml_buttons_layout.addWidget(gen_yaml_button)
        yaml_buttons_layout.addWidget(save_button)
        yaml_buttons_layout.addStretch(1)
        yaml_buttons.setLayout(yaml_buttons_layout)

        meta_layout.addRow("Name", self._builder_name_input)
        meta_layout.addRow("Tags", self._builder_tags_input)
        meta_layout.addRow("Description", self._builder_description_input)
        meta_layout.addRow("Generated data.yaml", self._builder_yaml_display)
        meta_layout.addRow(yaml_buttons)
        meta_group.setLayout(meta_layout)

        right_layout.addWidget(preview_group)
        right_layout.addWidget(class_group)
        right_layout.addWidget(split_group)
        right_layout.addWidget(meta_group)
        right_layout.addStretch(1)
        right_container.setLayout(right_layout)

        content_splitter.addWidget(left_group)
        content_splitter.addWidget(right_container)
        content_splitter.setSizes([320, 640])

        layout.addWidget(top_row)
        layout.addWidget(content_splitter)
        panel.setLayout(layout)

        return panel

    def _build_saved_models_panel(self) -> QWidget:
        """Create saved-model listing and action controls."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QWidget(panel)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        refresh_button = QPushButton("Refresh", controls)
        refresh_button.setProperty("secondary", True)
        refresh_button.clicked.connect(self.refresh_saved_models)

        export_button = QPushButton("Export Weights", controls)
        export_button.clicked.connect(self._export_selected_weights)

        push_button = QPushButton("Push to Device", controls)
        push_button.clicked.connect(self._push_selected_weights)

        delete_button = QPushButton("Delete", controls)
        delete_button.setProperty("danger", True)
        delete_button.clicked.connect(self._delete_selected_saved_run)

        controls_layout.addWidget(refresh_button)
        controls_layout.addWidget(export_button)
        controls_layout.addWidget(push_button)
        controls_layout.addWidget(delete_button)
        controls_layout.addStretch(1)
        controls.setLayout(controls_layout)

        self._saved_models_table = QTableWidget(0, 7, panel)
        self._saved_models_table.setHorizontalHeaderLabels(
            ["Name", "Architecture", "Dataset", "mAP50", "mAP50_95", "Date Saved", "Notes"]
        )
        self._saved_models_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._saved_models_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._saved_models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._saved_models_table.setAlternatingRowColors(True)
        self._saved_models_table.verticalHeader().setVisible(False)

        layout.addWidget(controls)
        layout.addWidget(self._saved_models_table)
        panel.setLayout(layout)

        return panel

    def _add_dataset_row(self, row: DatasetRow) -> None:
        """Append a dataset table row and cache the row payload.

        Args:
            row: Dataset row projection object.
        """

        self._dataset_rows[row.id] = row

        index = self._dataset_table.rowCount()
        self._dataset_table.insertRow(index)

        values = [
            row.name,
            row.source,
            str(row.num_classes),
            str(row.num_images),
            row.created_at,
            row.tags,
        ]

        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, row.id)
            self._dataset_table.setItem(index, col, item)

        self._dataset_table.resizeColumnsToContents()

    def _selected_dataset_id(self) -> int | None:
        """Return selected dataset ID from the library table.

        Returns:
            int | None: Selected dataset primary key.
        """

        selected = self._dataset_table.selectedItems()
        if not selected:
            return None
        data = selected[0].data(Qt.ItemDataRole.UserRole)
        return int(data) if data is not None else None

    def _selected_saved_run_id(self) -> int | None:
        """Return selected TrainingRun ID from saved-models table.

        Returns:
            int | None: Selected run ID.
        """

        selected = self._saved_models_table.selectedItems()
        if not selected:
            return None
        data = selected[0].data(Qt.ItemDataRole.UserRole)
        return int(data) if data is not None else None

    def _update_dataset_detail(self) -> None:
        """Refresh detail panel based on selected dataset row."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            self._dataset_detail.setPlainText("No dataset selected.")
            return

        row = self._dataset_rows.get(dataset_id)
        if row is None:
            self._dataset_detail.setPlainText("Dataset metadata unavailable.")
            return

        details = (
            f"ID: {row.id}\n"
            f"Name: {row.name}\n"
            f"Source: {row.source}\n"
            f"Path: {row.local_path}\n"
            f"Images: {row.num_images}\n"
            f"Classes: {row.num_classes}\n"
            f"Class Names: {', '.join(row.class_names) if row.class_names else '-'}\n"
            f"Tags: {row.tags or '-'}\n"
            f"Created: {row.created_at}\n\n"
            f"Description:\n{row.description or '-'}"
        )
        self._dataset_detail.setPlainText(details)

    def _open_dataset_context_menu(self, position: Any) -> None:
        """Open context menu for dataset-row actions.

        Args:
            position: Table-local click position.
        """

        clicked_item = self._dataset_table.itemAt(position)
        if clicked_item is not None:
            self._dataset_table.selectRow(clicked_item.row())

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return

        menu = QMenu(self)

        edit_action = QAction("Edit", menu)
        edit_action.triggered.connect(self._edit_selected_dataset)
        menu.addAction(edit_action)

        delete_action = QAction("Delete", menu)
        delete_action.triggered.connect(self._delete_selected_dataset)
        menu.addAction(delete_action)

        export_action = QAction("Export as ZIP", menu)
        export_action.triggered.connect(self._export_selected_dataset_zip)
        menu.addAction(export_action)

        open_action = QAction("View in Explorer", menu)
        open_action.triggered.connect(self._open_selected_dataset_folder)
        menu.addAction(open_action)

        menu.exec(self._dataset_table.viewport().mapToGlobal(position))

    def _create_manual_dataset(self) -> None:
        """Open dialog and create a new manual dataset row in SQLite."""

        dialog = DatasetFormDialog("New Dataset", self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        name = values["name"]
        local_path = values["local_path"]

        if not name:
            QMessageBox.warning(self, "Validation", "Dataset name is required.")
            return

        if not local_path:
            QMessageBox.warning(self, "Validation", "Local dataset path is required.")
            return

        dataset_path = Path(local_path)
        if not dataset_path.exists():
            QMessageBox.warning(self, "Validation", "Local path does not exist.")
            return

        class_names = _load_class_names_from_yaml(dataset_path / "data.yaml")
        num_images = _count_images(dataset_path)
        num_classes = len(class_names)

        session = get_session()
        try:
            record = Dataset(
                name=name,
                description=values["description"] or None,
                source=_coerce_dataset_source(values["source"]),
                local_path=str(dataset_path.resolve()),
                class_names=class_names,
                num_images=num_images,
                num_classes=num_classes,
                tags=values["tags"],
                project_id=self._active_project_id,
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Create Failed", f"Could not create dataset: {exc}")
            return
        finally:
            session.close()

        self.refresh_datasets()

    def _edit_selected_dataset(self) -> None:
        """Edit selected dataset metadata."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return

        row = self._dataset_rows.get(dataset_id)
        if row is None:
            return

        dialog = DatasetFormDialog(
            "Edit Dataset",
            self,
            initial={
                "name": row.name,
                "source": row.source,
                "local_path": row.local_path,
                "description": row.description,
                "tags": row.tags,
            },
            allow_source_change=True,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        if not values["name"]:
            QMessageBox.warning(self, "Validation", "Dataset name is required.")
            return

        session = get_session()
        try:
            record = session.get(Dataset, dataset_id)
            if record is None:
                QMessageBox.warning(self, "Missing", "Dataset no longer exists.")
                return

            record.name = values["name"]
            record.description = values["description"] or None
            record.source = _coerce_dataset_source(values["source"])
            record.local_path = values["local_path"]
            record.tags = values["tags"]

            path = Path(record.local_path)
            class_names = _load_class_names_from_yaml(path / "data.yaml")
            if class_names:
                record.class_names = class_names
                record.num_classes = len(class_names)

            record.num_images = _count_images(path) if path.exists() else 0

            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Update Failed", f"Could not update dataset: {exc}")
            return
        finally:
            session.close()

        self.refresh_datasets()

    def _delete_selected_dataset(self) -> None:
        """Delete selected dataset row with confirmation."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return

        row = self._dataset_rows.get(dataset_id)
        if row is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete Dataset",
            f"Delete dataset '{row.name}' from the library?\n"
            "This removes only the database record, not files on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        session = get_session()
        try:
            record = session.get(Dataset, dataset_id)
            if record is None:
                return
            session.delete(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Delete Failed", f"Could not delete dataset: {exc}")
            return
        finally:
            session.close()

        self.refresh_datasets()

    def _import_dataset_folder(self) -> None:
        """Register an existing YOLO dataset folder in the library."""

        selected = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset Folder")
        if not selected:
            return

        folder = Path(selected).resolve()
        if not folder.exists():
            QMessageBox.warning(self, "Import", "Selected folder does not exist.")
            return

        class_names = _load_class_names_from_yaml(folder / "data.yaml")
        if not class_names:
            class_names = _infer_class_names_from_labels(folder)

        dataset_name = folder.name
        num_images = _count_images(folder)

        session = get_session()
        try:
            record = Dataset(
                name=dataset_name,
                description=f"Imported from folder {folder}",
                source=DatasetSource.MANUAL,
                local_path=str(folder),
                class_names=class_names,
                num_images=num_images,
                num_classes=len(class_names),
                tags=["imported"],
                project_id=self._active_project_id,
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Import Failed", f"Could not import folder: {exc}")
            return
        finally:
            session.close()

        self.refresh_datasets()

    def _open_selected_dataset_folder(self) -> None:
        """Open selected dataset folder in the system file explorer."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return

        row = self._dataset_rows.get(dataset_id)
        if row is None:
            return

        target = Path(row.local_path)
        if not target.exists():
            QMessageBox.warning(self, "Path Missing", "Dataset folder does not exist on disk.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _export_selected_dataset_zip(self) -> None:
        """Export selected dataset folder into a ZIP archive."""

        dataset_id = self._selected_dataset_id()
        if dataset_id is None:
            return

        row = self._dataset_rows.get(dataset_id)
        if row is None:
            return

        source_path = Path(row.local_path)
        if not source_path.exists():
            QMessageBox.warning(self, "Export", "Dataset path is missing.")
            return

        default_name = f"{_slugify(row.name)}.zip"
        destination, _ = QFileDialog.getSaveFileName(self, "Export Dataset ZIP", default_name, "ZIP (*.zip)")
        if not destination:
            return

        zip_path = Path(destination)
        if zip_path.suffix.lower() != ".zip":
            zip_path = zip_path.with_suffix(".zip")

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file_path in source_path.rglob("*"):
                    if file_path.is_file():
                        archive.write(file_path, arcname=file_path.relative_to(source_path))
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not export dataset: {exc}")
            return

        QMessageBox.information(self, "Export Complete", f"Exported ZIP to:\n{zip_path}")

    def _add_builder_images(self) -> None:
        """Open file picker and append selected builder images."""

        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images",
            str(PROJECT_ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not selected:
            return

        self._add_builder_images_from_paths(selected)

    def _add_builder_images_from_paths(self, paths: Iterable[str]) -> None:
        """Append one or more image paths to the builder list.

        Args:
            paths: Iterable image paths.
        """

        for path in paths:
            resolved = str(Path(path).resolve())
            if not _is_image_file(Path(resolved)):
                continue
            if resolved in self._builder_images:
                continue

            self._builder_images.append(resolved)
            item = QListWidgetItem(Path(resolved).name)
            item.setData(Qt.ItemDataRole.UserRole, resolved)
            self._builder_image_list.addItem(item)

        if self._builder_image_list.count() > 0 and not self._builder_image_list.selectedItems():
            self._builder_image_list.setCurrentRow(0)

        self._auto_fill_builder_name_if_needed()

    def _clear_builder_images(self) -> None:
        """Clear all builder images and preview state."""

        self._builder_images.clear()
        self._builder_image_list.clear()
        self._builder_preview_label.setText("Select an image to preview annotations.")
        self._builder_preview_label.setPixmap(QPixmap())
        self._builder_preview_meta_label.setText("-")

    def _browse_labeling_root(self) -> None:
        """Select dataset root folder for labeling helpers."""

        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root",
            str(self._project_root),
        )
        if not selected:
            return

        self._labeling_root_input.setText(str(Path(selected).resolve()))
        self._scan_labeling_root()

    def _scan_labeling_root(self) -> None:
        """Scan selected dataset folder and update labeling summary."""

        root = Path(self._labeling_root_input.text().strip()) if self._labeling_root_input else None
        if root is None or not root.exists():
            self._labeling_status_label.setText("Dataset folder not found.")
            self._labeling_status_label.setProperty("role", "warning")
            self._refresh_label_style(self._labeling_status_label)
            return

        info = self._detect_labeling_structure(root)
        images_root = info["images_root"]
        splits = info["splits"]
        total_images = info["total_images"]

        split_text = ", ".join(splits) if splits else "none"
        self._labeling_status_label.setText(
            f"Images: {total_images} | Images Root: {images_root} | Splits: {split_text}"
        )
        self._labeling_status_label.setProperty("role", "")
        self._refresh_label_style(self._labeling_status_label)

    def _create_label_folders(self) -> None:
        """Create labels folder (and split subfolders when detected)."""

        root = Path(self._labeling_root_input.text().strip()) if self._labeling_root_input else None
        if root is None or not root.exists():
            QMessageBox.warning(self, "Labeling", "Select a valid dataset folder first.")
            return

        info = self._detect_labeling_structure(root)
        labels_root = root / "labels"
        try:
            labels_root.mkdir(parents=True, exist_ok=True)
            for split in info["splits"]:
                (labels_root / split).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Labeling", f"Failed to create label folders: {exc}")
            return

        self._scan_labeling_root()
        QMessageBox.information(
            self,
            "Labeling",
            f"Created labels folder at {labels_root}",
        )

    def _create_empty_label_files(self) -> None:
        """Create empty label .txt files for every image."""

        root = Path(self._labeling_root_input.text().strip()) if self._labeling_root_input else None
        if root is None or not root.exists():
            QMessageBox.warning(self, "Labeling", "Select a valid dataset folder first.")
            return

        info = self._detect_labeling_structure(root)
        images_root = Path(info["images_root"])
        labels_root = root / "labels"
        if not labels_root.exists():
            labels_root.mkdir(parents=True, exist_ok=True)

        created = 0
        for image_path in images_root.rglob("*"):
            if not _is_image_file(image_path):
                continue
            rel_path = image_path.relative_to(images_root)
            label_path = labels_root / rel_path.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not label_path.exists():
                label_path.write_text("", encoding="utf-8")
                created += 1

        self._scan_labeling_root()
        QMessageBox.information(
            self,
            "Labeling",
            f"Created {created} empty label files in {labels_root}",
        )

    def _open_labels_folder(self) -> None:
        """Open the labels folder in the file explorer."""

        root = Path(self._labeling_root_input.text().strip()) if self._labeling_root_input else None
        if root is None or not root.exists():
            QMessageBox.warning(self, "Labeling", "Select a valid dataset folder first.")
            return

        labels_root = root / "labels"
        if not labels_root.exists():
            QMessageBox.warning(self, "Labeling", "Labels folder does not exist yet.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(labels_root)))

    def _browse_labeling_tool(self) -> None:
        """Select an annotation tool executable."""

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Annotation Tool",
            str(self._project_root),
        )
        if not selected:
            return

        self._labeling_tool_input.setText(str(Path(selected).resolve()))

    def _launch_labeling_tool(self) -> None:
        """Launch configured annotation tool with the dataset root."""

        root = Path(self._labeling_root_input.text().strip()) if self._labeling_root_input else None
        if root is None or not root.exists():
            QMessageBox.warning(self, "Labeling", "Select a valid dataset folder first.")
            return

        tool_path = Path(self._labeling_tool_input.text().strip()) if self._labeling_tool_input else None
        if tool_path is None or not tool_path.exists():
            QMessageBox.warning(self, "Labeling", "Configure a valid annotation tool path first.")
            return

        try:
            subprocess.Popen([str(tool_path), str(root)])
        except Exception as exc:
            QMessageBox.critical(self, "Labeling", f"Failed to launch tool: {exc}")

    def _update_builder_preview(self) -> None:
        """Render selected image with YOLO annotation overlays."""

        selected = self._builder_image_list.selectedItems()
        if not selected:
            self._builder_preview_label.setText("Select an image to preview annotations.")
            self._builder_preview_meta_label.setText("-")
            return

        image_path = Path(selected[0].data(Qt.ItemDataRole.UserRole))
        if not image_path.exists():
            self._builder_preview_label.setText("Image path no longer exists.")
            return

        class_names = [self._builder_class_list.item(i).text() for i in range(self._builder_class_list.count())]
        boxes = _read_yolo_labels_for_image(image_path)

        pixmap = _render_preview_pixmap(image_path, boxes, class_names)
        if pixmap is None:
            self._builder_preview_label.setText("Unable to render image preview.")
            return

        scaled = pixmap.scaled(
            self._builder_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._builder_preview_label.setPixmap(scaled)

        self._builder_preview_meta_label.setText(
            f"{image_path.name} | Boxes: {len(boxes)} | Size: {pixmap.width()}x{pixmap.height()}"
        )

    def resizeEvent(self, event: Any) -> None:
        """Refresh preview scaling when panel size changes.

        Args:
            event: Resize event.
        """

        super().resizeEvent(event)
        self._update_builder_preview()

    def _add_builder_class(self) -> None:
        """Append one class label to builder class manager."""

        name, ok = QInputDialog.getText(self, "Add Class", "Class label:")
        name = name.strip()
        if not ok or not name:
            return

        if any(self._builder_class_list.item(i).text() == name for i in range(self._builder_class_list.count())):
            QMessageBox.information(self, "Duplicate", "Class label already exists.")
            return

        self._builder_class_list.addItem(name)

    def _remove_builder_class(self) -> None:
        """Remove selected class label from builder class manager."""

        row = self._builder_class_list.currentRow()
        if row < 0:
            return
        self._builder_class_list.takeItem(row)

    def _rename_builder_class(self) -> None:
        """Rename selected class label in builder class manager."""

        item = self._builder_class_list.currentItem()
        if item is None:
            return

        new_name, ok = QInputDialog.getText(self, "Rename Class", "Class label:", text=item.text())
        new_name = new_name.strip()
        if not ok or not new_name:
            return

        item.setText(new_name)

    def _on_split_slider_changed(self) -> None:
        """Maintain valid train/val/test percentages and update label."""

        train = self._builder_train_slider.value()
        val = self._builder_val_slider.value()
        if train + val > 95:
            val = 95 - train
            self._builder_val_slider.blockSignals(True)
            self._builder_val_slider.setValue(max(5, val))
            self._builder_val_slider.blockSignals(False)
            val = self._builder_val_slider.value()

        test = max(0, 100 - train - val)
        self._builder_split_label.setText(f"Train {train}% / Val {val}% / Test {test}%")

    def _generate_builder_yaml(self) -> None:
        """Generate preview `data.yaml` from builder settings."""

        class_names = [self._builder_class_list.item(i).text() for i in range(self._builder_class_list.count())]
        if not class_names:
            QMessageBox.warning(self, "Generate YAML", "Add class labels before generating data.yaml.")
            return

        dataset_name = self._builder_name_input.text().strip() or "dataset"
        slug = _slugify(dataset_name)
        self._builder_preview_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        yaml_path = self._builder_preview_root / f"{slug}_{timestamp}_data.yaml"

        payload = {
            "path": str((self._dataset_library_root / f"{slug}_{timestamp}").resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_names,
            "nc": len(class_names),
        }

        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

        self._builder_yaml_path = str(yaml_path.resolve())
        self._builder_yaml_display.setText(self._builder_yaml_path)

    def _save_builder_dataset(self) -> None:
        """Persist builder assets to disk and register dataset in SQLite."""

        name = self._builder_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Save Dataset", "Dataset name is required.")
            return

        if not self._builder_images:
            QMessageBox.warning(self, "Save Dataset", "Add at least one image.")
            return

        class_names = [self._builder_class_list.item(i).text() for i in range(self._builder_class_list.count())]
        if not class_names:
            QMessageBox.warning(self, "Save Dataset", "Add class labels before saving.")
            return

        tags = [tag.strip() for tag in self._builder_tags_input.text().split(",") if tag.strip()]
        description = self._builder_description_input.toPlainText().strip()

        self._dataset_library_root.mkdir(parents=True, exist_ok=True)

        slug = _slugify(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_root = self._dataset_library_root / f"{slug}_{timestamp}"

        images_root = dataset_root / "images"
        labels_root = dataset_root / "labels"

        for split in ("train", "val", "test"):
            (images_root / split).mkdir(parents=True, exist_ok=True)
            (labels_root / split).mkdir(parents=True, exist_ok=True)

        assignments = _split_assignments(
            self._builder_images,
            train_percent=self._builder_train_slider.value(),
            val_percent=self._builder_val_slider.value(),
        )

        copied_count = 0
        for split, image_paths in assignments.items():
            for image_str in image_paths:
                image_path = Path(image_str)
                if not image_path.exists():
                    continue

                target_image = images_root / split / image_path.name
                shutil.copy2(image_path, target_image)
                copied_count += 1

                source_label = image_path.with_suffix(".txt")
                target_label = labels_root / split / f"{image_path.stem}.txt"
                if source_label.exists():
                    shutil.copy2(source_label, target_label)
                else:
                    target_label.write_text("", encoding="utf-8")

        yaml_payload = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_names,
            "nc": len(class_names),
        }

        with (dataset_root / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yaml_payload, handle, sort_keys=False)

        session = get_session()
        try:
            record = Dataset(
                name=name,
                description=description or None,
                source=DatasetSource.MANUAL,
                local_path=str(dataset_root.resolve()),
                class_names=class_names,
                num_images=copied_count,
                num_classes=len(class_names),
                tags=tags,
                project_id=self._active_project_id,
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Save Failed", f"Could not save dataset: {exc}")
            return
        finally:
            session.close()

        self._builder_yaml_path = str((dataset_root / "data.yaml").resolve())
        self._builder_yaml_display.setText(self._builder_yaml_path)

        QMessageBox.information(self, "Dataset Saved", f"Saved dataset to:\n{dataset_root}")
        self.refresh_datasets()

    def _auto_fill_builder_name_if_needed(self) -> None:
        """Fill builder name from first image parent folder when empty."""

        if self._builder_name_input.text().strip():
            return

        if not self._builder_images:
            return

        first = Path(self._builder_images[0])
        self._builder_name_input.setText(first.parent.name or "dataset")

    @staticmethod
    def _detect_labeling_structure(root: Path) -> dict[str, Any]:
        """Detect dataset image root and train/val/test split folders."""

        images_root = root / "images" if (root / "images").is_dir() else root
        splits: list[str] = []
        for split in ("train", "val", "test"):
            if (images_root / split).is_dir():
                splits.append(split)

        total_images = _count_images(images_root)
        return {
            "images_root": str(images_root),
            "splits": splits,
            "total_images": total_images,
        }

    def _export_selected_weights(self) -> None:
        """Copy selected saved model weights to a user-selected destination."""

        run_id = self._selected_saved_run_id()
        if run_id is None:
            QMessageBox.information(self, "Saved Models", "Select a saved model first.")
            return

        row = self._saved_run_rows.get(run_id)
        if row is None:
            return

        weights_path = Path(str(row.get("weights_path") or ""))
        if not weights_path.exists():
            QMessageBox.warning(self, "Export", "Weights file is missing.")
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export Weights",
            f"{weights_path.stem}.pt",
            "PyTorch Weights (*.pt)",
        )
        if not destination:
            return

        target = Path(destination)
        if target.suffix.lower() != ".pt":
            target = target.with_suffix(".pt")

        try:
            shutil.copy2(weights_path, target)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not export weights: {exc}")
            return

        QMessageBox.information(self, "Exported", f"Weights exported to:\n{target}")

    def _push_selected_weights(self) -> None:
        """Prompt for device and prepare remote push operation for selected model."""

        run_id = self._selected_saved_run_id()
        if run_id is None:
            QMessageBox.information(self, "Push to Device", "Select a saved model first.")
            return

        row = self._saved_run_rows.get(run_id)
        if row is None:
            return

        session = get_session()
        try:
            devices = session.query(RemoteDevice).order_by(RemoteDevice.name.asc()).all()
        except Exception:
            LOGGER.exception("Failed to load remote devices.")
            devices = []
        finally:
            session.close()

        if not devices:
            QMessageBox.information(
                self,
                "Push to Device",
                "No remote devices are registered yet. Add devices in the Remote tab.",
            )
            return

        labels = [f"{device.name} ({device.host}:{device.port})" for device in devices]
        selected_label, ok = QInputDialog.getItem(
            self,
            "Select Device",
            "Remote Device:",
            labels,
            0,
            False,
        )
        if not ok:
            return

        device = devices[labels.index(selected_label)]

        QMessageBox.information(
            self,
            "Push Queued",
            "Push request prepared.\n"
            f"Model: {row.get('name')}\n"
            f"Device: {device.name} ({device.host}:{device.port})\n\n"
            "Remote transfer wiring will be completed with remote_manager integration.",
        )

    def _delete_selected_saved_run(self) -> None:
        """Remove saved-model flag (and optional copied weights) from a run."""

        run_id = self._selected_saved_run_id()
        if run_id is None:
            QMessageBox.information(self, "Saved Models", "Select a saved model first.")
            return

        row = self._saved_run_rows.get(run_id)
        if row is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete Saved Model",
            f"Remove '{row.get('name')}' from Saved Models?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        delete_weights = QMessageBox.question(
            self,
            "Delete Weights File",
            "Also delete the copied weights file from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        session = get_session()
        try:
            run = session.get(TrainingRun, run_id)
            if run is None:
                return
            run.is_saved = False
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Delete Failed", f"Could not update saved model: {exc}")
            return
        finally:
            session.close()

        if delete_weights == QMessageBox.StandardButton.Yes:
            weights_path = Path(str(row.get("weights_path") or ""))
            if weights_path.exists():
                try:
                    weights_path.unlink()
                except Exception:
                    LOGGER.exception("Unable to delete weights file: %s", weights_path)

        self.refresh_saved_models()


def _is_image_file(path: Path) -> bool:
    """Return whether path points to a supported image file.

    Args:
        path: Candidate filesystem path.

    Returns:
        bool: True when path extension is a supported image type.
    """

    return path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()


def _collect_images_from_directory(directory: Path) -> list[str]:
    """Collect supported image files recursively from a directory.

    Args:
        directory: Source folder.

    Returns:
        list[str]: Absolute image paths.
    """

    images: list[str] = []
    for candidate in directory.rglob("*"):
        if _is_image_file(candidate):
            images.append(str(candidate.resolve()))
    return images


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate while preserving first-seen order.

    Args:
        items: Sequence of string values.

    Returns:
        list[str]: Deduplicated list preserving insertion order.
    """

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _slugify(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug.

    Args:
        text: Raw text.

    Returns:
        str: Slug value.
    """

    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "dataset"


def _fmt_metric(value: float | None) -> str:
    """Format optional metric floats for table display.

    Args:
        value: Metric value or None.

    Returns:
        str: Formatted metric string.
    """

    if value is None:
        return "-"
    return f"{value:.4f}"


def _coerce_dataset_source(raw_source: str) -> DatasetSource:
    """Normalize user-provided source string into `DatasetSource` enum.

    Args:
        raw_source: Raw source string.

    Returns:
        DatasetSource: Normalized enum value.
    """

    value = (raw_source or "").strip().lower()
    for source in DatasetSource:
        if source.value == value:
            return source
    return DatasetSource.MANUAL


def _load_class_names_from_yaml(yaml_path: Path) -> list[str]:
    """Load class names list from a YOLO `data.yaml` file.

    Args:
        yaml_path: YAML file path.

    Returns:
        list[str]: Parsed class names.
    """

    if not yaml_path.exists():
        return []

    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        LOGGER.exception("Could not parse YAML: %s", yaml_path)
        return []

    names = payload.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        return [
            str(names[key])
            for key in sorted(
                names.keys(),
                key=lambda raw: int(raw) if str(raw).isdigit() else str(raw),
            )
        ]

    return []


def _infer_class_names_from_labels(dataset_root: Path) -> list[str]:
    """Infer class names from label IDs when yaml names are missing.

    Args:
        dataset_root: Dataset root directory.

    Returns:
        list[str]: Synthetic class names, e.g. `class_0`.
    """

    max_class_id = -1
    for label_path in dataset_root.rglob("*.txt"):
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id = int(float(line.split()[0]))
                max_class_id = max(max_class_id, class_id)
        except Exception:
            continue

    if max_class_id < 0:
        return []

    return [f"class_{idx}" for idx in range(max_class_id + 1)]


def _count_images(root: Path) -> int:
    """Count supported image files recursively in a folder.

    Args:
        root: Root directory.

    Returns:
        int: Number of images.
    """

    if not root.exists():
        return 0

    count = 0
    for candidate in root.rglob("*"):
        if _is_image_file(candidate):
            count += 1
    return count


def _split_assignments(images: list[str], train_percent: int, val_percent: int) -> dict[str, list[str]]:
    """Split image paths into train/val/test partitions.

    Args:
        images: Source image paths.
        train_percent: Train split percentage.
        val_percent: Validation split percentage.

    Returns:
        dict[str, list[str]]: Mapping with train/val/test image lists.
    """

    shuffled = list(images)
    random.Random(42).shuffle(shuffled)

    total = len(shuffled)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    train_count = int(total * (train_percent / 100.0))
    val_count = int(total * (val_percent / 100.0))

    if train_count <= 0:
        train_count = 1

    if train_count >= total:
        train_count = total
        val_count = 0

    # Guarantee at least one sample in val/test when data volume permits.
    if total >= 3:
        train_count = max(1, min(train_count, total - 2))
        val_count = max(1, min(val_count, total - train_count - 1))

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]

    return {"train": train, "val": val, "test": test}


def _read_yolo_labels_for_image(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO-format boxes from paired label file for an image.

    Args:
        image_path: Image file path.

    Returns:
        list[tuple[int, float, float, float, float]]: Parsed boxes.
    """

    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return []

    boxes: list[tuple[int, float, float, float, float]] = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            boxes.append((class_id, xc, yc, bw, bh))
    except Exception:
        LOGGER.exception("Failed parsing label file: %s", label_path)
        return []

    return boxes


def _render_preview_pixmap(
    image_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: list[str],
) -> QPixmap | None:
    """Render an image preview pixmap with optional annotation overlays.

    Args:
        image_path: Source image.
        boxes: YOLO label tuples.
        class_names: Available class names by index.

    Returns:
        QPixmap | None: Rendered pixmap, or None if rendering fails.
    """

    if cv2 is None:
        pixmap = QPixmap(str(image_path))
        return pixmap if not pixmap.isNull() else None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]

    for class_id, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2.0) * width)
        y1 = int((yc - bh / 2.0) * height)
        x2 = int((xc + bw / 2.0) * width)
        y2 = int((yc + bh / 2.0) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 212, 170), 2)

        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        cv2.putText(
            image,
            class_name,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (108, 99, 255),
            1,
            cv2.LINE_AA,
        )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


__all__ = ["DatasetTab"]

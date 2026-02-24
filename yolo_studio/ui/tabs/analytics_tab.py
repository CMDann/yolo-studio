"""Analytics tab for cross-run historical analysis and comparison."""

from __future__ import annotations

import csv
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph import exporters
from PyQt6.QtCore import QDate, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.models.database import Dataset, TrainingRun, TrainingRunStatus, get_session
from sqlalchemy.orm import joinedload

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RunInfo:
    id: int
    name: str
    architecture: str
    dataset_name: str
    dataset_id: int
    created_at: datetime
    completed_at: datetime | None
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    best_map50: float | None
    best_map50_95: float | None
    precision: float | None
    recall: float | None
    status: str
    output_dir: str | None
    config_yaml: dict[str, Any]
    is_saved: bool


class AnalyticsTab(QWidget):
    """Tab for historical analytics and experiment comparison."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._active_project_id: int | None = None
        self._project_root = PROJECT_ROOT
        self._eval_snapshot_root = self._project_root / "output" / "evaluations"
        self._runs: dict[int, RunInfo] = {}
        self._datasets: dict[int, Dataset] = {}

        self._search_input: QLineEdit
        self._from_date: QDateEdit
        self._to_date: QDateEdit
        self._architecture_combo: QComboBox
        self._dataset_combo: QComboBox
        self._refresh_button: QPushButton

        self._history_plot: pg.PlotWidget
        self._history_points: list[pg.ScatterPlotItem] = []
        self._run_detail_label: QLabel

        self._comparison_list: QListWidget
        self._comparison_table: QTableWidget
        self._comparison_export_button: QPushButton

        self._loss_list: QListWidget
        self._loss_plot: pg.PlotWidget

        self._heatmap_plot: pg.PlotWidget

        self._dataset_stats_combo: QComboBox
        self._dataset_image_label: QLabel
        self._dataset_class_plot: pg.PlotWidget
        self._dataset_density_plot: pg.PlotWidget

        self._leaderboard_table: QTableWidget

        self._export_dashboard_button: QPushButton

        self._build_ui()
        self.refresh_data()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        self._project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._eval_snapshot_root = self._project_root / "output" / "evaluations"
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(6, 6, 6, 6)
        filter_layout.setSpacing(6)

        self._search_input = QLineEdit(filter_group)
        self._search_input.setPlaceholderText("Search by run name or notes")

        self._from_date = QDateEdit(filter_group)
        self._from_date.setCalendarPopup(True)
        self._from_date.setDate(QDate.currentDate().addMonths(-3))

        self._to_date = QDateEdit(filter_group)
        self._to_date.setCalendarPopup(True)
        self._to_date.setDate(QDate.currentDate())

        self._architecture_combo = QComboBox(filter_group)
        self._architecture_combo.addItem("All Architectures", None)

        self._dataset_combo = QComboBox(filter_group)
        self._dataset_combo.addItem("All Datasets", None)

        self._refresh_button = QPushButton("Apply Filters", filter_group)
        self._refresh_button.clicked.connect(self.refresh_data)

        filter_layout.addWidget(self._search_input, stretch=2)
        filter_layout.addWidget(QLabel("From", filter_group))
        filter_layout.addWidget(self._from_date)
        filter_layout.addWidget(QLabel("To", filter_group))
        filter_layout.addWidget(self._to_date)
        filter_layout.addWidget(self._architecture_combo)
        filter_layout.addWidget(self._dataset_combo)
        filter_layout.addWidget(self._refresh_button)
        filter_group.setLayout(filter_layout)

        content = QWidget(self)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        history_group = self._build_history_group(content)
        compare_group = self._build_comparison_group(content)
        loss_group = self._build_loss_group(content)
        heatmap_group = self._build_heatmap_group(content)
        dataset_group = self._build_dataset_group(content)
        leaderboard_group = self._build_leaderboard_group(content)

        content_layout.addWidget(history_group)
        content_layout.addWidget(compare_group)
        content_layout.addWidget(loss_group)
        content_layout.addWidget(heatmap_group)
        content_layout.addWidget(dataset_group)
        content_layout.addWidget(leaderboard_group)
        content_layout.addStretch(1)
        content.setLayout(content_layout)

        export_row = QWidget(self)
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(6)

        self._export_dashboard_button = QPushButton("Export Dashboard", export_row)
        self._export_dashboard_button.clicked.connect(self._export_dashboard)

        export_layout.addStretch(1)
        export_layout.addWidget(self._export_dashboard_button)
        export_row.setLayout(export_layout)

        layout.addWidget(filter_group)
        layout.addWidget(content, stretch=1)
        layout.addWidget(export_row)
        self.setLayout(layout)

    def _build_history_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Training History", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._history_plot = pg.PlotWidget(group)
        self._history_plot.setBackground("w")
        self._history_plot.showGrid(x=True, y=True, alpha=0.3)
        self._history_plot.setLabel("left", "mAP")
        self._history_plot.setLabel("bottom", "Date")
        self._history_plot.addLegend()
        self._history_legend = self._history_plot.plotItem.legend
        self._history_plot.setMinimumHeight(300)

        self._run_detail_label = QLabel("Select a run to see details.", group)
        self._run_detail_label.setProperty("role", "subtle")
        self._run_detail_label.setWordWrap(True)

        layout.addWidget(self._history_plot, stretch=1)
        layout.addWidget(self._run_detail_label)
        group.setLayout(layout)
        return group

    def _build_comparison_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Run Comparison", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._comparison_list = QListWidget(group)
        self._comparison_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._comparison_list.itemSelectionChanged.connect(self._update_comparison_table)

        self._comparison_table = QTableWidget(group)
        self._comparison_table.setRowCount(0)
        self._comparison_table.setColumnCount(0)
        self._comparison_table.setSortingEnabled(False)
        self._comparison_table.setMinimumHeight(240)

        self._comparison_export_button = QPushButton("Export CSV", group)
        self._comparison_export_button.setProperty("secondary", True)
        self._comparison_export_button.clicked.connect(self._export_comparison_csv)

        layout.addWidget(QLabel("Select runs to compare", group))
        layout.addWidget(self._comparison_list, stretch=1)
        layout.addWidget(self._comparison_table, stretch=2)
        layout.addWidget(self._comparison_export_button, alignment=Qt.AlignmentFlag.AlignRight)
        group.setLayout(layout)
        return group

    def _build_loss_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Loss Curves Overlay", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._loss_list = QListWidget(group)
        self._loss_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._loss_list.itemSelectionChanged.connect(self._update_loss_overlay)

        self._loss_plot = pg.PlotWidget(group)
        self._loss_plot.setBackground("w")
        self._loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self._loss_plot.setLabel("left", "Loss")
        self._loss_plot.setLabel("bottom", "Epoch")
        self._loss_plot.setMinimumHeight(300)

        layout.addWidget(QLabel("Select up to 5 runs", group))
        layout.addWidget(self._loss_list)
        layout.addWidget(self._loss_plot, stretch=1)
        group.setLayout(layout)
        return group

    def _build_heatmap_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Class Performance Heatmap", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._heatmap_plot = pg.PlotWidget(group)
        self._heatmap_plot.setBackground("w")
        self._heatmap_plot.showGrid(x=True, y=True, alpha=0.3)
        self._heatmap_plot.setMinimumHeight(280)

        layout.addWidget(self._heatmap_plot, stretch=1)
        group.setLayout(layout)
        return group

    def _build_dataset_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Dataset Statistics", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._dataset_stats_combo = QComboBox(group)
        self._dataset_stats_combo.currentIndexChanged.connect(self._render_dataset_stats)

        self._dataset_image_label = QLabel("Select a dataset", group)
        self._dataset_image_label.setProperty("role", "subtle")

        self._dataset_class_plot = pg.PlotWidget(group)
        self._dataset_class_plot.setBackground("w")
        self._dataset_class_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dataset_class_plot.setTitle("Class Distribution")
        self._dataset_class_plot.setMinimumHeight(240)

        self._dataset_density_plot = pg.PlotWidget(group)
        self._dataset_density_plot.setBackground("w")
        self._dataset_density_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dataset_density_plot.setTitle("Annotation Density")
        self._dataset_density_plot.setMinimumHeight(240)

        layout.addWidget(self._dataset_stats_combo)
        layout.addWidget(self._dataset_image_label)
        layout.addWidget(self._dataset_class_plot, stretch=1)
        layout.addWidget(self._dataset_density_plot, stretch=1)
        group.setLayout(layout)
        return group

    def _build_leaderboard_group(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Model Leaderboard", parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._leaderboard_table = QTableWidget(group)
        self._leaderboard_table.setSortingEnabled(True)
        self._leaderboard_table.setMinimumHeight(260)

        layout.addWidget(self._leaderboard_table)
        group.setLayout(layout)
        return group

    def refresh_data(self) -> None:
        self._load_runs()
        self._load_datasets()
        self._render_history()
        self._populate_lists()
        self._render_leaderboard()
        self._render_heatmap()
        self._render_dataset_stats()

    def _load_runs(self) -> None:
        self._runs.clear()

        session = get_session()
        try:
            query = (
                session.query(TrainingRun)
                .options(joinedload(TrainingRun.dataset))
                .filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
            )
            if self._active_project_id is not None:
                query = query.filter(TrainingRun.project_id == self._active_project_id)

            search = self._search_input.text().strip().lower()
            if search:
                query = query.filter(TrainingRun.name.ilike(f"%{search}%"))

            from_date = self._from_date.date().toPyDate()
            to_date = self._to_date.date().toPyDate()
            query = query.filter(TrainingRun.created_at >= datetime.combine(from_date, datetime.min.time()))
            query = query.filter(TrainingRun.created_at <= datetime.combine(to_date, datetime.max.time()))

            arch = self._architecture_combo.currentData()
            if arch:
                query = query.filter(TrainingRun.model_architecture == arch)

            dataset_id = self._dataset_combo.currentData()
            if dataset_id:
                query = query.filter(TrainingRun.dataset_id == int(dataset_id))

            runs = query.order_by(TrainingRun.completed_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load training runs.")
            runs = []
        finally:
            session.close()

        for run in runs:
            dataset_name = run.dataset.name if run.dataset is not None else "-"
            dataset_id = run.dataset.id if run.dataset is not None else -1
            self._runs[run.id] = RunInfo(
                id=run.id,
                name=run.name,
                architecture=run.model_architecture,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                created_at=run.created_at,
                completed_at=run.completed_at,
                epochs=run.epochs,
                batch_size=run.batch_size,
                image_size=run.image_size,
                learning_rate=run.learning_rate,
                best_map50=run.best_map50,
                best_map50_95=run.best_map50_95,
                precision=None,
                recall=None,
                status=run.status.value,
                output_dir=run.output_dir,
                config_yaml=run.config_yaml or {},
                is_saved=bool(run.is_saved),
            )

        self._populate_filter_options()

    def _load_datasets(self) -> None:
        self._datasets.clear()
        session = get_session()
        try:
            query = session.query(Dataset)
            if self._active_project_id is not None:
                query = query.filter(Dataset.project_id == self._active_project_id)
            datasets = query.order_by(Dataset.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load datasets.")
            datasets = []
        finally:
            session.close()

        self._datasets = {dataset.id: dataset for dataset in datasets}

    def _populate_filter_options(self) -> None:
        current_arch = self._architecture_combo.currentData()
        current_dataset = self._dataset_combo.currentData()

        architectures = sorted({run.architecture for run in self._runs.values()})
        self._architecture_combo.blockSignals(True)
        self._architecture_combo.clear()
        self._architecture_combo.addItem("All Architectures", None)
        for arch in architectures:
            self._architecture_combo.addItem(arch, arch)
        if current_arch in architectures:
            index = self._architecture_combo.findData(current_arch)
            if index >= 0:
                self._architecture_combo.setCurrentIndex(index)
        self._architecture_combo.blockSignals(False)

        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        self._dataset_combo.addItem("All Datasets", None)
        for dataset in self._datasets.values():
            self._dataset_combo.addItem(dataset.name, dataset.id)
        if current_dataset is not None:
            index = self._dataset_combo.findData(current_dataset)
            if index >= 0:
                self._dataset_combo.setCurrentIndex(index)
        self._dataset_combo.blockSignals(False)

        self._dataset_stats_combo.blockSignals(True)
        self._dataset_stats_combo.clear()
        for dataset in self._datasets.values():
            self._dataset_stats_combo.addItem(dataset.name, dataset.id)
        self._dataset_stats_combo.blockSignals(False)

    def _render_history(self) -> None:
        self._history_plot.clear()
        self._history_points = []

        if not self._runs:
            return

        runs_sorted = sorted(self._runs.values(), key=lambda r: r.created_at)
        x_values = np.arange(len(runs_sorted))

        map50 = [run.best_map50 if run.best_map50 is not None else 0.0 for run in runs_sorted]
        map95 = [run.best_map50_95 if run.best_map50_95 is not None else 0.0 for run in runs_sorted]

        spots_50 = [
            {"pos": (x_values[idx], map50[idx]), "data": run.id}
            for idx, run in enumerate(runs_sorted)
        ]
        spots_95 = [
            {"pos": (x_values[idx], map95[idx]), "data": run.id}
            for idx, run in enumerate(runs_sorted)
        ]

        scatter50 = pg.ScatterPlotItem(
            spots=spots_50,
            symbol="o",
            size=10,
            brush=pg.mkBrush(0, 123, 255, 200),
        )
        scatter95 = pg.ScatterPlotItem(
            spots=spots_95,
            symbol="t",
            size=10,
            brush=pg.mkBrush(255, 99, 71, 200),
        )

        scatter50.sigClicked.connect(self._on_history_point_clicked)
        scatter95.sigClicked.connect(self._on_history_point_clicked)

        self._history_plot.addItem(scatter50)
        self._history_plot.addItem(scatter95)
        if self._history_legend is not None:
            self._history_legend.addItem(scatter50, "mAP50")
            self._history_legend.addItem(scatter95, "mAP50-95")

        self._history_points = [scatter50, scatter95]

        labels = [
            (
                x_values[idx],
                (run.completed_at or run.created_at).strftime("%Y-%m-%d"),
            )
            for idx, run in enumerate(runs_sorted)
        ]
        axis = self._history_plot.getAxis("bottom")
        axis.setTicks([labels])

    def _on_history_point_clicked(self, plot: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        if not points:
            return
        run_id = points[0].data()
        if run_id is None:
            return
        run = self._runs.get(run_id)
        if run is None:
            return
        self._run_detail_label.setText(
            "\n".join(
                [
                    f"Run: {run.name}",
                    f"Architecture: {run.architecture}",
                    f"Dataset: {run.dataset_name}",
                    f"mAP50: {run.best_map50 or 0:.3f}",
                    f"mAP50-95: {run.best_map50_95 or 0:.3f}",
                    f"Epochs: {run.epochs} | Batch: {run.batch_size} | LR: {run.learning_rate}",
                ]
            )
        )

    def _populate_lists(self) -> None:
        self._comparison_list.clear()
        self._loss_list.clear()

        for run in sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True):
            label = f"{run.name} ({run.architecture})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, run.id)
            self._comparison_list.addItem(item)

            loss_item = QListWidgetItem(label)
            loss_item.setData(Qt.ItemDataRole.UserRole, run.id)
            self._loss_list.addItem(loss_item)

    def _update_comparison_table(self) -> None:
        selected = [item.data(Qt.ItemDataRole.UserRole) for item in self._comparison_list.selectedItems()]
        runs = [self._runs[run_id] for run_id in selected if run_id in self._runs]
        if not runs:
            self._comparison_table.setRowCount(0)
            self._comparison_table.setColumnCount(0)
            return

        fields = [
            "epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
            "augmentation",
            "best_map50",
            "best_map50_95",
        ]

        self._comparison_table.setRowCount(len(fields))
        self._comparison_table.setColumnCount(len(runs))
        self._comparison_table.setHorizontalHeaderLabels([run.name for run in runs])
        self._comparison_table.setVerticalHeaderLabels([field.replace("_", " ") for field in fields])

        for col, run in enumerate(runs):
            for row, field in enumerate(fields):
                value = self._extract_run_field(run, field)
                item = QTableWidgetItem(value)
                self._comparison_table.setItem(row, col, item)

    def _extract_run_field(self, run: RunInfo, field: str) -> str:
        if field == "optimizer":
            return str(run.config_yaml.get("optimizer") or "-")
        if field == "augmentation":
            aug = run.config_yaml.get("augmentation") or {}
            enabled = [name for name, val in aug.items() if val]
            return ", ".join(enabled) if enabled else "-"
        value = getattr(run, field, "-")
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _export_comparison_csv(self) -> None:
        if self._comparison_table.columnCount() == 0:
            QMessageBox.information(self, "Export", "Select runs to compare.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Export Comparison CSV", "comparison.csv", "CSV Files (*.csv)")
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                headers = [self._comparison_table.horizontalHeaderItem(i).text() for i in range(self._comparison_table.columnCount())]
                writer.writerow(["Field"] + headers)

                for row in range(self._comparison_table.rowCount()):
                    field = self._comparison_table.verticalHeaderItem(row).text()
                    values = [self._comparison_table.item(row, col).text() if self._comparison_table.item(row, col) else "" for col in range(self._comparison_table.columnCount())]
                    writer.writerow([field] + values)
        except Exception as exc:
            QMessageBox.critical(self, "Export", f"CSV export failed: {exc}")

    def _update_loss_overlay(self) -> None:
        selected = [item.data(Qt.ItemDataRole.UserRole) for item in self._loss_list.selectedItems()]
        runs = [self._runs[run_id] for run_id in selected if run_id in self._runs][:5]

        self._loss_plot.clear()
        if not runs:
            return

        colors = [pg.intColor(i, hues=len(runs)) for i in range(len(runs))]

        for idx, run in enumerate(runs):
            epochs, losses = _load_loss_curve(run.output_dir)
            if not epochs:
                continue
            self._loss_plot.plot(epochs, losses, pen=pg.mkPen(colors[idx], width=2), name=run.name)

    def _render_heatmap(self) -> None:
        self._heatmap_plot.clear()
        data, class_labels, model_labels = _load_class_performance(self._runs, self._eval_snapshot_root)
        if data is None:
            return

        img = pg.ImageItem(data)
        self._heatmap_plot.addItem(img)
        self._heatmap_plot.getAxis("bottom").setTicks([[(idx, label) for idx, label in enumerate(model_labels)]])
        self._heatmap_plot.getAxis("left").setTicks([[(idx, label) for idx, label in enumerate(class_labels)]])
        self._heatmap_plot.setLabel("bottom", "Models")
        self._heatmap_plot.setLabel("left", "Classes")

    def _render_dataset_stats(self) -> None:
        dataset_id = self._dataset_stats_combo.currentData()
        if dataset_id is None:
            return

        dataset = self._datasets.get(int(dataset_id))
        if dataset is None:
            return

        image_count, class_counts, densities = _compute_dataset_stats(Path(dataset.local_path), dataset.class_names)
        self._dataset_image_label.setText(f"Images: {image_count} | Classes: {len(dataset.class_names)}")

        self._dataset_class_plot.clear()
        if class_counts:
            x = np.arange(len(class_counts))
            bar = pg.BarGraphItem(x=x, height=class_counts, width=0.6, brush=pg.mkBrush(0, 123, 255, 180))
            self._dataset_class_plot.addItem(bar)
            self._dataset_class_plot.getAxis("bottom").setTicks([[(idx, name) for idx, name in enumerate(dataset.class_names)]])

        self._dataset_density_plot.clear()
        if densities:
            hist, edges = np.histogram(densities, bins=min(20, max(5, len(densities) // 4)))
            bar = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=pg.mkBrush(34, 197, 94, 180))
            self._dataset_density_plot.addItem(bar)

    def _render_leaderboard(self) -> None:
        runs = sorted(
            [run for run in self._runs.values() if run.is_saved],
            key=lambda r: r.best_map50 or 0,
            reverse=True,
        )
        headers = [
            "Run",
            "Architecture",
            "Parameters",
            "Dataset",
            "mAP50",
            "mAP50-95",
            "Train Time (min)",
        ]

        self._leaderboard_table.setColumnCount(len(headers))
        self._leaderboard_table.setHorizontalHeaderLabels(headers)
        self._leaderboard_table.setRowCount(len(runs))

        for row, run in enumerate(runs):
            values = [
                run.name,
                run.architecture,
                _format_params(run.config_yaml),
                run.dataset_name,
                f"{run.best_map50 or 0:.3f}",
                f"{run.best_map50_95 or 0:.3f}",
                f"{_train_time_minutes(run):.1f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                self._leaderboard_table.setItem(row, col, item)

        self._leaderboard_table.resizeColumnsToContents()

    def _export_dashboard(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export Dashboard", "analytics.zip", "Zip Files (*.zip)")
        if not filename:
            return

        zip_path = Path(filename)
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                _export_plot(zf, self._history_plot, "history.png")
                _export_plot(zf, self._loss_plot, "loss_overlay.png")
                _export_plot(zf, self._heatmap_plot, "class_heatmap.png")
                _export_plot(zf, self._dataset_class_plot, "dataset_class_distribution.png")
                _export_plot(zf, self._dataset_density_plot, "dataset_annotation_density.png")

                _export_table_csv(zf, self._comparison_table, "comparison.csv")
                _export_table_csv(zf, self._leaderboard_table, "leaderboard.csv")

            QMessageBox.information(self, "Export", f"Dashboard exported to:\n{zip_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export", f"Export failed: {exc}")


def _export_plot(zf: zipfile.ZipFile, plot: pg.PlotWidget, name: str) -> None:
    exporter = exporters.ImageExporter(plot.plotItem)
    image = exporter.export(toBytes=True)
    zf.writestr(name, image.getvalue())


def _export_table_csv(zf: zipfile.ZipFile, table: QTableWidget, name: str) -> None:
    if table.rowCount() == 0 or table.columnCount() == 0:
        return

    rows = []
    header = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    rows.append(["Field"] + header if table.verticalHeaderItem(0) else header)

    if table.verticalHeaderItem(0):
        for row in range(table.rowCount()):
            label = table.verticalHeaderItem(row).text()
            values = [table.item(row, col).text() if table.item(row, col) else "" for col in range(table.columnCount())]
            rows.append([label] + values)
    else:
        for row in range(table.rowCount()):
            values = [table.item(row, col).text() if table.item(row, col) else "" for col in range(table.columnCount())]
            rows.append(values)

    output = []
    for row in rows:
        output.append(",".join([_csv_escape(value) for value in row]))

    zf.writestr(name, "\n".join(output))


def _csv_escape(value: str) -> str:
    if "," in value or "\"" in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def _train_time_minutes(run: RunInfo) -> float:
    if run.completed_at is None:
        return 0.0
    delta = run.completed_at - run.created_at
    return delta.total_seconds() / 60.0


def _format_params(config: dict[str, Any]) -> str:
    for key in ("model_params", "params", "num_params"):
        if key in config:
            try:
                return f"{int(float(config[key])):,}"
            except Exception:
                return str(config[key])
    return "-"


def _load_loss_curve(output_dir: str | None) -> tuple[list[int], list[float]]:
    if not output_dir:
        return [], []

    csv_path = Path(output_dir) / "results.csv"
    if not csv_path.exists():
        return [], []

    epochs = []
    losses = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epoch = row.get("epoch")
                if epoch is None:
                    continue
                try:
                    epoch_num = int(float(epoch))
                except Exception:
                    continue

                loss_values = []
                for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
                    raw = row.get(key)
                    if raw is None:
                        continue
                    try:
                        loss_values.append(float(raw))
                    except Exception:
                        continue

                if not loss_values:
                    continue

                epochs.append(epoch_num)
                losses.append(sum(loss_values))
    except Exception:
        LOGGER.exception("Failed to parse loss curve: %s", csv_path)
        return [], []

    return epochs, losses


def _load_class_performance(
    runs: dict[int, RunInfo],
    snapshot_root: Path,
) -> tuple[np.ndarray | None, list[str], list[str]]:
    if not snapshot_root.exists():
        return None, [], []

    class_labels: list[str] = []
    model_labels: list[str] = []
    rows: list[list[float]] = []

    for run in runs.values():
        snapshot_path = snapshot_root / f"run_{run.id}.json"
        if not snapshot_path.exists():
            continue

        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        per_class_map = payload.get("per_class_map")
        class_names = payload.get("class_names") or []
        if not per_class_map or not class_names:
            continue

        if not class_labels:
            class_labels = [str(name) for name in class_names]

        row = [0.0] * len(class_labels)
        for idx, value in per_class_map.items():
            try:
                cls_idx = int(idx)
            except Exception:
                continue
            if cls_idx < len(row):
                row[cls_idx] = float(value)

        rows.append(row)
        model_labels.append(run.name)

    if not rows:
        return None, [], []

    data = np.array(rows).T
    return data, class_labels, model_labels


def _compute_dataset_stats(dataset_root: Path, class_names: list[str]) -> tuple[int, list[int], list[int]]:
    images = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
        images.extend(dataset_root.rglob(f"*{ext}"))

    label_counts = [0] * len(class_names)
    densities = []

    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        try:
            lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_idx = int(float(parts[0]))
            except Exception:
                continue
            if 0 <= cls_idx < len(label_counts):
                label_counts[cls_idx] += 1
                count += 1
        if count:
            densities.append(count)

    return len(images), label_counts, densities


__all__ = ["AnalyticsTab"]

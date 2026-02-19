"""Remote devices tab for edge-device management and deploy/test workflows."""

from __future__ import annotations

import asyncio
import inspect
import logging
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.models.database import (
    Dataset,
    RemoteDevice,
    RemoteDeviceStatus,
    RemoteDeviceType,
    RemoteTestResult,
    TrainingRun,
    get_session,
)
from ui.widgets.device_card import DeviceCard


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_FALLBACK_ROOT = PROJECT_ROOT / "runs" / "remote_fallback"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class DeviceInfo:
    """Lightweight remote-device payload used by workers and selectors."""

    id: int
    name: str
    device_type: RemoteDeviceType
    host: str
    port: int
    auth_token: str
    status: RemoteDeviceStatus
    last_seen: datetime | None


@dataclass(slots=True)
class SavedRunInfo:
    """Saved training-run payload used by model selector."""

    id: int
    name: str
    architecture: str
    weights_path: str


class AddDeviceDialog(QDialog):
    """Dialog used to create a new remote-device record."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize add-device form fields.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self.setWindowTitle("Add Remote Device")
        self.resize(460, 280)

        self._name_input = QLineEdit(self)

        self._type_combo = QComboBox(self)
        self._type_combo.addItem("Jetson Nano", RemoteDeviceType.JETSON_NANO)
        self._type_combo.addItem("Jetson Xavier", RemoteDeviceType.XAVIER)
        self._type_combo.addItem("Raspberry Pi", RemoteDeviceType.RASPBERRY_PI)
        self._type_combo.addItem("Other", RemoteDeviceType.OTHER)

        self._host_input = QLineEdit(self)
        self._host_input.setPlaceholderText("192.168.1.101")

        self._port_input = QSpinBox(self)
        self._port_input.setRange(1, 65535)
        self._port_input.setValue(8765)

        self._token_input = QLineEdit(self)
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)

        form = QFormLayout()
        form.addRow("Name", self._name_input)
        form.addRow("Type", self._type_combo)
        form.addRow("Host", self._host_input)
        form.addRow("Port", self._port_input)
        form.addRow("Auth Token", self._token_input)

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
        """Return normalized add-device form values.

        Returns:
            dict[str, Any]: Device field payload.
        """

        return {
            "name": self._name_input.text().strip(),
            "device_type": self._type_combo.currentData(),
            "host": self._host_input.text().strip(),
            "port": int(self._port_input.value()),
            "auth_token": self._token_input.text().strip(),
        }


class PingAllWorker(QThread):
    """Worker that pings all registered devices without blocking the UI."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, devices: list[DeviceInfo], parent: Any | None = None) -> None:
        """Initialize ping worker.

        Args:
            devices: Device payloads to ping.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._devices = list(devices)

    def run(self) -> None:
        """Ping all devices and emit status updates/results."""

        try:
            total = len(self._devices)
            if total == 0:
                self.finished.emit({"devices": []})
                return

            manager = _create_remote_manager()
            results: list[dict[str, Any]] = []

            for index, device in enumerate(self._devices, start=1):
                self.status.emit(f"Pinging {device.name} ({device.host}:{device.port})...")

                try:
                    record = self._ping_one_device(device, manager)
                except Exception as exc:
                    record = {
                        "device_id": device.id,
                        "status": RemoteDeviceStatus.OFFLINE,
                        "last_seen": None,
                        "info": None,
                        "error": str(exc),
                    }

                results.append(record)
                progress = int((index / total) * 100)
                self.progress.emit(progress)

            self.status.emit("Ping completed.")
            self.finished.emit({"devices": results})
        except Exception as exc:
            LOGGER.exception("Ping-all worker failed.")
            self.error.emit(str(exc))

    def _ping_one_device(self, device: DeviceInfo, manager: Any | None) -> dict[str, Any]:
        """Ping one device and return normalized result payload.

        Args:
            device: Device metadata payload.
            manager: Optional remote manager instance.

        Returns:
            dict[str, Any]: Ping result payload.
        """

        info_payload: dict[str, Any] | None = None

        if manager is not None:
            response = _try_remote_ping(manager, device)
            if response is not None:
                ok = bool(response.get("ok", True))
                status = RemoteDeviceStatus.ONLINE if ok else RemoteDeviceStatus.OFFLINE
                last_seen = datetime.now(timezone.utc) if ok else None
                info_payload = response
                return {
                    "device_id": device.id,
                    "status": status,
                    "last_seen": last_seen,
                    "info": info_payload,
                    "error": None,
                }

        socket_ok = _tcp_probe(device.host, device.port, timeout=1.5)
        status = RemoteDeviceStatus.ONLINE if socket_ok else RemoteDeviceStatus.OFFLINE
        last_seen = datetime.now(timezone.utc) if socket_ok else None

        return {
            "device_id": device.id,
            "status": status,
            "last_seen": last_seen,
            "info": info_payload,
            "error": None,
        }


class DeployTestWorker(QThread):
    """Worker that deploys model weights and runs inference test on a device."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        device: DeviceInfo,
        weights_path: str,
        dataset_path: str,
        conf: float,
        iou: float,
        parent: Any | None = None,
    ) -> None:
        """Initialize deploy/test worker payload.

        Args:
            device: Target device metadata.
            weights_path: Local weights path.
            dataset_path: Dataset folder or data.yaml path.
            conf: Confidence threshold.
            iou: IoU threshold.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._device = device
        self._weights_path = str(Path(weights_path).resolve())
        self._dataset_path = str(Path(dataset_path).resolve())
        self._conf = float(conf)
        self._iou = float(iou)

    def run(self) -> None:
        """Execute deploy/test flow with remote-manager or local fallback."""

        try:
            if not Path(self._weights_path).exists():
                raise RuntimeError("Selected weights file does not exist.")

            dataset_target = _resolve_dataset_target(Path(self._dataset_path))

            manager = _create_remote_manager()
            if manager is not None:
                try:
                    self.status.emit("Starting remote deploy/test...")
                    result = self._run_remote(manager, dataset_target)
                    self.progress.emit(100)
                    self.status.emit("Remote deploy/test complete.")
                    self.finished.emit(result)
                    return
                except Exception as exc:
                    # Fall back to local execution when remote manager is unavailable or incompatible.
                    self.status.emit(f"Remote manager path failed ({exc}); running local fallback.")

            result = self._run_local_fallback(dataset_target)
            self.finished.emit(result)
        except Exception as exc:
            LOGGER.exception("Deploy/test worker failed.")
            self.error.emit(str(exc))

    def _run_remote(self, manager: Any, dataset_target: Path) -> dict[str, Any]:
        """Execute remote deploy/test using remote manager integration points.

        Args:
            manager: Remote manager instance.
            dataset_target: Resolved dataset target path.

        Returns:
            dict[str, Any]: Normalized test result payload.
        """

        method_candidates = (
            "deploy_and_test",
            "run_remote_test",
            "deploy_model_and_test",
        )

        method: Callable[..., Any] | None = None
        for method_name in method_candidates:
            candidate = getattr(manager, method_name, None)
            if callable(candidate):
                method = candidate
                break

        if method is None:
            raise RuntimeError("Remote manager does not expose a deploy/test method.")

        progress_cb = self.progress.emit
        status_cb = self.status.emit

        call_attempts = [
            {
                "device": {
                    "id": self._device.id,
                    "name": self._device.name,
                    "host": self._device.host,
                    "port": self._device.port,
                    "auth_token": self._device.auth_token,
                    "device_type": self._device.device_type.value,
                },
                "weights_path": self._weights_path,
                "model_path": self._weights_path,
                "dataset_path": str(dataset_target),
                "test_dataset_path": str(dataset_target),
                "conf": self._conf,
                "iou": self._iou,
                "progress_callback": progress_cb,
                "status_callback": status_cb,
            },
            {
                "payload": {
                    "device": {
                        "id": self._device.id,
                        "name": self._device.name,
                        "host": self._device.host,
                        "port": self._device.port,
                        "auth_token": self._device.auth_token,
                        "device_type": self._device.device_type.value,
                    },
                    "weights_path": self._weights_path,
                    "dataset_path": str(dataset_target),
                    "conf": self._conf,
                    "iou": self._iou,
                },
            },
        ]

        last_error: Exception | None = None
        response: Any | None = None

        for kwargs in call_attempts:
            try:
                response = _call_with_optional_async(method, **kwargs)
                break
            except TypeError as exc:
                last_error = exc
                continue

        if response is None:
            raise RuntimeError(f"Unable to call remote manager method: {last_error}")

        return _normalize_deploy_result(
            response=response,
            fallback_dataset=dataset_target,
            fallback_output_dir=None,
        )

    def _run_local_fallback(self, dataset_target: Path) -> dict[str, Any]:
        """Run local fallback validation when remote manager is unavailable.

        Args:
            dataset_target: Dataset folder or data.yaml path.

        Returns:
            dict[str, Any]: Normalized test result payload.
        """

        self.progress.emit(8)
        self.status.emit("Local fallback: loading Ultralytics model...")

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Remote manager is unavailable and ultralytics is not installed for fallback testing."
            ) from exc

        output_root = REMOTE_FALLBACK_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root.mkdir(parents=True, exist_ok=True)

        model = YOLO(self._weights_path)

        self.progress.emit(25)
        self.status.emit("Local fallback: running validation/prediction...")

        val_result: Any = None
        try:
            if dataset_target.is_file() and dataset_target.suffix.lower() in {".yaml", ".yml"}:
                val_kwargs = {
                    "data": str(dataset_target),
                    "conf": self._conf,
                    "iou": self._iou,
                    "project": str(output_root.parent),
                    "name": output_root.name,
                    "exist_ok": True,
                    "verbose": False,
                }
                val_result = model.val(**val_kwargs)
            else:
                raise RuntimeError("No data.yaml available for validation.")
        except Exception:
            # Prediction mode is used when dataset annotations or data.yaml are unavailable.
            predict_source = str(dataset_target.parent if dataset_target.is_file() else dataset_target)
            model.predict(
                source=predict_source,
                conf=self._conf,
                iou=self._iou,
                project=str(output_root.parent),
                name=output_root.name,
                save=True,
                exist_ok=True,
                verbose=False,
            )
            val_result = None

        self.progress.emit(70)
        self.status.emit("Local fallback: preparing result previews...")

        preview_dir = output_root / "preview_images"
        preview_dir.mkdir(parents=True, exist_ok=True)

        preview_sources = _collect_dataset_images(dataset_target)
        if not preview_sources:
            preview_sources = _collect_dataset_images(Path(self._dataset_path))

        output_images = _create_preview_images(
            source_images=preview_sources[:8],
            output_dir=preview_dir,
            overlay_text=f"{self._device.name} fallback",
        )

        metrics = _extract_val_metrics(val_result)

        self.progress.emit(100)
        self.status.emit("Local fallback deploy/test complete.")

        return {
            "metrics": metrics,
            "output_images": [str(path) for path in output_images],
            "output_images_dir": str(preview_dir.resolve()),
            "num_images_tested": len(preview_sources),
            "notes": "Executed via local fallback because remote manager was unavailable.",
        }


class RemoteTab(QWidget):
    """Tab that manages remote devices and deploy/test execution flows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the remote tab UI and load DB-backed state.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._active_project_id: int | None = None
        self._device_rows: dict[int, DeviceInfo] = {}
        self._saved_runs: dict[int, SavedRunInfo] = {}
        self._dataset_paths: dict[int, str] = {}

        self._selected_device_id: int | None = None
        self._custom_dataset_path: str | None = None

        self._ping_worker: PingAllWorker | None = None
        self._deploy_worker: DeployTestWorker | None = None

        self._latest_test_result: dict[str, Any] | None = None

        self._device_grid_layout: QGridLayout
        self._device_combo: QComboBox
        self._model_combo: QComboBox
        self._dataset_combo: QComboBox
        self._custom_dataset_path_input: QLineEdit
        self._conf_slider: QSlider
        self._iou_slider: QSlider
        self._conf_value_label: QLabel
        self._iou_value_label: QLabel
        self._deploy_progress_bar: QProgressBar
        self._deploy_status_label: QLabel
        self._results_map50: QLabel
        self._results_map50_95: QLabel
        self._results_precision: QLabel
        self._results_recall: QLabel
        self._results_speed: QLabel
        self._thumbnail_layout: QHBoxLayout
        self._save_results_button: QPushButton
        self._deploy_button: QPushButton

        self._build_ui()
        self.refresh_all()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        self.refresh_model_selector()
        self.refresh_dataset_selector()

    def refresh_all(self) -> None:
        """Refresh devices, model selector, and dataset selector from SQLite."""

        self.refresh_devices()
        self.refresh_model_selector()
        self.refresh_dataset_selector()

    def refresh_devices(self) -> None:
        """Load devices from DB and refresh cards/selectors."""

        self._device_rows.clear()

        session = get_session()
        try:
            devices = session.query(RemoteDevice).order_by(RemoteDevice.name.asc()).all()
        except Exception:
            LOGGER.exception("Failed to load remote devices.")
            devices = []
        finally:
            session.close()

        for device in devices:
            info = DeviceInfo(
                id=device.id,
                name=device.name,
                device_type=device.device_type,
                host=device.host,
                port=device.port,
                auth_token=device.auth_token,
                status=device.status,
                last_seen=device.last_seen,
            )
            self._device_rows[info.id] = info

        self._render_device_cards()
        self._rebuild_device_selector()

    def refresh_model_selector(self) -> None:
        """Load saved training runs that have valid weights paths."""

        selected_run_id = self._model_combo.currentData()

        self._saved_runs.clear()
        self._model_combo.clear()

        session = get_session()
        try:
            query = session.query(TrainingRun).filter(TrainingRun.is_saved.is_(True))
            if self._active_project_id is not None:
                query = query.filter(TrainingRun.project_id == self._active_project_id)
            runs = query.order_by(TrainingRun.completed_at.desc(), TrainingRun.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed loading saved runs for remote selector.")
            runs = []
        finally:
            session.close()

        for run in runs:
            if not run.weights_path:
                continue
            if not Path(run.weights_path).exists():
                continue

            info = SavedRunInfo(
                id=run.id,
                name=run.name,
                architecture=run.model_architecture,
                weights_path=run.weights_path,
            )
            self._saved_runs[info.id] = info
            self._model_combo.addItem(f"{info.name} (#{info.id}) [{info.architecture}]", info.id)

        if selected_run_id is not None:
            index = self._model_combo.findData(selected_run_id)
            if index >= 0:
                self._model_combo.setCurrentIndex(index)

    def refresh_dataset_selector(self) -> None:
        """Load datasets from DB for test-dataset selector."""

        selected_dataset_id = self._dataset_combo.currentData()

        self._dataset_paths.clear()
        self._dataset_combo.clear()

        session = get_session()
        try:
            query = session.query(Dataset)
            if self._active_project_id is not None:
                query = query.filter(Dataset.project_id == self._active_project_id)
            datasets = query.order_by(Dataset.name.asc()).all()
        except Exception:
            LOGGER.exception("Failed loading datasets for remote selector.")
            datasets = []
        finally:
            session.close()

        for dataset in datasets:
            path = str(Path(dataset.local_path).resolve())
            self._dataset_paths[dataset.id] = path
            self._dataset_combo.addItem(f"{dataset.name} (#{dataset.id})", dataset.id)

        if selected_dataset_id is not None:
            index = self._dataset_combo.findData(selected_dataset_id)
            if index >= 0:
                self._dataset_combo.setCurrentIndex(index)

        self._update_dataset_path_preview()

    def _build_ui(self) -> None:
        """Compose top device-manager and bottom test-runner sections."""

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_device_manager_panel())
        splitter.addWidget(self._build_test_runner_panel())
        splitter.setSizes([360, 520])

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(splitter)
        self.setLayout(root)

    def _build_device_manager_panel(self) -> QWidget:
        """Create device-manager top panel."""

        group = QGroupBox("Device Manager", self)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls = QWidget(group)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        add_button = QPushButton("Add Device", controls)
        add_button.clicked.connect(self._add_device)

        ping_button = QPushButton("Ping All", controls)
        ping_button.clicked.connect(self._ping_all_devices)

        remove_button = QPushButton("Remove Device", controls)
        remove_button.setProperty("danger", True)
        remove_button.clicked.connect(self._remove_selected_device)

        refresh_button = QPushButton("Refresh", controls)
        refresh_button.setProperty("secondary", True)
        refresh_button.clicked.connect(self.refresh_devices)

        controls_layout.addWidget(add_button)
        controls_layout.addWidget(ping_button)
        controls_layout.addWidget(remove_button)
        controls_layout.addWidget(refresh_button)
        controls_layout.addStretch(1)
        controls.setLayout(controls_layout)

        scroll = QScrollArea(group)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(scroll)
        self._device_grid_layout = QGridLayout()
        self._device_grid_layout.setContentsMargins(0, 0, 0, 0)
        self._device_grid_layout.setSpacing(10)
        container.setLayout(self._device_grid_layout)

        scroll.setWidget(container)

        layout.addWidget(controls)
        layout.addWidget(scroll, stretch=1)
        group.setLayout(layout)

        return group

    def _build_test_runner_panel(self) -> QWidget:
        """Create deploy/test bottom panel."""

        group = QGroupBox("Test Runner", self)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        form = QFormLayout()

        self._device_combo = QComboBox(group)
        self._device_combo.currentIndexChanged.connect(self._on_device_combo_changed)

        self._model_combo = QComboBox(group)

        dataset_row = QWidget(group)
        dataset_row_layout = QHBoxLayout()
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.setSpacing(8)

        self._dataset_combo = QComboBox(dataset_row)
        self._dataset_combo.currentIndexChanged.connect(self._update_dataset_path_preview)

        browse_dataset_button = QPushButton("Browse Folder", dataset_row)
        browse_dataset_button.setProperty("secondary", True)
        browse_dataset_button.clicked.connect(self._browse_custom_dataset)

        clear_dataset_button = QPushButton("Clear", dataset_row)
        clear_dataset_button.setProperty("secondary", True)
        clear_dataset_button.clicked.connect(self._clear_custom_dataset)

        dataset_row_layout.addWidget(self._dataset_combo, stretch=1)
        dataset_row_layout.addWidget(browse_dataset_button)
        dataset_row_layout.addWidget(clear_dataset_button)
        dataset_row.setLayout(dataset_row_layout)

        self._custom_dataset_path_input = QLineEdit(group)
        self._custom_dataset_path_input.setReadOnly(True)
        self._custom_dataset_path_input.setObjectName("codeField")

        conf_row = QWidget(group)
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

        iou_row = QWidget(group)
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

        form.addRow("Device", self._device_combo)
        form.addRow("Model", self._model_combo)
        form.addRow("Dataset", dataset_row)
        form.addRow("Dataset Path", self._custom_dataset_path_input)
        form.addRow("Confidence", conf_row)
        form.addRow("IoU", iou_row)

        self._deploy_button = QPushButton("Deploy & Test", group)
        self._deploy_button.clicked.connect(self._start_deploy_and_test)

        self._deploy_progress_bar = QProgressBar(group)
        self._deploy_progress_bar.setRange(0, 100)
        self._deploy_progress_bar.setValue(0)

        self._deploy_status_label = QLabel("Ready for deploy/test.", group)
        self._deploy_status_label.setProperty("role", "subtle")

        results_group = QGroupBox("Results", group)
        results_layout = QGridLayout()
        results_layout.setHorizontalSpacing(12)
        results_layout.setVerticalSpacing(8)

        self._results_map50 = QLabel("-", results_group)
        self._results_map50.setObjectName("metricValue")

        self._results_map50_95 = QLabel("-", results_group)
        self._results_map50_95.setObjectName("metricValue")

        self._results_precision = QLabel("-", results_group)
        self._results_precision.setObjectName("metricValue")

        self._results_recall = QLabel("-", results_group)
        self._results_recall.setObjectName("metricValue")

        self._results_speed = QLabel("-", results_group)
        self._results_speed.setObjectName("metricValue")

        results_layout.addWidget(QLabel("mAP50", results_group), 0, 0)
        results_layout.addWidget(self._results_map50, 0, 1)
        results_layout.addWidget(QLabel("mAP50_95", results_group), 0, 2)
        results_layout.addWidget(self._results_map50_95, 0, 3)
        results_layout.addWidget(QLabel("Precision", results_group), 1, 0)
        results_layout.addWidget(self._results_precision, 1, 1)
        results_layout.addWidget(QLabel("Recall", results_group), 1, 2)
        results_layout.addWidget(self._results_recall, 1, 3)
        results_layout.addWidget(QLabel("Inference Speed (ms/img)", results_group), 2, 0)
        results_layout.addWidget(self._results_speed, 2, 1)

        results_group.setLayout(results_layout)

        thumbnails_group = QGroupBox("Annotated Result Thumbnails", group)
        thumbnails_layout = QVBoxLayout()

        thumb_scroll = QScrollArea(thumbnails_group)
        thumb_scroll.setWidgetResizable(True)
        thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        thumb_scroll.setFrameShape(QFrame.Shape.NoFrame)

        thumb_container = QWidget(thumb_scroll)
        self._thumbnail_layout = QHBoxLayout()
        self._thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self._thumbnail_layout.setSpacing(8)
        thumb_container.setLayout(self._thumbnail_layout)

        thumb_scroll.setWidget(thumb_container)

        thumbnails_layout.addWidget(thumb_scroll)
        thumbnails_group.setLayout(thumbnails_layout)

        self._save_results_button = QPushButton("Save Results", group)
        self._save_results_button.setProperty("secondary", True)
        self._save_results_button.setEnabled(False)
        self._save_results_button.clicked.connect(self._save_test_results)

        layout.addLayout(form)
        layout.addWidget(self._deploy_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._deploy_progress_bar)
        layout.addWidget(self._deploy_status_label)
        layout.addWidget(results_group)
        layout.addWidget(thumbnails_group)
        layout.addWidget(self._save_results_button, alignment=Qt.AlignmentFlag.AlignRight)

        group.setLayout(layout)

        self._update_threshold_labels()
        self._set_thumbnail_placeholder("No result images yet.")

        return group

    def _render_device_cards(self) -> None:
        """Render grid of device cards from current device payloads."""

        _clear_layout(self._device_grid_layout)

        if not self._device_rows:
            placeholder = QLabel("No remote devices registered.", self)
            placeholder.setProperty("role", "subtle")
            self._device_grid_layout.addWidget(placeholder, 0, 0)
            return

        columns = 3
        for index, device in enumerate(self._device_rows.values()):
            row = index // columns
            col = index % columns

            card = DeviceCard(
                device_id=device.id,
                name=device.name,
                device_type=device.device_type,
                host=device.host,
                port=device.port,
                status=device.status,
                last_seen=device.last_seen,
                parent=self,
            )
            card.clicked.connect(self._on_device_card_clicked)
            card.set_selected(device.id == self._selected_device_id)
            self._device_grid_layout.addWidget(card, row, col)

        self._device_grid_layout.setColumnStretch(columns, 1)

    def _rebuild_device_selector(self) -> None:
        """Rebuild device dropdown selector from current device payloads."""

        previous = self._selected_device_id

        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        for device in self._device_rows.values():
            label = f"{device.name} ({device.host}:{device.port})"
            self._device_combo.addItem(label, device.id)
        self._device_combo.blockSignals(False)

        if previous is not None:
            index = self._device_combo.findData(previous)
            if index >= 0:
                self._device_combo.setCurrentIndex(index)
                self._selected_device_id = previous
                self._render_device_cards()
                return

        if self._device_combo.count() > 0:
            self._device_combo.setCurrentIndex(0)
            self._selected_device_id = int(self._device_combo.currentData())
        else:
            self._selected_device_id = None

        self._render_device_cards()

    def _on_device_card_clicked(self, device_id: int) -> None:
        """Handle card-click selection updates.

        Args:
            device_id: Selected device ID.
        """

        self._selected_device_id = int(device_id)
        index = self._device_combo.findData(self._selected_device_id)
        if index >= 0:
            self._device_combo.blockSignals(True)
            self._device_combo.setCurrentIndex(index)
            self._device_combo.blockSignals(False)
        self._render_device_cards()

    def _on_device_combo_changed(self, _index: int) -> None:
        """Handle dropdown selection updates.

        Args:
            _index: Current selector index (unused).
        """

        data = self._device_combo.currentData()
        self._selected_device_id = int(data) if data is not None else None
        self._render_device_cards()

    def _add_device(self) -> None:
        """Open add-device dialog and persist remote device in DB."""

        dialog = AddDeviceDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        if not values["name"]:
            QMessageBox.warning(self, "Validation", "Device name is required.")
            return
        if not values["host"]:
            QMessageBox.warning(self, "Validation", "Host is required.")
            return
        if not values["auth_token"]:
            QMessageBox.warning(self, "Validation", "Auth token is required.")
            return

        session = get_session()
        try:
            record = RemoteDevice(
                name=values["name"],
                device_type=values["device_type"],
                host=values["host"],
                port=values["port"],
                auth_token=values["auth_token"],
                status=RemoteDeviceStatus.UNKNOWN,
                last_seen=None,
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Add Device Failed", f"Could not add device: {exc}")
            return
        finally:
            session.close()

        self.refresh_devices()

    def _remove_selected_device(self) -> None:
        """Delete selected device from DB after confirmation."""

        device_id = self._selected_device_id
        if device_id is None:
            QMessageBox.information(self, "Remove Device", "Select a device first.")
            return

        device = self._device_rows.get(device_id)
        if device is None:
            return

        answer = QMessageBox.question(
            self,
            "Remove Device",
            f"Remove device '{device.name}' from the registry?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        session = get_session()
        try:
            record = session.get(RemoteDevice, device_id)
            if record is None:
                return
            session.delete(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Remove Device Failed", f"Could not remove device: {exc}")
            return
        finally:
            session.close()

        self.refresh_devices()

    def _ping_all_devices(self) -> None:
        """Run async-style ping-all flow in a worker thread."""

        if self._ping_worker is not None and self._ping_worker.isRunning():
            QMessageBox.information(self, "Ping Running", "Ping operation is already in progress.")
            return

        devices = list(self._device_rows.values())
        if not devices:
            QMessageBox.information(self, "Ping Devices", "No devices to ping.")
            return

        self._deploy_status_label.setText("Pinging all devices...")
        self._deploy_progress_bar.setValue(0)

        self._ping_worker = PingAllWorker(devices=devices, parent=self)
        self._ping_worker.progress.connect(self._deploy_progress_bar.setValue)
        self._ping_worker.status.connect(self._deploy_status_label.setText)
        self._ping_worker.finished.connect(self._on_ping_all_finished)
        self._ping_worker.error.connect(self._on_ping_all_error)
        self._ping_worker.start()

    def _on_ping_all_finished(self, payload: dict[str, Any]) -> None:
        """Persist ping statuses and refresh UI.

        Args:
            payload: Ping results payload.
        """

        records = payload.get("devices") or []

        session = get_session()
        try:
            for record in records:
                device_id = int(record.get("device_id"))
                status = record.get("status")
                last_seen = record.get("last_seen")

                db_device = session.get(RemoteDevice, device_id)
                if db_device is None:
                    continue

                if isinstance(status, RemoteDeviceStatus):
                    db_device.status = status
                else:
                    try:
                        db_device.status = RemoteDeviceStatus(str(status))
                    except Exception:
                        db_device.status = RemoteDeviceStatus.UNKNOWN

                db_device.last_seen = last_seen if isinstance(last_seen, datetime) else db_device.last_seen

            session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed persisting ping results.")
        finally:
            session.close()

        self._deploy_status_label.setText("Ping completed.")
        self._deploy_progress_bar.setValue(100)
        self.refresh_devices()
        self._ping_worker = None

    def _on_ping_all_error(self, message: str) -> None:
        """Handle ping worker errors.

        Args:
            message: Error text.
        """

        self._deploy_status_label.setText(f"Ping failed: {message}")
        self._deploy_progress_bar.setValue(0)
        QMessageBox.critical(self, "Ping Error", message)
        self._ping_worker = None

    def _browse_custom_dataset(self) -> None:
        """Select a custom dataset folder for the next test run."""

        selected = QFileDialog.getExistingDirectory(self, "Select Test Dataset Folder")
        if not selected:
            return

        self._custom_dataset_path = str(Path(selected).resolve())
        self._update_dataset_path_preview()

    def _clear_custom_dataset(self) -> None:
        """Clear custom dataset override and return to DB selector path."""

        self._custom_dataset_path = None
        self._update_dataset_path_preview()

    def _update_dataset_path_preview(self) -> None:
        """Update resolved dataset-path preview based on current selector state."""

        resolved = self._resolve_selected_dataset_path()
        self._custom_dataset_path_input.setText(resolved or "")

    def _resolve_selected_dataset_path(self) -> str | None:
        """Resolve effective dataset path from custom or DB selector values.

        Returns:
            str | None: Effective dataset path if available.
        """

        if self._custom_dataset_path:
            return self._custom_dataset_path

        dataset_id = self._dataset_combo.currentData()
        if dataset_id is None:
            return None

        return self._dataset_paths.get(int(dataset_id))

    def _update_threshold_labels(self) -> None:
        """Update confidence/IoU value labels from sliders."""

        self._conf_value_label.setText(f"{self._conf_slider.value() / 100.0:.2f}")
        self._iou_value_label.setText(f"{self._iou_slider.value() / 100.0:.2f}")

    def _start_deploy_and_test(self) -> None:
        """Start deploy/test worker with current selector values."""

        if self._deploy_worker is not None and self._deploy_worker.isRunning():
            QMessageBox.information(self, "Deploy/Test Running", "A deploy/test run is already in progress.")
            return

        device_id = self._device_combo.currentData()
        if device_id is None:
            QMessageBox.warning(self, "Validation", "Select a remote device.")
            return

        run_id = self._model_combo.currentData()
        if run_id is None:
            QMessageBox.warning(self, "Validation", "Select a saved model.")
            return

        dataset_path = self._resolve_selected_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Validation", "Select a dataset path.")
            return

        device = self._device_rows.get(int(device_id))
        run_info = self._saved_runs.get(int(run_id))
        if device is None or run_info is None:
            QMessageBox.warning(self, "Validation", "Selected device/model is no longer available.")
            return

        if not Path(run_info.weights_path).exists():
            QMessageBox.warning(self, "Validation", "Selected weights file does not exist.")
            return

        if not Path(dataset_path).exists():
            QMessageBox.warning(self, "Validation", "Selected dataset path does not exist.")
            return

        self._latest_test_result = None
        self._save_results_button.setEnabled(False)
        self._deploy_button.setEnabled(False)

        self._clear_result_metrics()
        self._set_thumbnail_placeholder("Running deploy/test...")

        self._deploy_progress_bar.setValue(0)
        self._deploy_status_label.setText("Starting deploy/test...")

        self._deploy_worker = DeployTestWorker(
            device=device,
            weights_path=run_info.weights_path,
            dataset_path=dataset_path,
            conf=self._conf_slider.value() / 100.0,
            iou=self._iou_slider.value() / 100.0,
            parent=self,
        )
        self._deploy_worker.progress.connect(self._deploy_progress_bar.setValue)
        self._deploy_worker.status.connect(self._deploy_status_label.setText)
        self._deploy_worker.finished.connect(self._on_deploy_test_finished)
        self._deploy_worker.error.connect(self._on_deploy_test_error)
        self._deploy_worker.start()

    def _on_deploy_test_finished(self, result: dict[str, Any]) -> None:
        """Handle completed deploy/test payload and update result panels.

        Args:
            result: Test result payload.
        """

        self._latest_test_result = dict(result)

        metrics = result.get("metrics") or {}
        self._results_map50.setText(_format_metric(metrics.get("map50")))
        self._results_map50_95.setText(_format_metric(metrics.get("map50_95")))
        self._results_precision.setText(_format_metric(metrics.get("precision")))
        self._results_recall.setText(_format_metric(metrics.get("recall")))
        self._results_speed.setText(_format_metric(metrics.get("speed_ms"), fallback="-", decimals=2))

        output_images = [str(path) for path in (result.get("output_images") or [])]
        self._render_thumbnails(output_images)

        self._deploy_status_label.setText("Deploy/test complete.")
        self._deploy_progress_bar.setValue(100)
        self._deploy_button.setEnabled(True)

        if self._device_combo.currentData() is not None and self._model_combo.currentData() is not None:
            self._save_results_button.setEnabled(True)

        self._deploy_worker = None

    def _on_deploy_test_error(self, message: str) -> None:
        """Handle deploy/test worker errors.

        Args:
            message: Error text.
        """

        self._deploy_status_label.setText(f"Deploy/test failed: {message}")
        self._deploy_progress_bar.setValue(0)
        self._deploy_button.setEnabled(True)
        self._save_results_button.setEnabled(False)
        self._set_thumbnail_placeholder("No result images available.")
        QMessageBox.critical(self, "Deploy & Test Error", message)
        self._deploy_worker = None

    def _save_test_results(self) -> None:
        """Persist latest test metrics into RemoteTestResult table."""

        if not self._latest_test_result:
            QMessageBox.information(self, "Save Results", "No test results available.")
            return

        device_id = self._device_combo.currentData()
        run_id = self._model_combo.currentData()
        if device_id is None or run_id is None:
            QMessageBox.warning(self, "Save Results", "Select both a device and model before saving.")
            return

        dataset_path = self._resolve_selected_dataset_path()
        if not dataset_path:
            QMessageBox.warning(self, "Save Results", "Dataset path is unavailable.")
            return

        notes, accepted = QInputDialog.getMultiLineText(
            self,
            "Save Remote Test Result",
            "Optional notes:",
            str(self._latest_test_result.get("notes") or ""),
        )
        if not accepted:
            notes = str(self._latest_test_result.get("notes") or "")

        metrics = self._latest_test_result.get("metrics") or {}
        output_images_dir = self._latest_test_result.get("output_images_dir")
        num_images_tested = int(self._latest_test_result.get("num_images_tested") or 0)

        session = get_session()
        try:
            record = RemoteTestResult(
                device_id=int(device_id),
                training_run_id=int(run_id),
                run_at=datetime.now(timezone.utc),
                test_dataset_path=str(Path(dataset_path).resolve()),
                source_type="dataset",
                source_path=str(Path(dataset_path).resolve()),
                num_images_tested=num_images_tested,
                map50=_to_float(metrics.get("map50")),
                map50_95=_to_float(metrics.get("map50_95")),
                precision=_to_float(metrics.get("precision")),
                recall=_to_float(metrics.get("recall")),
                output_images_dir=str(output_images_dir) if output_images_dir else None,
                notes=notes.strip() or None,
            )
            session.add(record)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Save Failed", f"Could not save remote test result: {exc}")
            return
        finally:
            session.close()

        QMessageBox.information(self, "Saved", "Remote test result saved to database.")

    def _clear_result_metrics(self) -> None:
        """Clear all result metric labels."""

        self._results_map50.setText("-")
        self._results_map50_95.setText("-")
        self._results_precision.setText("-")
        self._results_recall.setText("-")
        self._results_speed.setText("-")

    def _render_thumbnails(self, image_paths: list[str]) -> None:
        """Render thumbnail strip from image path list.

        Args:
            image_paths: Annotated image paths.
        """

        _clear_layout(self._thumbnail_layout)

        if not image_paths:
            self._set_thumbnail_placeholder("No result images available.")
            return

        rendered = 0
        for image_path in image_paths:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                continue

            label = QLabel(self)
            label.setPixmap(
                pixmap.scaled(
                    180,
                    120,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            label.setToolTip(image_path)
            label.setFrameShape(QFrame.Shape.StyledPanel)
            self._thumbnail_layout.addWidget(label)
            rendered += 1

        if rendered == 0:
            self._set_thumbnail_placeholder("Result images could not be rendered.")
            return

        self._thumbnail_layout.addStretch(1)

    def _set_thumbnail_placeholder(self, message: str) -> None:
        """Show a placeholder message in thumbnail strip.

        Args:
            message: Placeholder text.
        """

        _clear_layout(self._thumbnail_layout)
        placeholder = QLabel(message, self)
        placeholder.setProperty("role", "subtle")
        self._thumbnail_layout.addWidget(placeholder)
        self._thumbnail_layout.addStretch(1)


def _create_remote_manager() -> Any | None:
    """Instantiate remote-manager integration class when available.

    Returns:
        Any | None: Remote manager instance or None.
    """

    try:
        from core.services.remote_manager import RemoteManager  # type: ignore
    except Exception:
        return None

    if not callable(RemoteManager):
        return None

    try:
        return RemoteManager()
    except TypeError:
        try:
            return RemoteManager(parent=None)
        except Exception:
            return None


def _try_remote_ping(manager: Any, device: DeviceInfo) -> dict[str, Any] | None:
    """Attempt ping call against flexible remote-manager APIs.

    Args:
        manager: Remote manager object.
        device: Device payload.

    Returns:
        dict[str, Any] | None: Ping response if available.
    """

    methods = ("ping_device", "ping", "heartbeat")
    for method_name in methods:
        method = getattr(manager, method_name, None)
        if not callable(method):
            continue

        attempts = [
            {
                "host": device.host,
                "port": device.port,
                "auth_token": device.auth_token,
                "device": {
                    "id": device.id,
                    "host": device.host,
                    "port": device.port,
                    "auth_token": device.auth_token,
                },
            },
            {
                "device": {
                    "id": device.id,
                    "host": device.host,
                    "port": device.port,
                    "auth_token": device.auth_token,
                }
            },
        ]

        for kwargs in attempts:
            try:
                response = _call_with_optional_async(method, **kwargs)
                if isinstance(response, dict):
                    return response
                return {"ok": bool(response), "raw": response}
            except TypeError:
                continue
            except Exception:
                LOGGER.debug("Remote ping attempt failed for method %s", method_name, exc_info=True)
                continue

    return None


def _call_with_optional_async(callable_obj: Callable[..., Any], **kwargs: Any) -> Any:
    """Invoke a callable that may return a coroutine.

    Args:
        callable_obj: Target callable.
        **kwargs: Keyword args for invocation.

    Returns:
        Any: Resolved return value.
    """

    result = callable_obj(**kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _tcp_probe(host: str, port: int, timeout: float) -> bool:
    """Probe TCP endpoint availability.

    Args:
        host: Hostname/IP.
        port: Port number.
        timeout: Probe timeout seconds.

    Returns:
        bool: True when endpoint accepts connection.
    """

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _resolve_dataset_target(path: Path) -> Path:
    """Resolve dataset target path for validation/inference calls.

    Args:
        path: Selected dataset path.

    Returns:
        Path: data.yaml path when present, otherwise original folder path.
    """

    resolved = path.resolve()
    if resolved.is_file() and resolved.suffix.lower() in {".yaml", ".yml"}:
        return resolved

    candidate_yaml = resolved / "data.yaml"
    if candidate_yaml.exists():
        return candidate_yaml

    return resolved


def _normalize_deploy_result(response: Any, fallback_dataset: Path, fallback_output_dir: Path | None) -> dict[str, Any]:
    """Normalize remote-manager response into UI-expected payload.

    Args:
        response: Raw manager response.
        fallback_dataset: Dataset path used for fallback defaults.
        fallback_output_dir: Fallback output directory if response omits it.

    Returns:
        dict[str, Any]: Normalized result payload.
    """

    if isinstance(response, dict):
        raw = dict(response)
    else:
        raw = {"raw": response}

    metrics_payload = raw.get("metrics")
    if not isinstance(metrics_payload, dict):
        metrics_payload = raw

    metrics = {
        "map50": _pick_metric(metrics_payload, ("map50", "mAP50", "metrics/mAP50(B)")),
        "map50_95": _pick_metric(metrics_payload, ("map50_95", "mAP50-95", "metrics/mAP50-95(B)")),
        "precision": _pick_metric(metrics_payload, ("precision", "mp", "metrics/precision(B)")),
        "recall": _pick_metric(metrics_payload, ("recall", "mr", "metrics/recall(B)")),
        "speed_ms": _pick_metric(metrics_payload, ("speed_ms", "inference_ms", "speed")),
    }

    output_images = raw.get("output_images") or raw.get("images") or []
    if isinstance(output_images, str):
        output_images = [output_images]
    output_images = [str(path) for path in output_images if path]

    output_images_dir = raw.get("output_images_dir")
    if output_images_dir is None and output_images:
        output_images_dir = str(Path(output_images[0]).resolve().parent)
    if output_images_dir is None and fallback_output_dir is not None:
        output_images_dir = str(fallback_output_dir.resolve())

    num_images_tested = _to_int(raw.get("num_images_tested"))
    if num_images_tested is None:
        num_images_tested = len(_collect_dataset_images(fallback_dataset))

    return {
        "metrics": metrics,
        "output_images": output_images,
        "output_images_dir": output_images_dir,
        "num_images_tested": num_images_tested,
        "notes": str(raw.get("notes") or "Remote deploy/test completed.").strip(),
    }


def _extract_val_metrics(val_result: Any) -> dict[str, float | None]:
    """Extract common metric values from Ultralytics validation result object.

    Args:
        val_result: Validation return object.

    Returns:
        dict[str, float | None]: Normalized metric fields.
    """

    metrics: dict[str, float | None] = {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "speed_ms": None,
    }

    if val_result is None:
        return metrics

    if isinstance(val_result, dict):
        metrics["map50"] = _pick_metric(val_result, ("map50", "mAP50", "metrics/mAP50(B)"))
        metrics["map50_95"] = _pick_metric(val_result, ("map50_95", "mAP50-95", "metrics/mAP50-95(B)"))
        metrics["precision"] = _pick_metric(val_result, ("precision", "metrics/precision(B)", "mp"))
        metrics["recall"] = _pick_metric(val_result, ("recall", "metrics/recall(B)", "mr"))
        metrics["speed_ms"] = _pick_metric(val_result, ("speed", "inference_ms", "speed_ms"))
        return metrics

    box = getattr(val_result, "box", None)
    if box is not None:
        metrics["map50"] = _to_float(getattr(box, "map50", None))
        metrics["map50_95"] = _to_float(getattr(box, "map", None))
        metrics["precision"] = _to_float(getattr(box, "mp", None))
        metrics["recall"] = _to_float(getattr(box, "mr", None))

    speed = getattr(val_result, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_ms"] = _pick_metric(speed, ("inference", "inference_ms", "speed"))

    results_dict = getattr(val_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics["map50"] = metrics["map50"] or _pick_metric(results_dict, ("metrics/mAP50(B)", "map50"))
        metrics["map50_95"] = metrics["map50_95"] or _pick_metric(
            results_dict,
            ("metrics/mAP50-95(B)", "map50_95"),
        )
        metrics["precision"] = metrics["precision"] or _pick_metric(
            results_dict,
            ("metrics/precision(B)", "precision"),
        )
        metrics["recall"] = metrics["recall"] or _pick_metric(
            results_dict,
            ("metrics/recall(B)", "recall"),
        )

    return metrics


def _pick_metric(mapping: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Pick first available numeric metric value from key candidates.

    Args:
        mapping: Metric mapping.
        keys: Candidate keys.

    Returns:
        float | None: Metric value.
    """

    for key in keys:
        if key in mapping:
            numeric = _to_float(mapping.get(key))
            if numeric is not None:
                return numeric

    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        key_lower = key.lower()
        for existing_key, value in lower_map.items():
            if key_lower in existing_key:
                numeric = _to_float(value)
                if numeric is not None:
                    return numeric

    return None


def _to_float(value: Any) -> float | None:
    """Convert scalar-like values to float.

    Args:
        value: Scalar-like input.

    Returns:
        float | None: Float value if conversion succeeds.
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    """Convert scalar-like values to int.

    Args:
        value: Scalar-like input.

    Returns:
        int | None: Int value if conversion succeeds.
    """

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _format_metric(value: Any, fallback: str = "-", decimals: int = 4) -> str:
    """Format metric value text for UI labels.

    Args:
        value: Metric value.
        fallback: Fallback when value is missing.
        decimals: Decimal precision.

    Returns:
        str: Formatted label text.
    """

    numeric = _to_float(value)
    if numeric is None:
        return fallback
    return f"{numeric:.{decimals}f}"


def _collect_dataset_images(dataset_path: Path) -> list[Path]:
    """Collect dataset image files recursively.

    Args:
        dataset_path: Dataset folder or yaml path.

    Returns:
        list[Path]: Image file paths.
    """

    if dataset_path.is_file():
        dataset_root = dataset_path.parent
    else:
        dataset_root = dataset_path

    images: list[Path] = []
    for file_path in dataset_root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(file_path.resolve())

    return images


def _create_preview_images(source_images: list[Path], output_dir: Path, overlay_text: str) -> list[Path]:
    """Create preview images with lightweight overlay for thumbnail strip.

    Args:
        source_images: Source image list.
        output_dir: Output folder for generated previews.
        overlay_text: Overlay text marker.

    Returns:
        list[Path]: Generated preview image paths.
    """

    output_paths: list[Path] = []

    for index, source in enumerate(source_images, start=1):
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(image)
        width, height = image.size

        draw.rectangle([(3, 3), (width - 4, height - 4)], outline=(0, 212, 170), width=3)
        draw.rectangle([(6, 6), (min(width - 6, 280), 34)], fill=(22, 22, 22))
        draw.text((10, 10), overlay_text, fill=(232, 232, 232))

        out_path = output_dir / f"result_{index:03d}.jpg"
        image.save(out_path, format="JPEG", quality=92)
        output_paths.append(out_path.resolve())

    return output_paths


def _clear_layout(layout: QGridLayout | QHBoxLayout | QVBoxLayout) -> None:
    """Delete all widgets/items from a layout.

    Args:
        layout: Layout to clear.
    """

    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()

        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)  # type: ignore[arg-type]


__all__ = [
    "RemoteTab",
    "AddDeviceDialog",
    "PingAllWorker",
    "DeployTestWorker",
]

"""Discover tab for searching and importing assets from Roboflow and Hugging Face."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy import or_

from core.models.database import (
    BaseWeight,
    BaseWeightSource,
    Dataset,
    DatasetSource,
    get_session,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
ROBOFLOW_DOWNLOAD_ROOT = PROJECT_ROOT / "datasets" / "roboflow_downloads"
HF_MODEL_CACHE_ROOT = PROJECT_ROOT / "models" / "huggingface"
HF_DATASET_CACHE_ROOT = PROJECT_ROOT / "datasets" / "huggingface"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class RoboflowSearchWorker(QThread):
    """Background worker for Roboflow project search operations."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, api_key: str, query: str, page: int, page_size: int = 8, parent: Any | None = None) -> None:
        """Initialize search worker state.

        Args:
            api_key: Roboflow API key.
            query: Search text.
            page: 1-based page index.
            page_size: Number of result cards per page.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._api_key = api_key.strip()
        self._query = query.strip()
        self._page = max(1, page)
        self._page_size = max(1, page_size)

    def run(self) -> None:
        """Execute search and emit normalized result payload."""

        try:
            if not self._api_key:
                raise RuntimeError("Roboflow API key is required.")

            self.progress.emit(10)
            self.status.emit("Connecting to Roboflow...")

            try:
                from roboflow import Roboflow
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("roboflow package is not installed.") from exc

            rf = Roboflow(api_key=self._api_key)
            self.progress.emit(25)

            results, has_next, total = self._perform_search(rf)

            self.progress.emit(100)
            self.status.emit("Roboflow search complete.")
            self.finished.emit(
                {
                    "results": results,
                    "page": self._page,
                    "page_size": self._page_size,
                    "has_next": has_next,
                    "total": total,
                }
            )
        except Exception as exc:
            LOGGER.exception("Roboflow search failed.")
            self.error.emit(str(exc))

    def _perform_search(self, rf: Any) -> tuple[list[dict[str, Any]], bool, int | None]:
        """Perform a Roboflow search attempt and normalize pagination metadata.

        Args:
            rf: Instantiated Roboflow client.

        Returns:
            tuple[list[dict[str, Any]], bool, int | None]: Results, has-next indicator, total count if known.
        """

        # Preferred path: SDK-based project search when available.
        for method_name in ("search", "search_projects", "universe_search"):
            method = getattr(rf, method_name, None)
            if not callable(method):
                continue

            self.status.emit(f"Searching with SDK method '{method_name}'...")
            self.progress.emit(40)

            try:
                raw = method(query=self._query, page=self._page, per_page=self._page_size)
            except TypeError:
                try:
                    raw = method(self._query, self._page, self._page_size)
                except TypeError:
                    raw = method(self._query)

            return _paginate_roboflow_response(raw, self._page, self._page_size)

        # Fallback: direct project lookup (workspace/project) when Universe search is unavailable.
        token = self._query.strip()
        if "/" not in token:
            raise RuntimeError(
                "Current roboflow SDK does not expose Universe search in this environment. "
                "Use query format 'workspace/project' or upgrade the SDK."
            )

        workspace_slug, project_slug = token.split("/", 1)
        workspace_slug = workspace_slug.strip()
        project_slug = project_slug.strip()
        if not workspace_slug or not project_slug:
            raise RuntimeError("Query must be in format 'workspace/project'.")

        self.status.emit("Resolving project directly from workspace/project...")
        self.progress.emit(70)

        workspace = rf.workspace(workspace_slug)
        project = workspace.project(project_slug)
        item = _normalize_roboflow_project(project, workspace_slug=workspace_slug, project_slug=project_slug)

        return [item], False, 1


class RoboflowDownloadWorker(QThread):
    """Background worker for downloading Roboflow datasets."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        workspace_slug: str,
        project_slug: str,
        version: int | None,
        download_root: Path,
        parent: Any | None = None,
    ) -> None:
        """Initialize Roboflow download worker.

        Args:
            api_key: Roboflow API key.
            workspace_slug: Workspace slug.
            project_slug: Project slug.
            version: Optional project version number.
            download_root: Local root folder for downloads.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._api_key = api_key.strip()
        self._workspace_slug = workspace_slug.strip()
        self._project_slug = project_slug.strip()
        self._version = version
        self._download_root = download_root

    def run(self) -> None:
        """Download dataset and emit path + metadata payload."""

        try:
            if not self._api_key:
                raise RuntimeError("Roboflow API key is required.")
            if not self._workspace_slug or not self._project_slug:
                raise RuntimeError("Workspace and project slugs are required for download.")

            try:
                from roboflow import Roboflow
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("roboflow package is not installed.") from exc

            self.progress.emit(10)
            self.status.emit("Connecting to Roboflow workspace...")
            rf = Roboflow(api_key=self._api_key)

            workspace = rf.workspace(self._workspace_slug)
            project = workspace.project(self._project_slug)

            version_number = self._resolve_version(project)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_dir = self._download_root / f"{self._workspace_slug}_{self._project_slug}_v{version_number}_{timestamp}"
            download_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(50)
            self.status.emit(f"Downloading YOLO dataset v{version_number}...")

            version_obj = project.version(version_number)
            downloaded = version_obj.download("yolov8", location=str(download_dir))

            dataset_path = _resolve_downloaded_path(downloaded, download_dir)
            class_names = _load_class_names_from_yaml(dataset_path / "data.yaml")
            if not class_names:
                class_names = _infer_class_names_from_labels(dataset_path)

            num_images = _count_images(dataset_path)

            self.progress.emit(100)
            self.status.emit("Roboflow dataset download complete.")
            self.finished.emit(
                {
                    "dataset_path": str(dataset_path.resolve()),
                    "workspace": self._workspace_slug,
                    "project_slug": self._project_slug,
                    "version": version_number,
                    "class_names": class_names,
                    "num_images": num_images,
                }
            )
        except Exception as exc:
            LOGGER.exception("Roboflow download failed.")
            self.error.emit(str(exc))

    def _resolve_version(self, project: Any) -> int:
        """Resolve target project version.

        Args:
            project: Roboflow project object.

        Returns:
            int: Version number to download.
        """

        if self._version is not None:
            return int(self._version)

        versions_method = getattr(project, "versions", None)
        if callable(versions_method):
            try:
                versions = versions_method()
                candidates: list[int] = []
                if isinstance(versions, dict):
                    iterable = versions.values()
                else:
                    iterable = versions if isinstance(versions, Iterable) else []

                for item in iterable:
                    if isinstance(item, dict):
                        value = item.get("id") or item.get("version")
                    else:
                        value = getattr(item, "id", None) or getattr(item, "version", None)
                    if value is None:
                        continue
                    try:
                        candidates.append(int(value))
                    except Exception:
                        continue

                if candidates:
                    return max(candidates)
            except Exception:
                LOGGER.debug("Could not resolve project versions, defaulting to v1.", exc_info=True)

        return 1


class HuggingFaceSearchWorker(QThread):
    """Background worker for Hugging Face model search."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, token: str, query: str, page: int, page_size: int = 8, parent: Any | None = None) -> None:
        """Initialize search worker state.

        Args:
            token: Hugging Face token (can be blank for public queries).
            query: Search text.
            page: 1-based page index.
            page_size: Number of result cards per page.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._token = token.strip()
        self._query = query.strip() or "yolo"
        self._page = max(1, page)
        self._page_size = max(1, page_size)

    def run(self) -> None:
        """Execute model search and emit normalized card payload."""

        try:
            self.progress.emit(10)
            self.status.emit("Connecting to Hugging Face Hub...")

            try:
                from huggingface_hub import HfApi
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("huggingface_hub package is not installed.") from exc

            api = HfApi(token=self._token or None)

            # Fetch enough results to support paging in the UI.
            fetch_limit = min(500, self._page * self._page_size * 2)

            self.progress.emit(35)
            self.status.emit("Searching models tagged for YOLO...")
            models = list(
                api.list_models(
                    search=self._query,
                    sort="downloads",
                    direction=-1,
                    limit=fetch_limit,
                    full=True,
                )
            )

            filtered: list[dict[str, Any]] = []
            for model in models:
                model_id = str(getattr(model, "modelId", ""))
                tags = [str(tag).lower() for tag in (getattr(model, "tags", None) or [])]
                if "yolo" not in model_id.lower() and "yolo" not in tags:
                    continue

                filtered.append(
                    {
                        "repo_id": model_id,
                        "model_name": model_id,
                        "author": str(getattr(model, "author", "-")),
                        "downloads": _to_int(getattr(model, "downloads", None)),
                        "likes": _to_int(getattr(model, "likes", None)),
                        "last_updated": _normalize_datetime_text(getattr(model, "lastModified", None)),
                        "task": str(getattr(model, "pipeline_tag", None) or "-")
                        if getattr(model, "pipeline_tag", None)
                        else "-",
                    }
                )

            start = (self._page - 1) * self._page_size
            end = start + self._page_size
            page_items = filtered[start:end]
            has_next = end < len(filtered)

            self.progress.emit(100)
            self.status.emit("Hugging Face search complete.")
            self.finished.emit(
                {
                    "results": page_items,
                    "page": self._page,
                    "page_size": self._page_size,
                    "has_next": has_next,
                    "total": len(filtered),
                }
            )
        except Exception as exc:
            LOGGER.exception("Hugging Face search failed.")
            self.error.emit(str(exc))


class HuggingFaceModelDownloadWorker(QThread):
    """Background worker that downloads model weights from Hugging Face Hub."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, token: str, repo_id: str, cache_root: Path, parent: Any | None = None) -> None:
        """Initialize model download worker state.

        Args:
            token: Hugging Face token.
            repo_id: Hub repository ID.
            cache_root: Root directory used for model cache.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._token = token.strip()
        self._repo_id = repo_id.strip()
        self._cache_root = cache_root

    def run(self) -> None:
        """Download weight files and emit downloaded path payload."""

        try:
            if not self._repo_id:
                raise RuntimeError("repo_id is required.")

            try:
                from huggingface_hub import HfApi, hf_hub_download, snapshot_download
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("huggingface_hub package is not installed.") from exc

            self._cache_root.mkdir(parents=True, exist_ok=True)
            local_repo_dir = self._cache_root / self._repo_id.replace("/", "__")
            local_repo_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(20)
            self.status.emit("Inspecting repository files...")

            api = HfApi(token=self._token or None)
            files = api.list_repo_files(repo_id=self._repo_id)
            candidate = _pick_weight_file(files)

            downloaded_path: Path | None = None

            if candidate is not None:
                self.status.emit(f"Downloading weight file: {candidate}")
                self.progress.emit(60)
                downloaded = hf_hub_download(
                    repo_id=self._repo_id,
                    filename=candidate,
                    token=self._token or None,
                    local_dir=str(local_repo_dir),
                    local_dir_use_symlinks=False,
                )
                downloaded_path = Path(downloaded).resolve()
            else:
                self.status.emit("No direct weight file found. Downloading repository snapshot...")
                self.progress.emit(55)
                snapshot_dir = snapshot_download(
                    repo_id=self._repo_id,
                    token=self._token or None,
                    local_dir=str(local_repo_dir),
                    local_dir_use_symlinks=False,
                )
                downloaded_path = _find_weight_file_in_tree(Path(snapshot_dir))

            if downloaded_path is None or not downloaded_path.exists():
                raise RuntimeError("No supported weight file (.pt/.onnx/.engine/.tflite) found in repository.")

            self.progress.emit(100)
            self.status.emit("Model download complete.")
            self.finished.emit(
                {
                    "repo_id": self._repo_id,
                    "weights_path": str(downloaded_path),
                }
            )
        except Exception as exc:
            LOGGER.exception("Hugging Face model download failed.")
            self.error.emit(str(exc))


class HuggingFaceDatasetImportWorker(QThread):
    """Background worker that imports Hugging Face datasets and maps them to YOLO format."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, token: str, repo_id: str, cache_root: Path, parent: Any | None = None) -> None:
        """Initialize dataset import worker state.

        Args:
            token: Hugging Face token.
            repo_id: Dataset repository ID.
            cache_root: Root directory used for dataset cache.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)
        self._token = token.strip()
        self._repo_id = repo_id.strip()
        self._cache_root = cache_root

    def run(self) -> None:
        """Download and convert dataset assets into YOLO-compatible structure."""

        try:
            if not self._repo_id:
                raise RuntimeError("repo_id is required.")

            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("huggingface_hub package is not installed.") from exc

            self._cache_root.mkdir(parents=True, exist_ok=True)
            local_repo_dir = self._cache_root / self._repo_id.replace("/", "__")
            local_repo_dir.mkdir(parents=True, exist_ok=True)

            self.progress.emit(20)
            self.status.emit("Downloading dataset snapshot...")
            snapshot_dir = Path(
                snapshot_download(
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    token=self._token or None,
                    local_dir=str(local_repo_dir),
                    local_dir_use_symlinks=False,
                )
            ).resolve()

            self.progress.emit(60)
            self.status.emit("Mapping dataset to YOLO format...")
            mapped_path, class_names, num_images = _map_hf_snapshot_to_yolo(snapshot_dir, local_repo_dir)

            self.progress.emit(100)
            self.status.emit("Dataset import complete.")
            self.finished.emit(
                {
                    "repo_id": self._repo_id,
                    "dataset_path": str(mapped_path.resolve()),
                    "class_names": class_names,
                    "num_images": num_images,
                }
            )
        except Exception as exc:
            LOGGER.exception("Hugging Face dataset import failed.")
            self.error.emit(str(exc))


class DiscoverTab(QWidget):
    """Tab for discovering remote datasets/models and importing them locally."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the discover tab.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._config = _load_config(CONFIG_PATH)

        self._rf_current_page = 1
        self._rf_page_size = 8
        self._rf_has_next = False

        self._hf_current_page = 1
        self._hf_page_size = 8
        self._hf_has_next = False

        self._downloaded_hf_weights: dict[str, str] = {}

        self._rf_search_worker: RoboflowSearchWorker | None = None
        self._rf_download_worker: RoboflowDownloadWorker | None = None
        self._hf_search_worker: HuggingFaceSearchWorker | None = None
        self._hf_model_download_worker: HuggingFaceModelDownloadWorker | None = None
        self._hf_dataset_import_worker: HuggingFaceDatasetImportWorker | None = None

        self._stacked_sections: QStackedWidget

        self._rf_api_key_input: QLineEdit
        self._rf_search_input: QLineEdit
        self._rf_status_label: QLabel
        self._rf_progress_bar: QProgressBar
        self._rf_results_layout: QVBoxLayout
        self._rf_prev_button: QPushButton
        self._rf_next_button: QPushButton
        self._rf_page_label: QLabel

        self._hf_token_input: QLineEdit
        self._hf_search_input: QLineEdit
        self._hf_status_label: QLabel
        self._hf_progress_bar: QProgressBar
        self._hf_results_layout: QVBoxLayout
        self._hf_prev_button: QPushButton
        self._hf_next_button: QPushButton
        self._hf_page_label: QLabel

        self._build_ui()
        self._restore_credentials()

    def _build_ui(self) -> None:
        """Compose segmented top bar and stacked discover sections."""

        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        segment_bar = QWidget(self)
        segment_layout = QHBoxLayout()
        segment_layout.setContentsMargins(0, 0, 0, 0)
        segment_layout.setSpacing(8)

        rf_button = QPushButton("Roboflow Universe", segment_bar)
        hf_button = QPushButton("Hugging Face Hub", segment_bar)

        for button in (rf_button, hf_button):
            button.setCheckable(True)
            button.setProperty("secondary", True)
            button.setMinimumHeight(34)

        rf_button.setChecked(True)

        group = QButtonGroup(segment_bar)
        group.setExclusive(True)
        group.addButton(rf_button, 0)
        group.addButton(hf_button, 1)
        group.idClicked.connect(self._switch_section)

        segment_layout.addWidget(rf_button)
        segment_layout.addWidget(hf_button)
        segment_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        segment_bar.setLayout(segment_layout)

        self._stacked_sections = QStackedWidget(self)
        self._stacked_sections.addWidget(self._build_roboflow_section())
        self._stacked_sections.addWidget(self._build_huggingface_section())

        root.addWidget(segment_bar)
        root.addWidget(self._stacked_sections)
        self.setLayout(root)

    def _build_roboflow_section(self) -> QWidget:
        """Create Roboflow discover section."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        key_row = QWidget(panel)
        key_layout = QHBoxLayout()
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)

        self._rf_api_key_input = QLineEdit(key_row)
        self._rf_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._rf_api_key_input.setPlaceholderText("Roboflow API key")

        save_key_button = QPushButton("Save Key", key_row)
        save_key_button.setProperty("secondary", True)
        save_key_button.clicked.connect(self._save_credentials)

        key_layout.addWidget(QLabel("API Key", key_row))
        key_layout.addWidget(self._rf_api_key_input, stretch=1)
        key_layout.addWidget(save_key_button)
        key_row.setLayout(key_layout)

        search_row = QWidget(panel)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(8)

        self._rf_search_input = QLineEdit(search_row)
        self._rf_search_input.setPlaceholderText("Search projects (or use workspace/project)")
        self._rf_search_input.returnPressed.connect(lambda: self._start_roboflow_search(reset_page=True))

        search_button = QPushButton("Search", search_row)
        search_button.clicked.connect(lambda: self._start_roboflow_search(reset_page=True))

        search_layout.addWidget(QLabel("Search", search_row))
        search_layout.addWidget(self._rf_search_input, stretch=1)
        search_layout.addWidget(search_button)
        search_row.setLayout(search_layout)

        self._rf_status_label = QLabel("Enter query and search Roboflow projects.", panel)
        self._rf_status_label.setProperty("role", "subtle")

        self._rf_progress_bar = QProgressBar(panel)
        self._rf_progress_bar.setRange(0, 100)
        self._rf_progress_bar.setValue(0)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        results_container = QWidget(scroll)
        self._rf_results_layout = QVBoxLayout()
        self._rf_results_layout.setContentsMargins(0, 0, 0, 0)
        self._rf_results_layout.setSpacing(8)
        results_container.setLayout(self._rf_results_layout)
        scroll.setWidget(results_container)

        pagination_row = QWidget(panel)
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(8)

        self._rf_prev_button = QPushButton("Previous", pagination_row)
        self._rf_prev_button.setProperty("secondary", True)
        self._rf_prev_button.clicked.connect(self._goto_prev_roboflow_page)

        self._rf_next_button = QPushButton("Next", pagination_row)
        self._rf_next_button.setProperty("secondary", True)
        self._rf_next_button.clicked.connect(self._goto_next_roboflow_page)

        self._rf_page_label = QLabel("Page 1", pagination_row)
        self._rf_page_label.setObjectName("metricValue")

        pagination_layout.addWidget(self._rf_prev_button)
        pagination_layout.addWidget(self._rf_next_button)
        pagination_layout.addWidget(self._rf_page_label)
        pagination_layout.addStretch(1)
        pagination_row.setLayout(pagination_layout)

        layout.addWidget(key_row)
        layout.addWidget(search_row)
        layout.addWidget(self._rf_status_label)
        layout.addWidget(self._rf_progress_bar)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(pagination_row)
        panel.setLayout(layout)

        self._render_empty_state(self._rf_results_layout, "No Roboflow results yet.")
        self._update_roboflow_pagination_controls()
        return panel

    def _build_huggingface_section(self) -> QWidget:
        """Create Hugging Face discover section."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        token_row = QWidget(panel)
        token_layout = QHBoxLayout()
        token_layout.setContentsMargins(0, 0, 0, 0)
        token_layout.setSpacing(8)

        self._hf_token_input = QLineEdit(token_row)
        self._hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token_input.setPlaceholderText("Hugging Face token (optional for public assets)")

        save_token_button = QPushButton("Save Token", token_row)
        save_token_button.setProperty("secondary", True)
        save_token_button.clicked.connect(self._save_credentials)

        token_layout.addWidget(QLabel("Token", token_row))
        token_layout.addWidget(self._hf_token_input, stretch=1)
        token_layout.addWidget(save_token_button)
        token_row.setLayout(token_layout)

        search_row = QWidget(panel)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(8)

        self._hf_search_input = QLineEdit(search_row)
        self._hf_search_input.setPlaceholderText("Search Hugging Face models (filtered to YOLO)")
        self._hf_search_input.setText("yolo")
        self._hf_search_input.returnPressed.connect(lambda: self._start_hf_search(reset_page=True))

        search_button = QPushButton("Search", search_row)
        search_button.clicked.connect(lambda: self._start_hf_search(reset_page=True))

        search_layout.addWidget(QLabel("Search", search_row))
        search_layout.addWidget(self._hf_search_input, stretch=1)
        search_layout.addWidget(search_button)
        search_row.setLayout(search_layout)

        self._hf_status_label = QLabel("Search Hugging Face models to begin.", panel)
        self._hf_status_label.setProperty("role", "subtle")

        self._hf_progress_bar = QProgressBar(panel)
        self._hf_progress_bar.setRange(0, 100)
        self._hf_progress_bar.setValue(0)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        results_container = QWidget(scroll)
        self._hf_results_layout = QVBoxLayout()
        self._hf_results_layout.setContentsMargins(0, 0, 0, 0)
        self._hf_results_layout.setSpacing(8)
        results_container.setLayout(self._hf_results_layout)
        scroll.setWidget(results_container)

        pagination_row = QWidget(panel)
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(8)

        self._hf_prev_button = QPushButton("Previous", pagination_row)
        self._hf_prev_button.setProperty("secondary", True)
        self._hf_prev_button.clicked.connect(self._goto_prev_hf_page)

        self._hf_next_button = QPushButton("Next", pagination_row)
        self._hf_next_button.setProperty("secondary", True)
        self._hf_next_button.clicked.connect(self._goto_next_hf_page)

        self._hf_page_label = QLabel("Page 1", pagination_row)
        self._hf_page_label.setObjectName("metricValue")

        pagination_layout.addWidget(self._hf_prev_button)
        pagination_layout.addWidget(self._hf_next_button)
        pagination_layout.addWidget(self._hf_page_label)
        pagination_layout.addStretch(1)
        pagination_row.setLayout(pagination_layout)

        layout.addWidget(token_row)
        layout.addWidget(search_row)
        layout.addWidget(self._hf_status_label)
        layout.addWidget(self._hf_progress_bar)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(pagination_row)
        panel.setLayout(layout)

        self._render_empty_state(self._hf_results_layout, "No Hugging Face results yet.")
        self._update_hf_pagination_controls()
        return panel

    def _switch_section(self, section_index: int) -> None:
        """Switch between Roboflow and Hugging Face sections.

        Args:
            section_index: Index into stacked sections.
        """

        self._stacked_sections.setCurrentIndex(max(0, min(section_index, self._stacked_sections.count() - 1)))

    def _restore_credentials(self) -> None:
        """Restore API credentials from local config file."""

        self._rf_api_key_input.setText(str(self._config.get("roboflow_api_key", "")))
        self._hf_token_input.setText(str(self._config.get("huggingface_token", "")))

    def _save_credentials(self) -> None:
        """Persist current API credentials into local config file."""

        self._config["roboflow_api_key"] = self._rf_api_key_input.text().strip()
        self._config["huggingface_token"] = self._hf_token_input.text().strip()
        _save_config(CONFIG_PATH, self._config)

        self._rf_status_label.setText("Roboflow API key saved to config.json")
        self._hf_status_label.setText("Hugging Face token saved to config.json")

    def _start_roboflow_search(self, reset_page: bool) -> None:
        """Start Roboflow search worker for current query/page.

        Args:
            reset_page: Whether to reset paging to page 1.
        """

        if self._rf_search_worker is not None and self._rf_search_worker.isRunning():
            QMessageBox.information(self, "Search Running", "Roboflow search is already running.")
            return

        query = self._rf_search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Validation", "Enter a Roboflow search query.")
            return

        api_key = self._rf_api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Validation", "Roboflow API key is required.")
            return

        if reset_page:
            self._rf_current_page = 1

        self._rf_progress_bar.setValue(0)
        self._rf_status_label.setText("Searching Roboflow...")

        self._rf_search_worker = RoboflowSearchWorker(
            api_key=api_key,
            query=query,
            page=self._rf_current_page,
            page_size=self._rf_page_size,
            parent=self,
        )
        self._rf_search_worker.progress.connect(self._rf_progress_bar.setValue)
        self._rf_search_worker.status.connect(self._rf_status_label.setText)
        self._rf_search_worker.finished.connect(self._on_roboflow_search_finished)
        self._rf_search_worker.error.connect(self._on_roboflow_search_error)
        self._rf_search_worker.start()

    def _on_roboflow_search_finished(self, payload: dict[str, Any]) -> None:
        """Render Roboflow search results.

        Args:
            payload: Search result payload.
        """

        self._rf_has_next = bool(payload.get("has_next", False))
        total = payload.get("total")
        results = payload.get("results") or []

        self._render_roboflow_results(results)

        if total is None:
            self._rf_status_label.setText(f"Roboflow results: {len(results)} on page {self._rf_current_page}")
        else:
            self._rf_status_label.setText(
                f"Roboflow results: page {self._rf_current_page}, showing {len(results)} of {total}"
            )

        self._rf_progress_bar.setValue(100)
        self._update_roboflow_pagination_controls()

        self._rf_search_worker = None

    def _on_roboflow_search_error(self, message: str) -> None:
        """Handle Roboflow search errors.

        Args:
            message: Error text.
        """

        self._rf_status_label.setText(f"Roboflow search failed: {message}")
        self._rf_progress_bar.setValue(0)
        QMessageBox.critical(self, "Roboflow Search Error", message)
        self._rf_search_worker = None

    def _render_roboflow_results(self, results: list[dict[str, Any]]) -> None:
        """Render Roboflow project cards.

        Args:
            results: Normalized result records.
        """

        _clear_layout(self._rf_results_layout)

        if not results:
            self._render_empty_state(self._rf_results_layout, "No Roboflow results found.")
            return

        for item in results:
            card = QFrame(self)
            card.setProperty("card", True)
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)

            title = QLabel(item.get("project_name") or item.get("project_slug") or "Unnamed Project", card)
            title.setStyleSheet("font-size: 14px; font-weight: 600;")

            owner = item.get("owner") or item.get("workspace") or "-"
            meta_line = QLabel(
                (
                    f"Owner: {owner} | Task: {item.get('task_type', '-')} | "
                    f"Images: {item.get('image_count', '-')} | Classes: {item.get('class_count', '-')}"
                ),
                card,
            )
            meta_line.setProperty("role", "subtle")

            license_label = QLabel(f"License: {item.get('license', '-')}", card)
            license_label.setProperty("role", "subtle")

            actions = QWidget(card)
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(0, 0, 0, 0)
            actions_layout.setSpacing(8)

            download_button = QPushButton("Download Dataset", actions)
            download_button.clicked.connect(
                lambda _checked=False, record=item: self._start_roboflow_download(record)
            )

            actions_layout.addWidget(download_button)
            actions_layout.addStretch(1)
            actions.setLayout(actions_layout)

            card_layout.addWidget(title)
            card_layout.addWidget(meta_line)
            card_layout.addWidget(license_label)
            card_layout.addWidget(actions)
            card.setLayout(card_layout)

            self._rf_results_layout.addWidget(card)

        self._rf_results_layout.addStretch(1)

    def _start_roboflow_download(self, record: dict[str, Any]) -> None:
        """Start dataset download from a Roboflow result card.

        Args:
            record: Result record backing the selected card.
        """

        if self._rf_download_worker is not None and self._rf_download_worker.isRunning():
            QMessageBox.information(self, "Download Running", "A Roboflow download is already in progress.")
            return

        workspace = str(record.get("workspace") or record.get("owner") or "").strip()
        project_slug = str(record.get("project_slug") or "").strip()

        if not workspace or not project_slug:
            QMessageBox.warning(
                self,
                "Missing Metadata",
                "Selected result does not include workspace/project slugs required for download.",
            )
            return

        api_key = self._rf_api_key_input.text().strip()
        version = _to_int(record.get("version"))

        self._rf_progress_bar.setValue(0)
        self._rf_status_label.setText("Starting Roboflow dataset download...")

        self._rf_download_worker = RoboflowDownloadWorker(
            api_key=api_key,
            workspace_slug=workspace,
            project_slug=project_slug,
            version=version,
            download_root=ROBOFLOW_DOWNLOAD_ROOT,
            parent=self,
        )
        self._rf_download_worker.progress.connect(self._rf_progress_bar.setValue)
        self._rf_download_worker.status.connect(self._rf_status_label.setText)
        self._rf_download_worker.finished.connect(self._on_roboflow_download_finished)
        self._rf_download_worker.error.connect(self._on_roboflow_download_error)
        self._rf_download_worker.start()

    def _on_roboflow_download_finished(self, payload: dict[str, Any]) -> None:
        """Handle successful Roboflow dataset download.

        Args:
            payload: Download payload.
        """

        try:
            workspace = str(payload.get("workspace", ""))
            project_slug = str(payload.get("project_slug", ""))
            dataset_name = f"{workspace}/{project_slug}".strip("/") or "Roboflow Dataset"

            self._upsert_dataset_record(
                name=dataset_name,
                source=DatasetSource.ROBOFLOW,
                local_path=str(payload.get("dataset_path")),
                class_names=list(payload.get("class_names") or []),
                num_images=_to_int(payload.get("num_images")) or 0,
                tags=["roboflow", workspace, project_slug],
                description=f"Imported from Roboflow project {workspace}/{project_slug}",
                roboflow_project_id=f"{workspace}/{project_slug}",
                hf_repo_id=None,
            )

            self._rf_status_label.setText("Roboflow dataset downloaded and registered in SQLite.")
            self._rf_progress_bar.setValue(100)
            QMessageBox.information(
                self,
                "Download Complete",
                f"Dataset downloaded to:\n{payload.get('dataset_path')}\n\n"
                "The dataset has been registered in the local library.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Registration Error", f"Dataset download succeeded but DB registration failed: {exc}")
        finally:
            self._rf_download_worker = None

    def _on_roboflow_download_error(self, message: str) -> None:
        """Handle Roboflow dataset download errors.

        Args:
            message: Error text.
        """

        self._rf_status_label.setText(f"Roboflow download failed: {message}")
        self._rf_progress_bar.setValue(0)
        QMessageBox.critical(self, "Roboflow Download Error", message)
        self._rf_download_worker = None

    def _goto_prev_roboflow_page(self) -> None:
        """Navigate to previous Roboflow results page."""

        if self._rf_current_page <= 1:
            return
        self._rf_current_page -= 1
        self._start_roboflow_search(reset_page=False)

    def _goto_next_roboflow_page(self) -> None:
        """Navigate to next Roboflow results page."""

        if not self._rf_has_next:
            return
        self._rf_current_page += 1
        self._start_roboflow_search(reset_page=False)

    def _update_roboflow_pagination_controls(self) -> None:
        """Refresh Roboflow pagination widget state."""

        self._rf_page_label.setText(f"Page {self._rf_current_page}")
        self._rf_prev_button.setEnabled(self._rf_current_page > 1)
        self._rf_next_button.setEnabled(self._rf_has_next)

    def _start_hf_search(self, reset_page: bool) -> None:
        """Start Hugging Face search worker.

        Args:
            reset_page: Whether to reset paging to page 1.
        """

        if self._hf_search_worker is not None and self._hf_search_worker.isRunning():
            QMessageBox.information(self, "Search Running", "Hugging Face search is already running.")
            return

        query = self._hf_search_input.text().strip() or "yolo"
        if reset_page:
            self._hf_current_page = 1

        self._hf_progress_bar.setValue(0)
        self._hf_status_label.setText("Searching Hugging Face models...")

        self._hf_search_worker = HuggingFaceSearchWorker(
            token=self._hf_token_input.text().strip(),
            query=query,
            page=self._hf_current_page,
            page_size=self._hf_page_size,
            parent=self,
        )
        self._hf_search_worker.progress.connect(self._hf_progress_bar.setValue)
        self._hf_search_worker.status.connect(self._hf_status_label.setText)
        self._hf_search_worker.finished.connect(self._on_hf_search_finished)
        self._hf_search_worker.error.connect(self._on_hf_search_error)
        self._hf_search_worker.start()

    def _on_hf_search_finished(self, payload: dict[str, Any]) -> None:
        """Render Hugging Face search results.

        Args:
            payload: Search result payload.
        """

        self._hf_has_next = bool(payload.get("has_next", False))
        total = payload.get("total")
        results = payload.get("results") or []

        self._render_hf_results(results)

        self._hf_progress_bar.setValue(100)
        if total is None:
            self._hf_status_label.setText(f"Hugging Face results: {len(results)} on page {self._hf_current_page}")
        else:
            self._hf_status_label.setText(
                f"Hugging Face results: page {self._hf_current_page}, showing {len(results)} of {total}"
            )

        self._update_hf_pagination_controls()
        self._hf_search_worker = None

    def _on_hf_search_error(self, message: str) -> None:
        """Handle Hugging Face search errors.

        Args:
            message: Error text.
        """

        self._hf_status_label.setText(f"Hugging Face search failed: {message}")
        self._hf_progress_bar.setValue(0)
        QMessageBox.critical(self, "Hugging Face Search Error", message)
        self._hf_search_worker = None

    def _render_hf_results(self, results: list[dict[str, Any]]) -> None:
        """Render Hugging Face model cards.

        Args:
            results: Normalized result records.
        """

        _clear_layout(self._hf_results_layout)

        if not results:
            self._render_empty_state(self._hf_results_layout, "No Hugging Face models found.")
            return

        for item in results:
            card = QFrame(self)
            card.setProperty("card", True)
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)

            title = QLabel(item.get("model_name") or item.get("repo_id") or "Unnamed Model", card)
            title.setStyleSheet("font-size: 14px; font-weight: 600;")

            meta_line = QLabel(
                (
                    f"Author: {item.get('author', '-')} | Task: {item.get('task', '-')} | "
                    f"Downloads: {item.get('downloads', '-')} | Likes: {item.get('likes', '-')}"
                ),
                card,
            )
            meta_line.setProperty("role", "subtle")

            updated_line = QLabel(f"Last Updated: {item.get('last_updated', '-')}", card)
            updated_line.setProperty("role", "subtle")

            action_row = QWidget(card)
            action_layout = QHBoxLayout()
            action_layout.setContentsMargins(0, 0, 0, 0)
            action_layout.setSpacing(8)

            download_button = QPushButton("Download Model", action_row)
            download_button.clicked.connect(
                lambda _checked=False, record=item: self._start_hf_model_download(record)
            )

            register_button = QPushButton("Register as Base Weights", action_row)
            register_button.setProperty("secondary", True)
            register_button.clicked.connect(
                lambda _checked=False, record=item: self._register_hf_base_weight(record)
            )

            import_button = QPushButton("Import Dataset", action_row)
            import_button.clicked.connect(
                lambda _checked=False, record=item: self._start_hf_dataset_import(record)
            )

            action_layout.addWidget(download_button)
            action_layout.addWidget(register_button)
            action_layout.addWidget(import_button)
            action_layout.addStretch(1)
            action_row.setLayout(action_layout)

            card_layout.addWidget(title)
            card_layout.addWidget(meta_line)
            card_layout.addWidget(updated_line)
            card_layout.addWidget(action_row)
            card.setLayout(card_layout)

            self._hf_results_layout.addWidget(card)

        self._hf_results_layout.addStretch(1)

    def _start_hf_model_download(self, record: dict[str, Any]) -> None:
        """Start model download worker for selected Hugging Face model.

        Args:
            record: Result record.
        """

        if self._hf_model_download_worker is not None and self._hf_model_download_worker.isRunning():
            QMessageBox.information(self, "Download Running", "A Hugging Face model download is already in progress.")
            return

        repo_id = str(record.get("repo_id") or "").strip()
        if not repo_id:
            QMessageBox.warning(self, "Missing Metadata", "Selected record does not include repo_id.")
            return

        self._hf_progress_bar.setValue(0)
        self._hf_status_label.setText(f"Downloading model {repo_id}...")

        self._hf_model_download_worker = HuggingFaceModelDownloadWorker(
            token=self._hf_token_input.text().strip(),
            repo_id=repo_id,
            cache_root=HF_MODEL_CACHE_ROOT,
            parent=self,
        )
        self._hf_model_download_worker.progress.connect(self._hf_progress_bar.setValue)
        self._hf_model_download_worker.status.connect(self._hf_status_label.setText)
        self._hf_model_download_worker.finished.connect(self._on_hf_model_download_finished)
        self._hf_model_download_worker.error.connect(self._on_hf_model_download_error)
        self._hf_model_download_worker.start()

    def _on_hf_model_download_finished(self, payload: dict[str, Any]) -> None:
        """Handle successful Hugging Face model download.

        Args:
            payload: Download payload.
        """

        try:
            repo_id = str(payload.get("repo_id") or "")
            weights_path = str(payload.get("weights_path") or "")
            if repo_id and weights_path:
                self._downloaded_hf_weights[repo_id] = weights_path

            self._hf_status_label.setText("Model download complete.")
            self._hf_progress_bar.setValue(100)
            QMessageBox.information(
                self,
                "Model Downloaded",
                f"Model repository: {repo_id}\n"
                f"Weights path: {weights_path}",
            )
        finally:
            self._hf_model_download_worker = None

    def _on_hf_model_download_error(self, message: str) -> None:
        """Handle Hugging Face model download errors.

        Args:
            message: Error text.
        """

        self._hf_status_label.setText(f"Model download failed: {message}")
        self._hf_progress_bar.setValue(0)
        QMessageBox.critical(self, "Model Download Error", message)
        self._hf_model_download_worker = None

    def _register_hf_base_weight(self, record: dict[str, Any]) -> None:
        """Register selected or downloaded model path into BaseWeight table.

        Args:
            record: Result record.
        """

        repo_id = str(record.get("repo_id") or "").strip()
        if not repo_id:
            QMessageBox.warning(self, "Missing Metadata", "Selected record does not include repo_id.")
            return

        weights_path = self._downloaded_hf_weights.get(repo_id)
        if not weights_path or not Path(weights_path).exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Select Weight File",
                str(HF_MODEL_CACHE_ROOT),
                "Weights (*.pt *.onnx *.engine *.tflite)",
            )
            if not chosen:
                return
            weights_path = str(Path(chosen).resolve())

        local_path = Path(weights_path)
        if not local_path.exists():
            QMessageBox.warning(self, "Missing File", "Selected weight file does not exist.")
            return

        session = get_session()
        try:
            existing = (
                session.query(BaseWeight)
                .filter(BaseWeight.repo_id == repo_id)
                .order_by(BaseWeight.updated_at.desc())
                .first()
            )

            if existing is None:
                existing = BaseWeight(
                    name=str(record.get("model_name") or repo_id),
                    source=BaseWeightSource.HUGGINGFACE,
                    repo_id=repo_id,
                    local_path=str(local_path),
                    task=str(record.get("task") or "") or None,
                    downloads=_to_int(record.get("downloads")),
                    likes=_to_int(record.get("likes")),
                    last_updated=_parse_datetime(record.get("last_updated")),
                    notes="Registered from Discover tab.",
                    tags=["huggingface", "yolo"],
                )
                session.add(existing)
            else:
                existing.local_path = str(local_path)
                existing.task = str(record.get("task") or "") or existing.task
                existing.downloads = _to_int(record.get("downloads"))
                existing.likes = _to_int(record.get("likes"))
                parsed_updated = _parse_datetime(record.get("last_updated"))
                if parsed_updated is not None:
                    existing.last_updated = parsed_updated
                existing.updated_at = datetime.now(timezone.utc)

            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Register Failed", f"Could not register base weights: {exc}")
            return
        finally:
            session.close()

        QMessageBox.information(
            self,
            "Registered",
            f"Base weights registered in SQLite:\n{local_path}",
        )

    def _start_hf_dataset_import(self, record: dict[str, Any]) -> None:
        """Start dataset import worker from selected Hugging Face repo.

        Args:
            record: Result record.
        """

        if self._hf_dataset_import_worker is not None and self._hf_dataset_import_worker.isRunning():
            QMessageBox.information(self, "Import Running", "A Hugging Face dataset import is already in progress.")
            return

        repo_id = str(record.get("repo_id") or "").strip()
        if not repo_id:
            QMessageBox.warning(self, "Missing Metadata", "Selected record does not include repo_id.")
            return

        self._hf_progress_bar.setValue(0)
        self._hf_status_label.setText(f"Importing dataset from {repo_id}...")

        self._hf_dataset_import_worker = HuggingFaceDatasetImportWorker(
            token=self._hf_token_input.text().strip(),
            repo_id=repo_id,
            cache_root=HF_DATASET_CACHE_ROOT,
            parent=self,
        )
        self._hf_dataset_import_worker.progress.connect(self._hf_progress_bar.setValue)
        self._hf_dataset_import_worker.status.connect(self._hf_status_label.setText)
        self._hf_dataset_import_worker.finished.connect(self._on_hf_dataset_import_finished)
        self._hf_dataset_import_worker.error.connect(self._on_hf_dataset_import_error)
        self._hf_dataset_import_worker.start()

    def _on_hf_dataset_import_finished(self, payload: dict[str, Any]) -> None:
        """Handle successful Hugging Face dataset import.

        Args:
            payload: Import payload.
        """

        try:
            repo_id = str(payload.get("repo_id") or "")
            dataset_path = str(payload.get("dataset_path") or "")
            class_names = list(payload.get("class_names") or [])
            num_images = _to_int(payload.get("num_images")) or 0

            self._upsert_dataset_record(
                name=f"HF {repo_id}",
                source=DatasetSource.HUGGINGFACE,
                local_path=dataset_path,
                class_names=class_names,
                num_images=num_images,
                tags=["huggingface", "imported"],
                description=f"Imported from Hugging Face dataset repo {repo_id}",
                roboflow_project_id=None,
                hf_repo_id=repo_id,
            )

            self._hf_progress_bar.setValue(100)
            self._hf_status_label.setText("Dataset import complete and registered in SQLite.")
            QMessageBox.information(
                self,
                "Dataset Imported",
                f"Dataset path: {dataset_path}\n"
                "The dataset has been registered in the library.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Registration Error", f"Dataset import succeeded but DB registration failed: {exc}")
        finally:
            self._hf_dataset_import_worker = None

    def _on_hf_dataset_import_error(self, message: str) -> None:
        """Handle Hugging Face dataset import errors.

        Args:
            message: Error text.
        """

        self._hf_progress_bar.setValue(0)
        self._hf_status_label.setText(f"Dataset import failed: {message}")
        QMessageBox.critical(self, "Dataset Import Error", message)
        self._hf_dataset_import_worker = None

    def _goto_prev_hf_page(self) -> None:
        """Navigate to previous Hugging Face results page."""

        if self._hf_current_page <= 1:
            return
        self._hf_current_page -= 1
        self._start_hf_search(reset_page=False)

    def _goto_next_hf_page(self) -> None:
        """Navigate to next Hugging Face results page."""

        if not self._hf_has_next:
            return
        self._hf_current_page += 1
        self._start_hf_search(reset_page=False)

    def _update_hf_pagination_controls(self) -> None:
        """Refresh Hugging Face pagination widget state."""

        self._hf_page_label.setText(f"Page {self._hf_current_page}")
        self._hf_prev_button.setEnabled(self._hf_current_page > 1)
        self._hf_next_button.setEnabled(self._hf_has_next)

    @staticmethod
    def _render_empty_state(layout: QVBoxLayout, message: str) -> None:
        """Insert a placeholder card into a results layout.

        Args:
            layout: Target layout.
            message: Placeholder text.
        """

        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(10, 10, 10, 10)

        label = QLabel(message, card)
        label.setProperty("role", "subtle")
        card_layout.addWidget(label)
        card.setLayout(card_layout)

        layout.addWidget(card)
        layout.addStretch(1)

    def _upsert_dataset_record(
        self,
        name: str,
        source: DatasetSource,
        local_path: str,
        class_names: list[str],
        num_images: int,
        tags: list[str],
        description: str,
        roboflow_project_id: str | None,
        hf_repo_id: str | None,
    ) -> None:
        """Insert or update a dataset metadata record.

        Args:
            name: Dataset display name.
            source: Dataset source enum.
            local_path: Dataset root path on disk.
            class_names: Dataset class names.
            num_images: Number of images in dataset.
            tags: Dataset tags.
            description: Dataset description.
            roboflow_project_id: Optional Roboflow project identifier.
            hf_repo_id: Optional Hugging Face repo identifier.
        """

        dataset_path = str(Path(local_path).resolve())
        num_classes = len(class_names)

        session = get_session()
        try:
            filters: list[Any] = [Dataset.local_path == dataset_path]
            if hf_repo_id:
                filters.append(Dataset.hf_repo_id == hf_repo_id)
            if roboflow_project_id:
                filters.append(Dataset.roboflow_project_id == roboflow_project_id)

            existing = session.query(Dataset).filter(or_(*filters)).first()

            if existing is None:
                existing = Dataset(
                    name=name,
                    description=description,
                    source=source,
                    roboflow_project_id=roboflow_project_id,
                    hf_repo_id=hf_repo_id,
                    local_path=dataset_path,
                    class_names=class_names,
                    num_images=num_images,
                    num_classes=num_classes,
                    tags=_clean_tags(tags),
                )
                session.add(existing)
            else:
                existing.name = name
                existing.description = description or existing.description
                existing.source = source
                existing.roboflow_project_id = roboflow_project_id
                existing.hf_repo_id = hf_repo_id
                existing.local_path = dataset_path
                existing.class_names = class_names
                existing.num_images = num_images
                existing.num_classes = num_classes
                existing.tags = _clean_tags(tags)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load JSON config file from disk.

    Args:
        config_path: Config file path.

    Returns:
        dict[str, Any]: Parsed config dict or empty dict.
    """

    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        LOGGER.exception("Failed to load config file: %s", config_path)
        return {}


def _save_config(config_path: Path, payload: dict[str, Any]) -> None:
    """Write JSON config file to disk.

    Args:
        config_path: Config file path.
        payload: Config dictionary.
    """

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _clear_layout(layout: QVBoxLayout) -> None:
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
            _clear_layout(child_layout)


def _paginate_roboflow_response(raw: Any, page: int, page_size: int) -> tuple[list[dict[str, Any]], bool, int | None]:
    """Normalize Roboflow raw search response and apply pagination heuristics.

    Args:
        raw: Raw response from SDK call.
        page: 1-based page index.
        page_size: Number of items per page.

    Returns:
        tuple[list[dict[str, Any]], bool, int | None]: Page items, has-next, total if known.
    """

    total: int | None = None
    has_next = False

    if isinstance(raw, dict):
        items = raw.get("results") or raw.get("projects") or raw.get("data") or []
        normalized = _normalize_roboflow_search_items(items)

        total_value = raw.get("total") or raw.get("count")
        total = _to_int(total_value)

        if total is not None:
            has_next = page * page_size < total

        if not has_next:
            has_next = bool(raw.get("next") or raw.get("next_page"))

        if total is None and len(normalized) > page_size:
            start = (page - 1) * page_size
            end = start + page_size
            page_items = normalized[start:end]
            has_next = end < len(normalized)
            return page_items, has_next, len(normalized)

        if total is None:
            has_next = len(normalized) >= page_size

        return normalized, has_next, total

    normalized = _normalize_roboflow_search_items(raw)

    if len(normalized) > page_size:
        start = (page - 1) * page_size
        end = start + page_size
        page_items = normalized[start:end]
        has_next = end < len(normalized)
        return page_items, has_next, len(normalized)

    has_next = len(normalized) >= page_size
    return normalized, has_next, None


def _normalize_roboflow_search_items(raw_items: Any) -> list[dict[str, Any]]:
    """Normalize one or many Roboflow raw items into card dicts.

    Args:
        raw_items: Raw iterable or scalar item.

    Returns:
        list[dict[str, Any]]: Normalized result records.
    """

    if raw_items is None:
        return []

    if isinstance(raw_items, (list, tuple)):
        items = raw_items
    else:
        items = [raw_items]

    results: list[dict[str, Any]] = []
    for item in items:
        normalized = _normalize_roboflow_project(item)
        if normalized is not None:
            results.append(normalized)

    return results


def _normalize_roboflow_project(
    raw_item: Any,
    workspace_slug: str | None = None,
    project_slug: str | None = None,
) -> dict[str, Any] | None:
    """Normalize a Roboflow project object/dict into common fields.

    Args:
        raw_item: Raw project object or dict.
        workspace_slug: Optional forced workspace slug.
        project_slug: Optional forced project slug.

    Returns:
        dict[str, Any] | None: Normalized record if fields are available.
    """

    if raw_item is None:
        return None

    if isinstance(raw_item, dict):
        data = raw_item
    else:
        data = {
            "name": getattr(raw_item, "name", None),
            "id": getattr(raw_item, "id", None),
            "type": getattr(raw_item, "type", None),
            "images": getattr(raw_item, "images", None),
            "classes": getattr(raw_item, "classes", None),
            "license": getattr(raw_item, "license", None),
            "workspace": getattr(raw_item, "workspace", None),
            "project": getattr(raw_item, "project", None),
            "version": getattr(raw_item, "version", None),
            "_raw": getattr(raw_item, "__dict__", {}),
        }

    owner = workspace_slug or str(
        data.get("owner")
        or data.get("workspace")
        or data.get("workspace_slug")
        or data.get("username")
        or ""
    ).strip()

    slug = project_slug or str(
        data.get("project")
        or data.get("slug")
        or data.get("project_slug")
        or data.get("name")
        or ""
    ).strip()

    name = str(data.get("name") or slug or "").strip()
    if not name and not slug:
        return None

    image_count = _to_int(data.get("images") or data.get("num_images") or data.get("image_count"))

    classes = data.get("classes")
    if isinstance(classes, (list, tuple, set)):
        class_count = len(classes)
    elif isinstance(classes, dict):
        class_count = len(classes.keys())
    else:
        class_count = _to_int(data.get("class_count") or data.get("num_classes"))

    license_value = data.get("license")
    if isinstance(license_value, dict):
        license_text = str(license_value.get("name") or license_value.get("value") or "-")
    else:
        license_text = str(license_value or "-")

    task_type = str(data.get("task") or data.get("task_type") or data.get("type") or "-")

    version = _to_int(data.get("version") or data.get("default_version"))

    return {
        "project_name": name,
        "owner": owner or "-",
        "task_type": task_type,
        "image_count": image_count if image_count is not None else "-",
        "class_count": class_count if class_count is not None else "-",
        "license": license_text,
        "workspace": owner,
        "project_slug": slug or _slugify(name),
        "version": version,
    }


def _resolve_downloaded_path(download_result: Any, fallback_dir: Path) -> Path:
    """Resolve downloaded dataset folder path from Roboflow response object.

    Args:
        download_result: Roboflow download response object.
        fallback_dir: Fallback directory when response lacks explicit location.

    Returns:
        Path: Existing dataset directory path.
    """

    location: str | None = None

    if isinstance(download_result, str):
        location = download_result
    else:
        location = getattr(download_result, "location", None)

    if location:
        candidate = Path(location).resolve()
        if candidate.exists():
            if candidate.is_file():
                return candidate.parent
            return candidate

    # Some SDK variants nest dataset under fallback directory.
    if fallback_dir.exists():
        return fallback_dir.resolve()

    raise RuntimeError("Could not resolve downloaded dataset location.")


def _pick_weight_file(files: Iterable[str]) -> str | None:
    """Select preferred model weight file from repository file list.

    Args:
        files: Repository file paths.

    Returns:
        str | None: Best candidate file path.
    """

    priority_ext = [".pt", ".onnx", ".engine", ".tflite"]
    normalized = [str(path) for path in files]

    for ext in priority_ext:
        candidates = [path for path in normalized if path.lower().endswith(ext)]
        if candidates:
            # Prefer top-level files first.
            candidates.sort(key=lambda path: (path.count("/"), len(path)))
            return candidates[0]

    return None


def _find_weight_file_in_tree(root: Path) -> Path | None:
    """Find a weight file recursively in a local folder tree.

    Args:
        root: Root directory.

    Returns:
        Path | None: First preferred weight file path.
    """

    priority_ext = [".pt", ".onnx", ".engine", ".tflite"]
    for ext in priority_ext:
        matches = sorted(root.rglob(f"*{ext}"))
        if matches:
            return matches[0].resolve()
    return None


def _map_hf_snapshot_to_yolo(snapshot_dir: Path, local_repo_dir: Path) -> tuple[Path, list[str], int]:
    """Map Hugging Face dataset snapshot into YOLO structure where possible.

    Args:
        snapshot_dir: Downloaded dataset snapshot directory.
        local_repo_dir: Local cache directory for the repo.

    Returns:
        tuple[Path, list[str], int]: YOLO dataset root, class names, image count.
    """

    yaml_candidates = sorted(snapshot_dir.rglob("data.yaml"))
    if yaml_candidates:
        yolo_root = yaml_candidates[0].parent.resolve()
        class_names = _load_class_names_from_yaml(yolo_root / "data.yaml")
        if not class_names:
            class_names = _infer_class_names_from_labels(yolo_root)
        num_images = _count_images(yolo_root)
        return yolo_root, class_names, num_images

    images = _collect_images(snapshot_dir)
    if not images:
        raise RuntimeError("No image assets were found in the dataset repository.")

    yolo_root = local_repo_dir / "yolo_mapped"
    images_train = yolo_root / "images" / "train"
    labels_train = yolo_root / "labels" / "train"
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image in images:
        target_image = images_train / image.name
        shutil.copy2(image, target_image)
        copied += 1

        source_label = image.with_suffix(".txt")
        target_label = labels_train / f"{image.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, target_label)
        else:
            target_label.write_text("", encoding="utf-8")

    class_names = _infer_class_names_from_labels(yolo_root)
    if not class_names:
        class_names = ["object"]

    yaml_payload = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/train",
        "test": "images/train",
        "names": class_names,
        "nc": len(class_names),
    }
    with (yolo_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(yaml_payload, handle, sort_keys=False)

    return yolo_root.resolve(), class_names, copied


def _collect_images(root: Path) -> list[Path]:
    """Collect supported image files recursively from a root folder.

    Args:
        root: Root directory.

    Returns:
        list[Path]: Image file paths.
    """

    images: list[Path] = []
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file_path.resolve())
    return images


def _load_class_names_from_yaml(yaml_path: Path) -> list[str]:
    """Load class names from YOLO data.yaml if available.

    Args:
        yaml_path: YAML file path.

    Returns:
        list[str]: Class names.
    """

    if not yaml_path.exists():
        return []

    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        LOGGER.debug("Unable to parse data.yaml: %s", yaml_path, exc_info=True)
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
    """Infer synthetic class names from YOLO label files.

    Args:
        dataset_root: Dataset root folder.

    Returns:
        list[str]: Class names inferred from max class index.
    """

    max_class_id = -1
    for label_path in dataset_root.rglob("*.txt"):
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                max_class_id = max(max_class_id, class_id)
        except Exception:
            continue

    if max_class_id < 0:
        return []

    return [f"class_{idx}" for idx in range(max_class_id + 1)]


def _count_images(root: Path) -> int:
    """Count supported image files in a directory tree.

    Args:
        root: Root directory.

    Returns:
        int: Image count.
    """

    return len(_collect_images(root))


def _to_int(value: Any) -> int | None:
    """Convert scalar-like values to integers when possible.

    Args:
        value: Candidate scalar value.

    Returns:
        int | None: Parsed integer value.
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


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug from arbitrary text.

    Args:
        text: Raw text.

    Returns:
        str: Slug value.
    """

    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "asset"


def _clean_tags(tags: Iterable[str]) -> list[str]:
    """Normalize and deduplicate tags preserving insertion order.

    Args:
        tags: Candidate tag values.

    Returns:
        list[str]: Normalized tag list.
    """

    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tags:
        value = str(raw).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _normalize_datetime_text(value: Any) -> str:
    """Normalize datetime-like objects to ISO-like text for card display.

    Args:
        value: Datetime-like value.

    Returns:
        str: Readable datetime text.
    """

    if value is None:
        return "-"

    if isinstance(value, datetime):
        return value.isoformat()

    text = str(value).strip()
    return text or "-"


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime-like values into timezone-aware datetime.

    Args:
        value: Input datetime representation.

    Returns:
        datetime | None: Parsed UTC-aware datetime.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    text = str(value).strip()
    if not text or text == "-":
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


__all__ = [
    "DiscoverTab",
    "RoboflowSearchWorker",
    "RoboflowDownloadWorker",
    "HuggingFaceSearchWorker",
    "HuggingFaceModelDownloadWorker",
    "HuggingFaceDatasetImportWorker",
]

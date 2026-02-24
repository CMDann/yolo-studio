"""Discover tab for searching and importing assets from Roboflow and Hugging Face."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
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
OPEN_IMAGES_CLASSES_PATH = PROJECT_ROOT / "assets" / "openimages_classes.json"


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


class KaggleSearchWorker(QThread):
    """Background worker for Kaggle dataset/competition search."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        username: str,
        api_key: str,
        query: str,
        page: int,
        page_size: int,
        search_type: str,
        sort: str,
        tag: str,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._username = username
        self._api_key = api_key
        self._query = query.strip()
        self._page = max(1, page)
        self._page_size = max(1, page_size)
        self._search_type = search_type
        self._sort = sort
        self._tag = tag.strip()

    def run(self) -> None:
        try:
            _ensure_kaggle_credentials(self._username, self._api_key)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        try:
            from kaggle import api as kaggle_api  # type: ignore
        except Exception as exc:
            self.error.emit(f"Kaggle package not available: {exc}")
            return

        self.status.emit("Searching Kaggle...")
        self.progress.emit(10)

        try:
            if self._search_type == "competitions":
                search_text = self._query or (self._tag or "computer-vision")
                raw = kaggle_api.competitions_list(
                    search=search_text or None,
                    page=self._page,
                    page_size=self._page_size,
                    sort_by=self._sort,
                    category="active",
                )
                data = [_normalize_kaggle_competition(item) for item in raw]
            else:
                search_text = self._query
                if self._tag:
                    search_text = f"{search_text} {self._tag}".strip()
                raw = kaggle_api.dataset_list(
                    search=search_text or None,
                    page=self._page,
                    page_size=self._page_size,
                    file_type="csv",
                    sort_by=self._sort,
                    tag_ids=None,
                )
                data = [_normalize_kaggle_dataset(item) for item in raw]
        except Exception as exc:
            self.error.emit(f"Kaggle search failed: {exc}")
            return

        self.progress.emit(100)
        self.status.emit("Search complete.")
        self.finished.emit(data)


class KaggleDownloadWorker(QThread):
    """Download a Kaggle dataset and register if possible."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        username: str,
        api_key: str,
        dataset_ref: str,
        dest: Path,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._username = username
        self._api_key = api_key
        self._dataset_ref = dataset_ref
        self._dest = dest

    def run(self) -> None:
        try:
            _ensure_kaggle_credentials(self._username, self._api_key)
        except Exception as exc:
            self.error.emit(str(exc))
            return

        try:
            from kaggle import api as kaggle_api  # type: ignore
        except Exception as exc:
            self.error.emit(f"Kaggle package not available: {exc}")
            return

        self._dest.mkdir(parents=True, exist_ok=True)
        self.status.emit("Downloading Kaggle dataset...")
        self.progress.emit(10)

        try:
            kaggle_api.dataset_download_files(
                self._dataset_ref,
                path=str(self._dest),
                unzip=True,
                quiet=True,
            )
        except Exception as exc:
            self.error.emit(f"Kaggle download failed: {exc}")
            return

        self.progress.emit(80)
        self.status.emit("Scanning downloaded files...")

        self.progress.emit(100)
        self.finished.emit(str(self._dest))


class OpenImagesDownloadWorker(QThread):
    """Download Open Images subset via FiftyOne and export to YOLO."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str, list, int)
    error = pyqtSignal(str)

    def __init__(
        self,
        classes: list[str],
        splits: list[str],
        max_samples: int,
        bbox_only: bool,
        dest: Path,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._classes = classes
        self._splits = splits
        self._max_samples = max_samples
        self._bbox_only = bbox_only
        self._dest = dest

    def run(self) -> None:
        try:
            import fiftyone as fo  # type: ignore
        except Exception:
            self.error.emit(
                "Open Images requires the optional 'fiftyone' package. "
                "Install with: pip install fiftyone"
            )
            return

        self._dest.mkdir(parents=True, exist_ok=True)
        self.status.emit("Downloading Open Images subset...")
        self.progress.emit(5)

        total_images = 0
        try:
            for idx, split in enumerate(self._splits):
                dataset = fo.zoo.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types=["detections"] if self._bbox_only else None,
                    classes=self._classes,
                    max_samples=self._max_samples,
                )
                export_dir = self._dest / split
                dataset.export(
                    export_dir=str(export_dir),
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field="ground_truth",
                )
                total_images += len(dataset)
                self.progress.emit(int(((idx + 1) / max(1, len(self._splits))) * 90))
        except Exception as exc:
            self.error.emit(f"Open Images download failed: {exc}")
            return

        self.progress.emit(100)
        self.status.emit("Open Images download complete.")
        self.finished.emit(str(self._dest), self._classes, total_images)


class ConvertFormatDialog(QDialog):
    """Prompt for dataset conversion options."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Convert Dataset")
        self.resize(420, 220)

        self._skip_checkbox = QCheckBox("Skip conversion (keep raw files)", self)
        self._csv_checkbox = QCheckBox("Convert CSV annotations to YOLO (if supported)", self)
        self._skip_checkbox.setChecked(True)
        self._csv_checkbox.setChecked(False)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self._skip_checkbox)
        layout.addWidget(self._csv_checkbox)
        layout.addStretch(1)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def values(self) -> dict[str, bool]:
        return {
            "skip": self._skip_checkbox.isChecked(),
            "convert_csv": self._csv_checkbox.isChecked(),
        }

class DiscoverTab(QWidget):
    """Tab for discovering remote datasets/models and importing them locally."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the discover tab.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._active_project_id: int | None = None
        self._roboflow_api_key = ""
        self._hf_token = ""
        self._project_root = PROJECT_ROOT
        self._download_root = ROBOFLOW_DOWNLOAD_ROOT
        self._hf_model_cache_root = HF_MODEL_CACHE_ROOT
        self._hf_dataset_cache_root = HF_DATASET_CACHE_ROOT
        self._config = _load_config(CONFIG_PATH)
        self._discover_page_size = int(self._config.get("discover_page_size") or 8)

        self._rf_current_page = 1
        self._rf_page_size = self._discover_page_size
        self._rf_has_next = False

        self._hf_current_page = 1
        self._hf_page_size = self._discover_page_size
        self._hf_has_next = False

        self._kaggle_current_page = 1
        self._kaggle_page_size = self._discover_page_size
        self._kaggle_has_next = False

        self._kaggle_username = ""
        self._kaggle_api_key = ""

        self._downloaded_hf_weights: dict[str, str] = {}

        self._rf_search_worker: RoboflowSearchWorker | None = None
        self._rf_download_worker: RoboflowDownloadWorker | None = None
        self._hf_search_worker: HuggingFaceSearchWorker | None = None
        self._hf_model_download_worker: HuggingFaceModelDownloadWorker | None = None
        self._hf_dataset_import_worker: HuggingFaceDatasetImportWorker | None = None
        self._kaggle_search_worker: KaggleSearchWorker | None = None
        self._kaggle_download_worker: KaggleDownloadWorker | None = None
        self._open_images_worker: OpenImagesDownloadWorker | None = None

        self._stacked_sections: QStackedWidget

        self._rf_credentials_label: QLabel
        self._rf_search_input: QLineEdit
        self._rf_search_button: QPushButton
        self._rf_status_label: QLabel
        self._rf_progress_bar: QProgressBar
        self._rf_results_layout: QVBoxLayout
        self._rf_prev_button: QPushButton
        self._rf_next_button: QPushButton
        self._rf_page_label: QLabel

        self._hf_credentials_label: QLabel
        self._hf_search_input: QLineEdit
        self._hf_status_label: QLabel
        self._hf_progress_bar: QProgressBar
        self._hf_results_layout: QVBoxLayout
        self._hf_prev_button: QPushButton
        self._hf_next_button: QPushButton
        self._hf_page_label: QLabel

        self._kaggle_search_input: QLineEdit
        self._kaggle_search_button: QPushButton
        self._kaggle_type_combo: QComboBox
        self._kaggle_sort_combo: QComboBox
        self._kaggle_tag_input: QLineEdit
        self._kaggle_status_label: QLabel
        self._kaggle_progress_bar: QProgressBar
        self._kaggle_results_layout: QVBoxLayout
        self._kaggle_prev_button: QPushButton
        self._kaggle_next_button: QPushButton
        self._kaggle_page_label: QLabel

        self._oi_class_input: QLineEdit
        self._oi_class_list: QListWidget
        self._oi_train_check: QCheckBox
        self._oi_val_check: QCheckBox
        self._oi_test_check: QCheckBox
        self._oi_max_spin: QSpinBox
        self._oi_bbox_only_check: QCheckBox
        self._oi_size_label: QLabel
        self._oi_download_button: QPushButton

        self._build_ui()
        self._restore_credentials()

    def set_project_context(self, project_id: int | None, project_root: str | None = None) -> None:
        self._active_project_id = project_id
        self._project_root = Path(project_root) if project_root else PROJECT_ROOT
        self._download_root = self._project_root / "datasets" / "roboflow_downloads"
        self._hf_model_cache_root = self._project_root / "models" / "huggingface"
        self._hf_dataset_cache_root = self._project_root / "datasets" / "huggingface"

    def refresh_credentials(self) -> None:
        self._restore_credentials()

    def _build_ui(self) -> None:
        """Compose segmented top bar and stacked discover sections."""

        root = QVBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        segment_bar = QWidget(self)
        segment_layout = QHBoxLayout()
        segment_layout.setContentsMargins(0, 0, 0, 0)
        segment_layout.setSpacing(6)

        rf_button = QPushButton("Roboflow Universe", segment_bar)
        hf_button = QPushButton("Hugging Face Hub", segment_bar)
        kaggle_button = QPushButton("Kaggle", segment_bar)
        oi_button = QPushButton("Open Images", segment_bar)

        for button in (rf_button, hf_button, kaggle_button, oi_button):
            button.setCheckable(True)
            button.setProperty("secondary", True)
            button.setMinimumHeight(34)

        rf_button.setChecked(True)

        group = QButtonGroup(segment_bar)
        group.setExclusive(True)
        group.addButton(rf_button, 0)
        group.addButton(hf_button, 1)
        group.addButton(kaggle_button, 2)
        group.addButton(oi_button, 3)
        group.idClicked.connect(self._switch_section)

        segment_layout.addWidget(rf_button)
        segment_layout.addWidget(hf_button)
        segment_layout.addWidget(kaggle_button)
        segment_layout.addWidget(oi_button)
        segment_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        segment_bar.setLayout(segment_layout)

        self._stacked_sections = QStackedWidget(self)
        self._stacked_sections.addWidget(self._build_roboflow_section())
        self._stacked_sections.addWidget(self._build_huggingface_section())
        self._stacked_sections.addWidget(self._build_kaggle_section())
        self._stacked_sections.addWidget(self._build_open_images_section())

        root.addWidget(segment_bar)
        root.addWidget(self._stacked_sections)
        self.setLayout(root)

    def _build_roboflow_section(self) -> QWidget:
        """Create Roboflow discover section."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._rf_credentials_label = QLabel("Roboflow API key not set. Use File > Settings.", panel)
        self._rf_credentials_label.setProperty("role", "warning")

        search_row = QWidget(panel)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(6)

        self._rf_search_input = QLineEdit(search_row)
        self._rf_search_input.setPlaceholderText("Search projects (or use workspace/project)")
        self._rf_search_input.returnPressed.connect(lambda: self._start_roboflow_search(reset_page=True))

        self._rf_search_button = QPushButton("Search", search_row)
        self._rf_search_button.clicked.connect(lambda: self._start_roboflow_search(reset_page=True))

        search_layout.addWidget(QLabel("Search", search_row))
        search_layout.addWidget(self._rf_search_input, stretch=1)
        search_layout.addWidget(self._rf_search_button)
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
        self._rf_results_layout.setSpacing(6)
        results_container.setLayout(self._rf_results_layout)
        scroll.setWidget(results_container)

        pagination_row = QWidget(panel)
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(6)

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

        layout.addWidget(self._rf_credentials_label)
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
        layout.setSpacing(6)

        self._hf_credentials_label = QLabel(
            "Hugging Face token not set. Public assets only. Use File > Settings to add one.",
            panel,
        )
        self._hf_credentials_label.setProperty("role", "subtle")

        search_row = QWidget(panel)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(6)

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
        self._hf_results_layout.setSpacing(6)
        results_container.setLayout(self._hf_results_layout)
        scroll.setWidget(results_container)

        pagination_row = QWidget(panel)
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(6)

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

        layout.addWidget(self._hf_credentials_label)
        layout.addWidget(search_row)
        layout.addWidget(self._hf_status_label)
        layout.addWidget(self._hf_progress_bar)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(pagination_row)
        panel.setLayout(layout)

        self._render_empty_state(self._hf_results_layout, "No Hugging Face results yet.")
        self._update_hf_pagination_controls()
        return panel

    def _build_kaggle_section(self) -> QWidget:
        """Create Kaggle discover section."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._kaggle_status_label = QLabel(
            "Kaggle credentials not set. Use File > Settings.",
            panel,
        )
        self._kaggle_status_label.setProperty("role", "warning")

        search_row = QWidget(panel)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(6)

        self._kaggle_search_input = QLineEdit(search_row)
        self._kaggle_search_input.setPlaceholderText("Search Kaggle datasets...")
        self._kaggle_search_input.returnPressed.connect(lambda: self._start_kaggle_search(reset_page=True))

        search_button = QPushButton("Search", search_row)
        search_button.clicked.connect(lambda: self._start_kaggle_search(reset_page=True))
        self._kaggle_search_button = search_button

        search_layout.addWidget(QLabel("Search", search_row))
        search_layout.addWidget(self._kaggle_search_input, stretch=1)
        search_layout.addWidget(search_button)
        search_row.setLayout(search_layout)

        filter_row = QWidget(panel)
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(6)

        self._kaggle_type_combo = QComboBox(filter_row)
        self._kaggle_type_combo.addItem("Datasets", "datasets")
        self._kaggle_type_combo.addItem("Competitions", "competitions")

        self._kaggle_sort_combo = QComboBox(filter_row)
        self._kaggle_sort_combo.addItem("Hotness", "hotness")
        self._kaggle_sort_combo.addItem("Votes", "votes")
        self._kaggle_sort_combo.addItem("Updated", "updated")
        self._kaggle_sort_combo.addItem("Active", "active")

        self._kaggle_tag_input = QLineEdit(filter_row)
        self._kaggle_tag_input.setPlaceholderText("Tag filter (optional)")

        filter_layout.addWidget(QLabel("Type", filter_row))
        filter_layout.addWidget(self._kaggle_type_combo)
        filter_layout.addWidget(QLabel("Sort", filter_row))
        filter_layout.addWidget(self._kaggle_sort_combo)
        filter_layout.addWidget(QLabel("Tag", filter_row))
        filter_layout.addWidget(self._kaggle_tag_input, stretch=1)
        filter_row.setLayout(filter_layout)

        self._kaggle_progress_bar = QProgressBar(panel)
        self._kaggle_progress_bar.setRange(0, 100)
        self._kaggle_progress_bar.setValue(0)

        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        results_container = QWidget(scroll)
        self._kaggle_results_layout = QVBoxLayout()
        self._kaggle_results_layout.setContentsMargins(0, 0, 0, 0)
        self._kaggle_results_layout.setSpacing(6)
        results_container.setLayout(self._kaggle_results_layout)
        scroll.setWidget(results_container)

        pagination_row = QWidget(panel)
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        pagination_layout.setSpacing(6)

        self._kaggle_prev_button = QPushButton("Previous", pagination_row)
        self._kaggle_prev_button.setProperty("secondary", True)
        self._kaggle_prev_button.clicked.connect(self._goto_prev_kaggle_page)

        self._kaggle_next_button = QPushButton("Next", pagination_row)
        self._kaggle_next_button.setProperty("secondary", True)
        self._kaggle_next_button.clicked.connect(self._goto_next_kaggle_page)

        self._kaggle_page_label = QLabel("Page 1", pagination_row)
        self._kaggle_page_label.setObjectName("metricValue")

        pagination_layout.addWidget(self._kaggle_prev_button)
        pagination_layout.addWidget(self._kaggle_next_button)
        pagination_layout.addWidget(self._kaggle_page_label)
        pagination_layout.addStretch(1)
        pagination_row.setLayout(pagination_layout)

        layout.addWidget(self._kaggle_status_label)
        layout.addWidget(search_row)
        layout.addWidget(filter_row)
        layout.addWidget(self._kaggle_progress_bar)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(pagination_row)
        panel.setLayout(layout)

        self._render_empty_state(self._kaggle_results_layout, "No Kaggle results yet.")
        self._update_kaggle_pagination_controls()
        return panel

    def _build_open_images_section(self) -> QWidget:
        """Create Open Images download section."""

        panel = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        class_row = QWidget(panel)
        class_layout = QHBoxLayout()
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(6)

        self._oi_class_input = QLineEdit(class_row)
        self._oi_class_input.setPlaceholderText("Search Open Images classes...")
        self._oi_class_input.textChanged.connect(self._update_open_images_suggestions)

        add_button = QPushButton("Add Class", class_row)
        add_button.clicked.connect(self._add_open_images_class)

        class_layout.addWidget(self._oi_class_input, stretch=1)
        class_layout.addWidget(add_button)
        class_row.setLayout(class_layout)

        self._oi_class_list = QListWidget(panel)
        self._oi_class_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        remove_button = QPushButton("Remove Selected", panel)
        remove_button.setProperty("secondary", True)
        remove_button.clicked.connect(self._remove_open_images_class)

        split_row = QWidget(panel)
        split_layout = QHBoxLayout()
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(6)

        self._oi_train_check = QCheckBox("Train", split_row)
        self._oi_train_check.setChecked(True)
        self._oi_val_check = QCheckBox("Validation", split_row)
        self._oi_val_check.setChecked(True)
        self._oi_test_check = QCheckBox("Test", split_row)

        split_layout.addWidget(self._oi_train_check)
        split_layout.addWidget(self._oi_val_check)
        split_layout.addWidget(self._oi_test_check)
        split_layout.addStretch(1)
        split_row.setLayout(split_layout)

        settings_row = QWidget(panel)
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        self._oi_max_spin = QSpinBox(settings_row)
        self._oi_max_spin.setRange(100, 5000)
        self._oi_max_spin.setValue(500)
        self._oi_max_spin.setSingleStep(100)
        self._oi_max_spin.valueChanged.connect(self._update_open_images_size_estimate)

        self._oi_bbox_only_check = QCheckBox("Bounding boxes only", settings_row)
        self._oi_bbox_only_check.setChecked(True)
        self._oi_bbox_only_check.toggled.connect(self._update_open_images_size_estimate)

        settings_layout.addWidget(QLabel("Max images per class", settings_row))
        settings_layout.addWidget(self._oi_max_spin)
        settings_layout.addWidget(self._oi_bbox_only_check)
        settings_layout.addStretch(1)
        settings_row.setLayout(settings_layout)

        self._oi_size_label = QLabel("Estimated download size: -", panel)
        self._oi_size_label.setProperty("role", "subtle")

        self._oi_download_button = QPushButton("Download Open Images", panel)
        self._oi_download_button.clicked.connect(self._start_open_images_download)

        layout.addWidget(class_row)
        layout.addWidget(self._oi_class_list, stretch=1)
        layout.addWidget(remove_button)
        layout.addWidget(split_row)
        layout.addWidget(settings_row)
        layout.addWidget(self._oi_size_label)
        layout.addWidget(self._oi_download_button)
        panel.setLayout(layout)

        self._load_open_images_classes()
        self._update_open_images_size_estimate()
        return panel

    def _switch_section(self, section_index: int) -> None:
        """Switch between Roboflow and Hugging Face sections.

        Args:
            section_index: Index into stacked sections.
        """

        self._stacked_sections.setCurrentIndex(max(0, min(section_index, self._stacked_sections.count() - 1)))

    def _restore_credentials(self) -> None:
        """Restore API credentials from local config file."""

        self._config = _load_config(CONFIG_PATH)
        self._roboflow_api_key = str(self._config.get("roboflow_api_key", "")).strip()
        self._hf_token = str(self._config.get("huggingface_token", "")).strip()
        self._kaggle_username = str(self._config.get("kaggle_username", "")).strip()
        self._kaggle_api_key = str(self._config.get("kaggle_api_key", "")).strip()

        if self._roboflow_api_key:
            self._rf_credentials_label.setText("Roboflow API key loaded.")
            self._rf_credentials_label.setProperty("role", "subtle")
        else:
            self._rf_credentials_label.setText("Roboflow API key not set. Use File > Settings.")
            self._rf_credentials_label.setProperty("role", "warning")
        self._rf_search_input.setEnabled(bool(self._roboflow_api_key))
        self._rf_search_button.setEnabled(bool(self._roboflow_api_key))

        if self._hf_token:
            self._hf_credentials_label.setText("Hugging Face token loaded.")
            self._hf_credentials_label.setProperty("role", "subtle")
        else:
            self._hf_credentials_label.setText(
                "Hugging Face token not set. Public assets only. Use File > Settings to add one."
            )
            self._hf_credentials_label.setProperty("role", "subtle")

        if self._kaggle_username and self._kaggle_api_key:
            self._kaggle_status_label.setText("Kaggle credentials loaded.")
            self._kaggle_status_label.setProperty("role", "subtle")
        else:
            self._kaggle_status_label.setText("Kaggle credentials not set. Use File > Settings.")
            self._kaggle_status_label.setProperty("role", "warning")
        self._kaggle_search_input.setEnabled(bool(self._kaggle_username and self._kaggle_api_key))
        self._kaggle_search_button.setEnabled(bool(self._kaggle_username and self._kaggle_api_key))

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

        api_key = self._roboflow_api_key
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
            card_layout.setContentsMargins(8, 8, 8, 8)
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
            actions_layout.setSpacing(6)

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

        api_key = self._roboflow_api_key
        version = _to_int(record.get("version"))

        self._rf_progress_bar.setValue(0)
        self._rf_status_label.setText("Starting Roboflow dataset download...")

        self._rf_download_worker = RoboflowDownloadWorker(
            api_key=api_key,
            workspace_slug=workspace,
            project_slug=project_slug,
            version=version,
            download_root=self._download_root,
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
            token=self._hf_token,
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

    def _start_kaggle_search(self, reset_page: bool) -> None:
        if self._kaggle_search_worker is not None and self._kaggle_search_worker.isRunning():
            QMessageBox.information(self, "Search Running", "Kaggle search is already running.")
            return

        if not (self._kaggle_username and self._kaggle_api_key):
            QMessageBox.warning(self, "Validation", "Set Kaggle credentials in File > Settings.")
            return

        query = self._kaggle_search_input.text().strip()
        if reset_page:
            self._kaggle_current_page = 1

        self._kaggle_progress_bar.setValue(0)
        self._kaggle_status_label.setText("Searching Kaggle...")

        self._kaggle_search_worker = KaggleSearchWorker(
            username=self._kaggle_username,
            api_key=self._kaggle_api_key,
            query=query,
            page=self._kaggle_current_page,
            page_size=self._kaggle_page_size,
            search_type=self._kaggle_type_combo.currentData(),
            sort=self._kaggle_sort_combo.currentData(),
            tag=self._kaggle_tag_input.text().strip(),
            parent=self,
        )
        self._kaggle_search_worker.progress.connect(self._kaggle_progress_bar.setValue)
        self._kaggle_search_worker.status.connect(self._kaggle_status_label.setText)
        self._kaggle_search_worker.finished.connect(self._on_kaggle_search_finished)
        self._kaggle_search_worker.error.connect(self._on_kaggle_search_error)
        self._kaggle_search_worker.start()

    def _on_kaggle_search_finished(self, records: list[dict[str, Any]]) -> None:
        self._kaggle_search_worker = None
        _clear_layout(self._kaggle_results_layout)

        if not records:
            self._render_empty_state(self._kaggle_results_layout, "No Kaggle results found.")
            self._kaggle_has_next = False
            self._update_kaggle_pagination_controls()
            self._kaggle_progress_bar.setValue(100)
            self._kaggle_status_label.setText(
                f"No Kaggle results on page {self._kaggle_current_page}."
            )
            return

        search_type = self._kaggle_type_combo.currentData()
        for record in records[: self._kaggle_page_size]:
            if search_type == "competitions":
                card = _build_kaggle_competition_card(record, self)
            else:
                card = _build_kaggle_dataset_card(record, self, self._start_kaggle_download)
            self._kaggle_results_layout.addWidget(card)

        self._kaggle_has_next = len(records) >= self._kaggle_page_size
        self._update_kaggle_pagination_controls()
        self._kaggle_status_label.setText(
            f"Kaggle results: page {self._kaggle_current_page}, showing {len(records)}"
        )
        self._kaggle_progress_bar.setValue(100)

    def _on_kaggle_search_error(self, message: str) -> None:
        self._kaggle_search_worker = None
        self._kaggle_status_label.setText(f"Kaggle search failed: {message}")
        self._kaggle_progress_bar.setValue(0)
        QMessageBox.critical(self, "Kaggle Search Error", message)

    def _start_kaggle_download(self, dataset_ref: str) -> None:
        if self._kaggle_download_worker is not None and self._kaggle_download_worker.isRunning():
            QMessageBox.information(self, "Download Running", "A Kaggle download is already running.")
            return

        dataset_ref = str(dataset_ref or "").strip()
        if not dataset_ref:
            QMessageBox.warning(self, "Missing Metadata", "Selected dataset does not include a Kaggle reference.")
            return

        dest_root = self._project_root / "datasets" / "kaggle"
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / dataset_ref.replace("/", "__")

        self._kaggle_download_worker = KaggleDownloadWorker(
            username=self._kaggle_username,
            api_key=self._kaggle_api_key,
            dataset_ref=dataset_ref,
            dest=dest,
            parent=self,
        )
        self._kaggle_download_worker.progress.connect(self._kaggle_progress_bar.setValue)
        self._kaggle_download_worker.status.connect(self._kaggle_status_label.setText)
        self._kaggle_download_worker.finished.connect(self._on_kaggle_download_finished)
        self._kaggle_download_worker.error.connect(self._on_kaggle_download_error)
        self._kaggle_download_worker.start()

    def _on_kaggle_download_finished(self, dest_path: str) -> None:
        self._kaggle_download_worker = None
        dest = Path(dest_path)
        self._kaggle_status_label.setText("Kaggle download complete.")

        if _has_yolo_structure(dest):
            _register_dataset_from_path(
                dest,
                source=DatasetSource.KAGGLE,
                project_id=self._active_project_id,
                tags=["kaggle"],
            )
            QMessageBox.information(self, "Kaggle", f"Dataset registered from:\n{dest}")
            return

        dialog = ConvertFormatDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        if values.get("convert_csv"):
            converted = _convert_csv_to_yolo(dest)
            if converted:
                _register_dataset_from_path(
                    converted,
                    source=DatasetSource.KAGGLE,
                    project_id=self._active_project_id,
                    tags=["kaggle", "converted"],
                )
                QMessageBox.information(self, "Kaggle", f"Converted dataset registered:\n{converted}")
            else:
                QMessageBox.warning(self, "Kaggle", "CSV conversion not supported for this dataset.")

    def _on_kaggle_download_error(self, message: str) -> None:
        self._kaggle_download_worker = None
        self._kaggle_status_label.setText(f"Kaggle download failed: {message}")
        QMessageBox.critical(self, "Kaggle Download Error", message)

    def _load_open_images_classes(self) -> None:
        self._oi_classes: list[str] = []
        if OPEN_IMAGES_CLASSES_PATH.exists():
            try:
                payload = json.loads(OPEN_IMAGES_CLASSES_PATH.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    self._oi_classes = [str(item) for item in payload if str(item).strip()]
            except Exception:
                LOGGER.exception("Failed to load Open Images classes from %s", OPEN_IMAGES_CLASSES_PATH)

        if not self._oi_classes:
            self._oi_classes = ["Person", "Car", "Dog", "Cat"]

        completer = QCompleter(sorted(self._oi_classes), self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._oi_class_input.setCompleter(completer)

    def _update_open_images_suggestions(self) -> None:
        # Completer handles suggestions; keep for future hooks.
        pass

    def _add_open_images_class(self) -> None:
        raw = self._oi_class_input.text().strip()
        if not raw:
            return

        match = None
        for candidate in self._oi_classes:
            if candidate.lower() == raw.lower():
                match = candidate
                break
        if match is None:
            QMessageBox.warning(self, "Open Images", "Select a class from the Open Images list.")
            return

        existing = {self._oi_class_list.item(idx).text() for idx in range(self._oi_class_list.count())}
        if match in existing:
            self._oi_class_input.clear()
            return

        self._oi_class_list.addItem(QListWidgetItem(match))
        self._oi_class_input.clear()
        self._update_open_images_size_estimate()

    def _remove_open_images_class(self) -> None:
        row = self._oi_class_list.currentRow()
        if row >= 0:
            self._oi_class_list.takeItem(row)
            self._update_open_images_size_estimate()

    def _update_open_images_size_estimate(self) -> None:
        class_count = self._oi_class_list.count()
        max_images = self._oi_max_spin.value()
        if class_count == 0:
            self._oi_size_label.setText("Estimated download size: -")
            return

        estimated_images = class_count * max_images
        estimated_mb = estimated_images * 0.25
        if estimated_mb > 1024:
            estimate_text = f"~{estimated_mb / 1024:.1f} GB"
        else:
            estimate_text = f"~{estimated_mb:.0f} MB"
        self._oi_size_label.setText(
            f"Estimated download size: {estimate_text} for ~{estimated_images} images"
        )

    def _start_open_images_download(self) -> None:
        if self._open_images_worker is not None and self._open_images_worker.isRunning():
            QMessageBox.information(self, "Download Running", "Open Images download is already in progress.")
            return

        classes = [self._oi_class_list.item(idx).text() for idx in range(self._oi_class_list.count())]
        if not classes:
            QMessageBox.warning(self, "Validation", "Select at least one Open Images class.")
            return

        splits: list[str] = []
        if self._oi_train_check.isChecked():
            splits.append("train")
        if self._oi_val_check.isChecked():
            splits.append("validation")
        if self._oi_test_check.isChecked():
            splits.append("test")

        if not splits:
            QMessageBox.warning(self, "Validation", "Select at least one split.")
            return

        dest_root = self._project_root / "datasets" / "open_images"
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / _slugify("_".join(classes))

        self._oi_download_button.setEnabled(False)
        self._oi_size_label.setText("Starting Open Images download...")

        self._open_images_worker = OpenImagesDownloadWorker(
            classes=classes,
            splits=splits,
            max_samples=self._oi_max_spin.value(),
            bbox_only=self._oi_bbox_only_check.isChecked(),
            dest=dest,
            parent=self,
        )
        self._open_images_worker.progress.connect(lambda value: self._oi_size_label.setText(f"Progress: {value}%"))
        self._open_images_worker.status.connect(self._oi_size_label.setText)
        self._open_images_worker.finished.connect(self._on_open_images_finished)
        self._open_images_worker.error.connect(self._on_open_images_error)
        self._open_images_worker.start()

    def _on_open_images_finished(self, dest_path: str, classes: list[str], total_images: int) -> None:
        self._open_images_worker = None
        self._oi_download_button.setEnabled(True)

        dest = Path(dest_path)
        yaml_path = dest / "data.yaml"
        yaml_payload = {
            "path": str(dest.resolve()),
            "train": "train/images" if (dest / "train" / "images").exists() else None,
            "val": "validation/images" if (dest / "validation" / "images").exists() else None,
            "test": "test/images" if (dest / "test" / "images").exists() else None,
            "names": classes,
            "nc": len(classes),
        }
        yaml_payload = {k: v for k, v in yaml_payload.items() if v is not None}
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yaml_payload, handle, sort_keys=False)

        self._upsert_dataset_record(
            name=f"Open Images ({', '.join(classes)})",
            source=DatasetSource.OPEN_IMAGES,
            local_path=str(dest.resolve()),
            class_names=classes,
            num_images=total_images or _count_images(dest),
            tags=["open_images", *classes],
            description="Imported from Open Images V7 via FiftyOne.",
            roboflow_project_id=None,
            hf_repo_id=None,
        )

        QMessageBox.information(
            self,
            "Open Images Downloaded",
            f"Dataset saved to: {dest}\nClasses: {len(classes)}\nImages: {total_images}\nYAML: {yaml_path}",
        )

    def _on_open_images_error(self, message: str) -> None:
        self._open_images_worker = None
        self._oi_download_button.setEnabled(True)
        if "fiftyone" in message.lower():
            QMessageBox.warning(
                self,
                "Open Images Dependency",
                "Open Images download requires the optional 'fiftyone' package.\n"
                "Install with: pip install fiftyone\n"
                "Note: this is a large dependency and may take a while to install.",
            )
            return
        QMessageBox.critical(self, "Open Images Error", message)

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
            card_layout.setContentsMargins(8, 8, 8, 8)
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
            action_layout.setSpacing(6)

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
            token=self._hf_token,
            repo_id=repo_id,
            cache_root=self._hf_model_cache_root,
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
                str(self._hf_model_cache_root),
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
            query = session.query(BaseWeight).filter(BaseWeight.repo_id == repo_id)
            if self._active_project_id is not None:
                query = query.filter(BaseWeight.project_id == self._active_project_id)
            existing = query.order_by(BaseWeight.updated_at.desc()).first()

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
                    project_id=self._active_project_id,
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
                if self._active_project_id is not None:
                    existing.project_id = self._active_project_id

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
            token=self._hf_token,
            repo_id=repo_id,
            cache_root=self._hf_dataset_cache_root,
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

    def _update_kaggle_pagination_controls(self) -> None:
        self._kaggle_page_label.setText(f"Page {self._kaggle_current_page}")
        self._kaggle_prev_button.setEnabled(self._kaggle_current_page > 1)
        self._kaggle_next_button.setEnabled(self._kaggle_has_next)

    def _goto_prev_kaggle_page(self) -> None:
        if self._kaggle_current_page <= 1:
            return
        self._kaggle_current_page -= 1
        self._start_kaggle_search(reset_page=False)

    def _goto_next_kaggle_page(self) -> None:
        if not self._kaggle_has_next:
            return
        self._kaggle_current_page += 1
        self._start_kaggle_search(reset_page=False)

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
        card_layout.setContentsMargins(8, 8, 8, 8)

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
                    project_id=self._active_project_id,
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
                if self._active_project_id is not None:
                    existing.project_id = self._active_project_id

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


def _ensure_kaggle_credentials(username: str, api_key: str) -> None:
    """Ensure kaggle.json credentials exist without overwriting."""

    if not username or not api_key:
        raise RuntimeError("Kaggle credentials are required.")

    home = Path(os.path.expanduser("~"))
    kaggle_dir = home / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"

    if kaggle_file.exists():
        return

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    payload = {"username": username, "key": api_key}
    kaggle_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        kaggle_file.chmod(0o600)
    except Exception:
        LOGGER.debug("Unable to chmod kaggle.json", exc_info=True)


def _normalize_kaggle_dataset(raw: Any) -> dict[str, Any]:
    """Normalize Kaggle dataset object or dict."""

    if raw is None:
        return {}

    data: Mapping[str, Any]
    if isinstance(raw, dict):
        data = raw
    else:
        data = {
            "title": getattr(raw, "title", None),
            "ref": getattr(raw, "ref", None),
            "ownerName": getattr(raw, "ownerName", None),
            "totalBytes": getattr(raw, "totalBytes", None),
            "size": getattr(raw, "size", None),
            "downloadCount": getattr(raw, "downloadCount", None),
            "voteCount": getattr(raw, "voteCount", None),
            "lastUpdated": getattr(raw, "lastUpdated", None),
            "usabilityRating": getattr(raw, "usabilityRating", None),
            "subtitle": getattr(raw, "subtitle", None),
        }

    size_value = data.get("totalBytes") or data.get("size")
    size_text = _format_bytes(size_value)

    return {
        "title": str(data.get("title") or data.get("ref") or "Untitled"),
        "ref": str(data.get("ref") or ""),
        "author": str(data.get("ownerName") or data.get("author") or "-"),
        "size": size_text,
        "votes": _to_int(data.get("voteCount")) or 0,
        "downloads": _to_int(data.get("downloadCount")) or 0,
        "last_updated": _normalize_datetime_text(data.get("lastUpdated")),
        "usability": float(data.get("usabilityRating") or 0.0),
        "subtitle": str(data.get("subtitle") or ""),
    }


def _normalize_kaggle_competition(raw: Any) -> dict[str, Any]:
    """Normalize Kaggle competition object or dict."""

    if raw is None:
        return {}

    data: Mapping[str, Any]
    if isinstance(raw, dict):
        data = raw
    else:
        data = {
            "title": getattr(raw, "title", None),
            "ref": getattr(raw, "ref", None),
            "id": getattr(raw, "id", None),
            "reward": getattr(raw, "reward", None),
            "deadline": getattr(raw, "deadline", None),
            "teamCount": getattr(raw, "teamCount", None),
            "evaluationMetric": getattr(raw, "evaluationMetric", None),
            "url": getattr(raw, "url", None),
        }

    url = data.get("url")
    if not url and data.get("ref"):
        url = f"https://www.kaggle.com/competitions/{data.get('ref')}"

    return {
        "title": str(data.get("title") or data.get("ref") or "Competition"),
        "ref": str(data.get("ref") or ""),
        "prize": str(data.get("reward") or data.get("prize") or "-"),
        "deadline": _normalize_datetime_text(data.get("deadline")),
        "teams": _to_int(data.get("teamCount")) or 0,
        "metric": str(data.get("evaluationMetric") or data.get("metric") or "-"),
        "url": str(url or ""),
    }


def _build_kaggle_dataset_card(
    record: Mapping[str, Any],
    parent: QWidget,
    download_handler: Any,
) -> QFrame:
    card = QFrame(parent)
    card.setProperty("card", True)
    card_layout = QVBoxLayout()
    card_layout.setContentsMargins(8, 8, 8, 8)
    card_layout.setSpacing(6)

    title = QLabel(str(record.get("title") or "Untitled"), card)
    title.setStyleSheet("font-size: 14px; font-weight: 600;")

    meta_line = QLabel(
        (
            f"Author: {record.get('author', '-')}"
            f" | Size: {record.get('size', '-')}"
            f" | Votes: {record.get('votes', '-')}"
            f" | Downloads: {record.get('downloads', '-')}"
        ),
        card,
    )
    meta_line.setProperty("role", "subtle")

    updated_line = QLabel(f"Last Updated: {record.get('last_updated', '-')}", card)
    updated_line.setProperty("role", "subtle")

    badge_row = QWidget(card)
    badge_layout = QHBoxLayout()
    badge_layout.setContentsMargins(0, 0, 0, 0)
    badge_layout.setSpacing(6)

    usability = float(record.get("usability") or 0.0)
    badge = QLabel(f"Usability: {usability:.2f}", badge_row)
    badge.setObjectName("metricBadge")
    if usability >= 0.7:
        badge.setProperty("role", "success")
    elif usability >= 0.4:
        badge.setProperty("role", "warning")
    else:
        badge.setProperty("role", "error")

    badge_layout.addWidget(badge)
    badge_layout.addStretch(1)
    badge_row.setLayout(badge_layout)

    actions = QWidget(card)
    actions_layout = QHBoxLayout()
    actions_layout.setContentsMargins(0, 0, 0, 0)
    actions_layout.setSpacing(6)

    dataset_ref = str(record.get("ref") or "")
    download_button = QPushButton("Download Dataset", actions)
    if dataset_ref:
        download_button.clicked.connect(lambda _checked=False, ref=dataset_ref: download_handler(ref))
    else:
        download_button.setEnabled(False)

    actions_layout.addWidget(download_button)
    actions_layout.addStretch(1)
    actions.setLayout(actions_layout)

    card_layout.addWidget(title)
    if record.get("subtitle"):
        subtitle = QLabel(str(record.get("subtitle")), card)
        subtitle.setProperty("role", "subtle")
        card_layout.addWidget(subtitle)
    card_layout.addWidget(meta_line)
    card_layout.addWidget(updated_line)
    card_layout.addWidget(badge_row)
    card_layout.addWidget(actions)
    card.setLayout(card_layout)
    return card


def _build_kaggle_competition_card(record: Mapping[str, Any], parent: QWidget) -> QFrame:
    card = QFrame(parent)
    card.setProperty("card", True)
    card_layout = QVBoxLayout()
    card_layout.setContentsMargins(8, 8, 8, 8)
    card_layout.setSpacing(6)

    title = QLabel(str(record.get("title") or "Competition"), card)
    title.setStyleSheet("font-size: 14px; font-weight: 600;")

    meta_line = QLabel(
        (
            f"Prize: {record.get('prize', '-')}"
            f" | Deadline: {record.get('deadline', '-')}"
            f" | Teams: {record.get('teams', '-')}"
        ),
        card,
    )
    meta_line.setProperty("role", "subtle")

    metric_line = QLabel(f"Metric: {record.get('metric', '-')}", card)
    metric_line.setProperty("role", "subtle")

    actions = QWidget(card)
    actions_layout = QHBoxLayout()
    actions_layout.setContentsMargins(0, 0, 0, 0)
    actions_layout.setSpacing(6)

    view_button = QPushButton("View on Kaggle", actions)
    url = str(record.get("url") or "")
    view_button.clicked.connect(lambda _checked=False, link=url: _open_url(link))

    actions_layout.addWidget(view_button)
    actions_layout.addStretch(1)
    actions.setLayout(actions_layout)

    card_layout.addWidget(title)
    card_layout.addWidget(meta_line)
    card_layout.addWidget(metric_line)
    card_layout.addWidget(actions)
    card.setLayout(card_layout)
    return card


def _open_url(url: str) -> None:
    if not url:
        return
    QDesktopServices.openUrl(QUrl(url))


def _format_bytes(value: Any) -> str:
    if value is None:
        return "-"
    try:
        size = float(value)
    except Exception:
        return str(value)
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _has_yolo_structure(root: Path) -> bool:
    if not root.exists():
        return False
    if list(root.rglob("data.yaml")):
        return True
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.exists() and labels_dir.exists():
        return True
    return False


def _register_dataset_from_path(
    dataset_root: Path,
    source: DatasetSource,
    project_id: int | None,
    tags: list[str],
) -> None:
    dataset_root = dataset_root.resolve()
    yaml_paths = list(dataset_root.rglob("data.yaml"))
    if yaml_paths:
        dataset_root = yaml_paths[0].parent.resolve()

    class_names = _load_class_names_from_yaml(dataset_root / "data.yaml")
    if not class_names:
        class_names = _infer_class_names_from_labels(dataset_root)

    num_images = _count_images(dataset_root)
    dataset_path = str(dataset_root)
    name = dataset_root.name

    session = get_session()
    try:
        existing = session.query(Dataset).filter(Dataset.local_path == dataset_path).first()
        if existing is None:
            existing = Dataset(
                name=name,
                description=f"Imported dataset from {source.value}.",
                source=source,
                local_path=dataset_path,
                class_names=class_names,
                num_images=num_images,
                num_classes=len(class_names),
                tags=_clean_tags(tags),
                project_id=project_id,
            )
            session.add(existing)
        else:
            existing.name = name
            existing.source = source
            existing.class_names = class_names
            existing.num_images = num_images
            existing.num_classes = len(class_names)
            existing.tags = _clean_tags(tags)
            if project_id is not None:
                existing.project_id = project_id
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _convert_csv_to_yolo(root: Path) -> Path | None:
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        return None

    images_index = _index_images(root)
    if not images_index:
        return None

    column_sets = {
        "image": {"image", "image_path", "filepath", "file", "filename", "img", "img_path"},
        "label": {"class", "label", "category", "name"},
        "xmin": {"xmin", "x_min", "left", "x1"},
        "ymin": {"ymin", "y_min", "top", "y1"},
        "xmax": {"xmax", "x_max", "right", "x2"},
        "ymax": {"ymax", "y_max", "bottom", "y2"},
    }

    from PIL import Image

    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue

                header = {name.strip(): name for name in reader.fieldnames}

                def find_key(candidates: set[str]) -> str | None:
                    for key in header:
                        if key.lower() in candidates:
                            return header[key]
                    return None

                image_key = find_key(column_sets["image"])
                label_key = find_key(column_sets["label"])
                xmin_key = find_key(column_sets["xmin"])
                ymin_key = find_key(column_sets["ymin"])
                xmax_key = find_key(column_sets["xmax"])
                ymax_key = find_key(column_sets["ymax"])

                if not all([image_key, label_key, xmin_key, ymin_key, xmax_key, ymax_key]):
                    continue

                yolo_root = root / "yolo_converted"
                images_dir = yolo_root / "images" / "train"
                labels_dir = yolo_root / "labels" / "train"
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)

                class_map: dict[str, int] = {}
                labels_by_image: dict[str, list[str]] = {}

                for row in reader:
                    image_name = str(row.get(image_key) or "").strip()
                    if not image_name or image_name.startswith("._"):
                        continue

                    image_path = images_index.get(image_name.lower())
                    if image_path is None:
                        continue

                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                    except Exception:
                        continue

                    label_name = str(row.get(label_key) or "object").strip() or "object"
                    if label_name not in class_map:
                        class_map[label_name] = len(class_map)

                    try:
                        xmin = float(row.get(xmin_key) or 0)
                        ymin = float(row.get(ymin_key) or 0)
                        xmax = float(row.get(xmax_key) or 0)
                        ymax = float(row.get(ymax_key) or 0)
                    except Exception:
                        continue

                    box_w = max(0.0, xmax - xmin)
                    box_h = max(0.0, ymax - ymin)
                    if box_w <= 0 or box_h <= 0:
                        continue

                    x_center = xmin + box_w / 2.0
                    y_center = ymin + box_h / 2.0

                    x_center /= width
                    y_center /= height
                    box_w /= width
                    box_h /= height

                    line = f"{class_map[label_name]} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
                    labels_by_image.setdefault(image_path.name, []).append(line)

                if not labels_by_image:
                    continue

                for image_name, lines in labels_by_image.items():
                    image_path = images_index.get(image_name.lower())
                    if image_path is None:
                        continue
                    shutil.copy2(image_path, images_dir / image_path.name)
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    label_path.write_text("\n".join(lines), encoding="utf-8")

                class_names = [name for name, _idx in sorted(class_map.items(), key=lambda kv: kv[1])]
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

                return yolo_root.resolve()
        except Exception:
            continue

    return None


def _index_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name.lower()] = path
    return images

__all__ = [
    "DiscoverTab",
    "RoboflowSearchWorker",
    "RoboflowDownloadWorker",
    "HuggingFaceSearchWorker",
    "HuggingFaceModelDownloadWorker",
    "HuggingFaceDatasetImportWorker",
    "KaggleSearchWorker",
    "KaggleDownloadWorker",
    "OpenImagesDownloadWorker",
]

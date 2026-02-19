"""Dataset CRUD and import/export utilities for YOLO Studio.

This module centralizes dataset-oriented operations that are used by the UI
tabs and background workers. It provides a small service class with
production-friendly helpers for:
- creating/updating/deleting dataset records,
- registering existing YOLO-format dataset folders,
- copying datasets into a managed local library, and
- exporting registered datasets as zip archives.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from core.models.database import Dataset, DatasetSource, get_session


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetManagerError(RuntimeError):
    """Raised when dataset manager operations fail."""


class DatasetManager:
    """Service object for dataset persistence and filesystem operations."""

    def __init__(self, dataset_root: str | Path | None = None) -> None:
        """Initialize dataset manager configuration.

        Args:
            dataset_root: Optional managed dataset root directory.
        """

        self._dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT).resolve()
        self._dataset_root.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_root(self) -> Path:
        """Return the managed dataset root directory.

        Returns:
            Path: Root path used for managed dataset copies.
        """

        return self._dataset_root

    def list_datasets(self) -> list[Dataset]:
        """Return all dataset records ordered by newest first.

        Returns:
            list[Dataset]: Dataset rows from the database.
        """

        with get_session() as session:
            return session.query(Dataset).order_by(Dataset.created_at.desc()).all()

    def get_dataset(self, dataset_id: int) -> Dataset | None:
        """Fetch a dataset by identifier.

        Args:
            dataset_id: Dataset primary key.

        Returns:
            Dataset | None: Matching dataset or None.
        """

        with get_session() as session:
            return session.get(Dataset, int(dataset_id))

    def create_dataset(
        self,
        *,
        name: str,
        local_path: str | Path,
        description: str | None = None,
        source: DatasetSource | str = DatasetSource.MANUAL,
        roboflow_project_id: str | None = None,
        hf_repo_id: str | None = None,
        class_names: list[str] | None = None,
        num_images: int | None = None,
        num_classes: int | None = None,
        tags: list[str] | None = None,
    ) -> Dataset:
        """Create and persist a dataset row.

        Args:
            name: Human-readable dataset name.
            local_path: Filesystem path to dataset folder.
            description: Optional description text.
            source: Dataset source enum/value.
            roboflow_project_id: Optional Roboflow project ID.
            hf_repo_id: Optional Hugging Face repo ID.
            class_names: Optional list of class names.
            num_images: Optional image count override.
            num_classes: Optional class count override.
            tags: Optional tags.

        Returns:
            Dataset: Persisted dataset row.

        Raises:
            DatasetManagerError: If required values are invalid.
        """

        resolved_name = str(name).strip()
        if not resolved_name:
            raise DatasetManagerError("Dataset name is required.")

        path = Path(local_path).expanduser().resolve()
        if not path.exists():
            raise DatasetManagerError(f"Dataset path does not exist: {path}")

        normalized_classes = [item.strip() for item in (class_names or []) if item and item.strip()]
        image_count = int(num_images) if num_images is not None else _count_images(path)
        class_count = int(num_classes) if num_classes is not None else len(normalized_classes)
        if class_count == 0 and normalized_classes:
            class_count = len(normalized_classes)

        source_enum = _normalize_source(source)

        row = Dataset(
            name=resolved_name,
            description=(description or "").strip() or None,
            source=source_enum,
            roboflow_project_id=(roboflow_project_id or "").strip() or None,
            hf_repo_id=(hf_repo_id or "").strip() or None,
            local_path=str(path),
            class_names=normalized_classes,
            num_images=max(0, image_count),
            num_classes=max(0, class_count),
            tags=[item.strip() for item in (tags or []) if item and item.strip()],
        )

        with get_session() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def update_dataset(self, dataset_id: int, **fields: Any) -> Dataset:
        """Update mutable dataset fields and persist changes.

        Args:
            dataset_id: Dataset primary key.
            **fields: Mutable field updates.

        Returns:
            Dataset: Updated dataset row.

        Raises:
            DatasetManagerError: If dataset is missing or updates are invalid.
        """

        with get_session() as session:
            row = session.get(Dataset, int(dataset_id))
            if row is None:
                raise DatasetManagerError(f"Dataset not found: {dataset_id}")

            if "name" in fields:
                name = str(fields["name"] or "").strip()
                if not name:
                    raise DatasetManagerError("Dataset name cannot be empty.")
                row.name = name

            if "description" in fields:
                row.description = str(fields["description"] or "").strip() or None

            if "source" in fields:
                row.source = _normalize_source(fields["source"])

            if "roboflow_project_id" in fields:
                row.roboflow_project_id = str(fields["roboflow_project_id"] or "").strip() or None

            if "hf_repo_id" in fields:
                row.hf_repo_id = str(fields["hf_repo_id"] or "").strip() or None

            if "local_path" in fields:
                path = Path(fields["local_path"]).expanduser().resolve()
                if not path.exists():
                    raise DatasetManagerError(f"Dataset path does not exist: {path}")
                row.local_path = str(path)

            if "class_names" in fields:
                row.class_names = [
                    item.strip()
                    for item in (fields.get("class_names") or [])
                    if isinstance(item, str) and item.strip()
                ]
                if "num_classes" not in fields:
                    row.num_classes = len(row.class_names)

            if "num_images" in fields:
                row.num_images = max(0, int(fields["num_images"]))

            if "num_classes" in fields:
                row.num_classes = max(0, int(fields["num_classes"]))

            if "tags" in fields:
                row.tags = [
                    item.strip()
                    for item in (fields.get("tags") or [])
                    if isinstance(item, str) and item.strip()
                ]

            session.commit()
            session.refresh(row)
            return row

    def delete_dataset(self, dataset_id: int, remove_files: bool = False) -> bool:
        """Delete a dataset record with optional managed-files removal.

        Args:
            dataset_id: Dataset primary key.
            remove_files: Whether to delete dataset folder from disk.

        Returns:
            bool: True when a row was deleted, else False.
        """

        with get_session() as session:
            row = session.get(Dataset, int(dataset_id))
            if row is None:
                return False

            local_path = Path(row.local_path).expanduser().resolve()
            session.delete(row)
            session.commit()

        if remove_files and local_path.exists():
            # File deletion is restricted to dataset assets only.
            if _is_path_under(local_path, self._dataset_root):
                shutil.rmtree(local_path, ignore_errors=True)

        return True

    def register_dataset_folder(
        self,
        folder_path: str | Path,
        *,
        name: str | None = None,
        description: str | None = None,
        source: DatasetSource | str = DatasetSource.MANUAL,
        tags: list[str] | None = None,
        copy_into_library: bool = False,
    ) -> Dataset:
        """Register an existing YOLO-format folder as a dataset record.

        Args:
            folder_path: Existing dataset folder path.
            name: Optional record name override.
            description: Optional description text.
            source: Dataset source enum/value.
            tags: Optional tags.
            copy_into_library: When True, copy into managed dataset root first.

        Returns:
            Dataset: Persisted dataset row.

        Raises:
            DatasetManagerError: If folder path is invalid.
        """

        source_path = Path(folder_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_dir():
            raise DatasetManagerError(f"Dataset folder does not exist: {source_path}")

        target_path = source_path
        if copy_into_library:
            target_path = self._copy_to_library(source_path, preferred_name=name)

        data_yaml_path = _find_data_yaml(target_path)
        classes = _extract_class_names(target_path, data_yaml_path)
        num_images = _count_images(target_path)

        return self.create_dataset(
            name=(name or target_path.name).strip(),
            description=description,
            source=source,
            local_path=target_path,
            class_names=classes,
            num_images=num_images,
            num_classes=len(classes),
            tags=tags,
        )

    def export_dataset_zip(self, dataset_id: int, output_zip_path: str | Path | None = None) -> Path:
        """Create a zip archive from a registered dataset folder.

        Args:
            dataset_id: Dataset primary key.
            output_zip_path: Optional destination zip file path.

        Returns:
            Path: Created archive path.

        Raises:
            DatasetManagerError: If dataset or source folder is invalid.
        """

        row = self.get_dataset(dataset_id)
        if row is None:
            raise DatasetManagerError(f"Dataset not found: {dataset_id}")

        source_path = Path(row.local_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_dir():
            raise DatasetManagerError(f"Dataset folder does not exist: {source_path}")

        if output_zip_path is None:
            output_base = (self._dataset_root / _slugify(row.name)).resolve()
            output_base.parent.mkdir(parents=True, exist_ok=True)
            archive_path = Path(shutil.make_archive(str(output_base), "zip", root_dir=source_path))
        else:
            zip_path = Path(output_zip_path).expanduser().resolve()
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            root_without_suffix = zip_path.with_suffix("")
            archive_path = Path(shutil.make_archive(str(root_without_suffix), "zip", root_dir=source_path))
            if archive_path != zip_path:
                if zip_path.exists():
                    zip_path.unlink()
                archive_path.replace(zip_path)
                archive_path = zip_path

        return archive_path.resolve()

    def _copy_to_library(self, source_path: Path, preferred_name: str | None = None) -> Path:
        """Copy source dataset folder into managed dataset root.

        Args:
            source_path: Existing source dataset folder.
            preferred_name: Optional target folder name preference.

        Returns:
            Path: Copied dataset directory.

        Raises:
            DatasetManagerError: If copy operation fails.
        """

        base_name = _slugify(preferred_name or source_path.name or "dataset")
        candidate = self._dataset_root / base_name
        if candidate.exists():
            suffix = 2
            while (self._dataset_root / f"{base_name}_{suffix}").exists():
                suffix += 1
            candidate = self._dataset_root / f"{base_name}_{suffix}"

        try:
            shutil.copytree(source_path, candidate)
        except Exception as exc:
            raise DatasetManagerError(f"Failed copying dataset folder: {exc}") from exc

        return candidate.resolve()


def _normalize_source(value: DatasetSource | str) -> DatasetSource:
    """Normalize enum/string source value into `DatasetSource`.

    Args:
        value: Candidate source value.

    Returns:
        DatasetSource: Normalized source.

    Raises:
        DatasetManagerError: If value is not recognized.
    """

    if isinstance(value, DatasetSource):
        return value

    raw = str(value or "").strip().lower()
    if not raw:
        return DatasetSource.MANUAL

    for candidate in DatasetSource:
        if raw == candidate.value:
            return candidate

    raise DatasetManagerError(f"Unsupported dataset source: {value}")


def _extract_class_names(dataset_dir: Path, data_yaml_path: Path | None) -> list[str]:
    """Extract class names from `data.yaml` or `classes.txt`.

    Args:
        dataset_dir: Dataset folder path.
        data_yaml_path: Optional data.yaml path.

    Returns:
        list[str]: Parsed class names.
    """

    if data_yaml_path is not None:
        names = _read_names_from_data_yaml(data_yaml_path)
        if names:
            return names

    classes_txt = dataset_dir / "classes.txt"
    if classes_txt.exists():
        names = [
            line.strip()
            for line in classes_txt.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if names:
            return names

    return []


def _find_data_yaml(dataset_dir: Path) -> Path | None:
    """Locate YOLO data YAML file in the dataset directory.

    Args:
        dataset_dir: Dataset folder path.

    Returns:
        Path | None: Located data YAML path, if present.
    """

    direct_candidates = (
        dataset_dir / "data.yaml",
        dataset_dir / "dataset.yaml",
    )
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    for candidate in dataset_dir.glob("*.yaml"):
        if candidate.is_file():
            return candidate

    for candidate in dataset_dir.glob("*.yml"):
        if candidate.is_file():
            return candidate

    return None


def _read_names_from_data_yaml(path: Path) -> list[str]:
    """Read class names from YOLO dataset YAML payload.

    Args:
        path: Data YAML path.

    Returns:
        list[str]: Parsed class-name list.
    """

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    names = payload.get("names")
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()]

    if isinstance(names, dict):
        ordered = sorted(
            ((int(key), str(value).strip()) for key, value in names.items() if str(value).strip()),
            key=lambda item: item[0],
        )
        return [value for _, value in ordered]

    return []


def _count_images(path: Path) -> int:
    """Count image files under a dataset path.

    Args:
        path: Dataset folder or single file path.

    Returns:
        int: Number of detected image files.
    """

    if not path.exists():
        return 0

    if path.is_file():
        return 1 if path.suffix.lower() in IMAGE_EXTENSIONS else 0

    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def _slugify(value: str) -> str:
    """Create filesystem-safe slug text.

    Args:
        value: Input text.

    Returns:
        str: Safe slug string.
    """

    raw = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw).strip("_")
    return safe or "dataset"


def _is_path_under(path: Path, parent: Path) -> bool:
    """Return whether path is nested under parent path.

    Args:
        path: Candidate child path.
        parent: Candidate parent path.

    Returns:
        bool: True if `path` is under `parent`.
    """

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


__all__ = ["DatasetManager", "DatasetManagerError"]

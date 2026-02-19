"""Project management dialogs for YOLO Studio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import yaml

from core.models.database import BaseWeight, BaseWeightSource, Dataset, DatasetSource, Project, TrainingRun, get_session
from core.services.project_service import (
    ProjectConfig,
    now_iso,
    scan_for_datasets,
    scan_for_weights,
    write_project_yaml,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProjectInput:
    name: str
    description: str
    root_dir: str
    import_assets: bool


class ProjectWizardDialog(QDialog):
    """Wizard dialog for creating a new project."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.resize(480, 360)

        self._name_input = QLineEdit(self)
        self._description_input = QTextEdit(self)
        self._description_input.setMinimumHeight(80)
        self._root_input = QLineEdit(self)
        self._root_input.setPlaceholderText("Select project root directory")
        self._import_checkbox = QCheckBox("Import existing assets by scanning folder", self)

        browse_button = QPushButton("Browse", self)
        browse_button.clicked.connect(self._browse_root)

        root_row = QWidget(self)
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._root_input, stretch=1)
        root_layout.addWidget(browse_button)
        root_row.setLayout(root_layout)

        form = QFormLayout()
        form.addRow("Name", self._name_input)
        form.addRow("Description", self._description_input)
        form.addRow("Root Directory", root_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._import_checkbox)
        layout.addStretch(1)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _browse_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if selected:
            self._root_input.setText(str(Path(selected).resolve()))

    def values(self) -> ProjectInput:
        return ProjectInput(
            name=self._name_input.text().strip(),
            description=self._description_input.toPlainText().strip(),
            root_dir=self._root_input.text().strip(),
            import_assets=self._import_checkbox.isChecked(),
        )


class ProjectManagerDialog(QDialog):
    """Dialog for managing projects."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Projects")
        self.resize(640, 420)

        self._project_list = QListWidget(self)
        self._project_list.currentItemChanged.connect(self._update_stats)

        self._stats_label = QLabel("Select a project", self)
        self._stats_label.setProperty("role", "subtle")
        self._stats_label.setWordWrap(True)

        self._create_button = QPushButton("Create", self)
        self._rename_button = QPushButton("Rename", self)
        self._delete_button = QPushButton("Delete", self)
        self._duplicate_button = QPushButton("Duplicate", self)
        self._set_root_button = QPushButton("Set Root", self)

        self._create_button.clicked.connect(self._create_project)
        self._rename_button.clicked.connect(self._rename_project)
        self._delete_button.clicked.connect(self._delete_project)
        self._duplicate_button.clicked.connect(self._duplicate_project)
        self._set_root_button.clicked.connect(self._set_root)

        button_row = QWidget(self)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        for button in (
            self._create_button,
            self._rename_button,
            self._duplicate_button,
            self._set_root_button,
            self._delete_button,
        ):
            button_layout.addWidget(button)
        button_row.setLayout(button_layout)

        layout = QVBoxLayout()
        layout.addWidget(self._project_list, stretch=1)
        layout.addWidget(self._stats_label)
        layout.addWidget(button_row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_box.rejected.connect(self.reject)
        close_box.accepted.connect(self.accept)
        layout.addWidget(close_box)
        self.setLayout(layout)

        self.refresh()

    def refresh(self) -> None:
        self._project_list.clear()
        session = get_session()
        try:
            projects = session.query(Project).order_by(Project.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load projects.")
            projects = []
        finally:
            session.close()

        for project in projects:
            item = QListWidgetItem(project.name)
            item.setData(Qt.ItemDataRole.UserRole, project.id)
            self._project_list.addItem(item)

    def selected_project_id(self) -> int | None:
        item = self._project_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _update_stats(self) -> None:
        project_id = self.selected_project_id()
        if project_id is None:
            self._stats_label.setText("Select a project")
            return

        session = get_session()
        try:
            dataset_count = session.query(Dataset).filter(Dataset.project_id == project_id).count()
            run_count = session.query(TrainingRun).filter(TrainingRun.project_id == project_id).count()
            model_count = (
                session.query(TrainingRun)
                .filter(TrainingRun.project_id == project_id)
                .filter(TrainingRun.is_saved.is_(True))
                .count()
            )
        except Exception:
            LOGGER.exception("Failed to load project stats.")
            dataset_count = run_count = model_count = 0
        finally:
            session.close()

        self._stats_label.setText(
            f"Datasets: {dataset_count} | Runs: {run_count} | Models: {model_count}"
        )

    def _create_project(self) -> None:
        dialog = ProjectWizardDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        if not values.name or not values.root_dir:
            QMessageBox.warning(self, "Validation", "Project name and root directory are required.")
            return

        root = Path(values.root_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        session = get_session()
        try:
            project = Project(
                name=values.name,
                description=values.description or None,
                root_dir=str(root),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(project)
            session.commit()
            session.refresh(project)
            _write_project_yaml(project)
            if values.import_assets:
                _import_project_assets(project.id, root)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Create Failed", f"Could not create project: {exc}")
            return
        finally:
            session.close()

        self.refresh()

    def _rename_project(self) -> None:
        project_id = self.selected_project_id()
        if project_id is None:
            return

        item = self._project_list.currentItem()
        if item is None:
            return

        new_name, ok = QInputDialog.getText(self, "Rename Project", "New name:", text=item.text())
        if not ok or not new_name.strip():
            return

        session = get_session()
        try:
            project = session.get(Project, project_id)
            if project is None:
                return
            project.name = new_name.strip()
            project.updated_at = datetime.now(timezone.utc)
            session.commit()
            _write_project_yaml(project)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Rename Failed", f"Could not rename project: {exc}")
        finally:
            session.close()

        self.refresh()

    def _set_root(self) -> None:
        project_id = self.selected_project_id()
        if project_id is None:
            return

        selected = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if not selected:
            return

        root = str(Path(selected).resolve())
        session = get_session()
        try:
            project = session.get(Project, project_id)
            if project is None:
                return
            project.root_dir = root
            project.updated_at = datetime.now(timezone.utc)
            session.commit()
            _write_project_yaml(project)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Update Failed", f"Could not update root dir: {exc}")
        finally:
            session.close()

        self.refresh()

    def _duplicate_project(self) -> None:
        project_id = self.selected_project_id()
        if project_id is None:
            return

        session = get_session()
        try:
            project = session.get(Project, project_id)
            if project is None:
                return
            dup = Project(
                name=f"{project.name} Copy",
                description=project.description,
                root_dir=project.root_dir,
                git_remote=project.git_remote,
                git_branch=project.git_branch,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(dup)
            session.commit()
            session.refresh(dup)
            _write_project_yaml(dup)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Duplicate Failed", f"Could not duplicate project: {exc}")
        finally:
            session.close()

        self.refresh()

    def _delete_project(self) -> None:
        project_id = self.selected_project_id()
        if project_id is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete Project",
            "Delete selected project? This will not delete files on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        session = get_session()
        try:
            project = session.get(Project, project_id)
            if project is None:
                return
            session.delete(project)
            session.commit()
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Delete Failed", f"Could not delete project: {exc}")
        finally:
            session.close()

        self.refresh()


def _write_project_yaml(project: Project) -> None:
    root = Path(project.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    yaml_path = root / "project.yaml"
    payload = ProjectConfig(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at.isoformat() if project.created_at else now_iso(),
        updated_at=project.updated_at.isoformat() if project.updated_at else now_iso(),
        root_dir=str(root),
        git_remote=project.git_remote,
        git_branch=project.git_branch,
    )
    try:
        write_project_yaml(yaml_path, payload)
    except Exception:
        LOGGER.exception("Failed to write project.yaml")


def _import_project_assets(project_id: int, root: Path) -> None:
    dataset_paths = scan_for_datasets(root)
    weight_paths = scan_for_weights(root)

    session = get_session()
    try:
        for yaml_path in dataset_paths:
            dataset_root = yaml_path.parent
            existing = (
                session.query(Dataset)
                .filter(Dataset.local_path == str(dataset_root.resolve()))
                .first()
            )
            if existing is not None:
                continue

            class_names = _read_class_names(yaml_path)
            num_images = _count_images(dataset_root)
            record = Dataset(
                name=dataset_root.name,
                description=f"Imported from {dataset_root}",
                source=DatasetSource.MANUAL,
                local_path=str(dataset_root.resolve()),
                class_names=class_names,
                num_images=num_images,
                num_classes=len(class_names),
                tags=["imported"],
                project_id=project_id,
            )
            session.add(record)

        for weight_path in weight_paths:
            existing_weight = (
                session.query(BaseWeight)
                .filter(BaseWeight.local_path == str(weight_path.resolve()))
                .first()
            )
            if existing_weight is not None:
                continue

            record = BaseWeight(
                name=weight_path.stem,
                source=BaseWeightSource.LOCAL,
                local_path=str(weight_path.resolve()),
                project_id=project_id,
            )
            session.add(record)

        session.commit()
    except Exception:
        session.rollback()
        LOGGER.exception("Failed importing project assets.")
    finally:
        session.close()


def _read_class_names(yaml_path: Path) -> list[str]:
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    names = payload.get("names")
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()]
    if isinstance(names, dict):
        ordered = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
        return [str(item).strip() for item in ordered if str(item).strip()]
    return []


def _count_images(dataset_root: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    count = 0
    for ext in extensions:
        count += sum(1 for _ in dataset_root.rglob(f"*{ext}"))
    return count


__all__ = ["ProjectWizardDialog", "ProjectManagerDialog", "ProjectInput"]

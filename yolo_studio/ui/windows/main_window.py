"""Main application window for YOLO Studio."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import yaml

from core.models.database import BaseWeight, BaseWeightSource, Dataset, DatasetSource, Project, get_session
from core.services.project_service import ProjectConfig, load_project_yaml, now_iso, scan_for_datasets, scan_for_weights, write_project_yaml
from ui.dialogs.project_dialogs import ProjectManagerDialog, ProjectWizardDialog
from ui.dialogs.settings_dialog import SettingsDialog
from ui.styles.theme import apply_theme

from ..widgets.log_panel import LogPanel, QtLogHandler

try:
    from ..tabs.train_tab import TrainTab
except Exception:
    TrainTab = None  # type: ignore[assignment]

try:
    from ..tabs.dataset_tab import DatasetTab
except Exception:
    DatasetTab = None  # type: ignore[assignment]

try:
    from ..tabs.discover_tab import DiscoverTab
except Exception:
    DiscoverTab = None  # type: ignore[assignment]

try:
    from ..tabs.remote_tab import RemoteTab
except Exception:
    RemoteTab = None  # type: ignore[assignment]

try:
    from ..tabs.evaluate_tab import EvaluateTab
except Exception:
    EvaluateTab = None  # type: ignore[assignment]

try:
    from ..tabs.analytics_tab import AnalyticsTab
except Exception:
    AnalyticsTab = None  # type: ignore[assignment]

try:
    from ..tabs.camera_tab import CameraTab
except Exception:
    CameraTab = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


class PlaceholderTab(QWidget):
    """Fallback tab shown when a target tab implementation is unavailable."""

    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        """Initialize a placeholder tab.

        Args:
            title: Human-readable title for this placeholder.
            message: Description shown to the user.
            parent: Optional parent widget.
        """

        super().__init__(parent)

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")

        body = QLabel(message)
        body.setWordWrap(True)
        body.setProperty("role", "subtle")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addStretch(1)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Top-level window hosting tabs, status indicators, and log output."""

    project_changed = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window and UI components.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self.setWindowTitle("YOLO Studio")
        self._apply_initial_geometry()

        self._log_handler: QtLogHandler | None = None
        self._log_dock: QDockWidget | None = None
        self._log_panel: LogPanel | None = None
        self._status_summary_label: QLabel | None = None
        self._status_hint_label: QLabel | None = None
        self._tab_widget: QTabWidget | None = None
        self._project_toolbar: QToolBar | None = None
        self._project_stack: QStackedWidget | None = None
        self._welcome_widget: QWidget | None = None
        self._recent_list: QListWidget | None = None
        self._active_project_id: int | None = None
        self._project_scope: str = "none"
        self._projects: dict[int, Project] = {}

        self._setup_ui()
        self._setup_logging_bridge()

        LOGGER.info("Main window initialized.")
        self.show_status_message("Ready")

    def _setup_ui(self) -> None:
        """Create and compose core UI sections."""

        self._setup_tabs()
        self._setup_status_bar()
        self._setup_log_dock()
        self._setup_menu()

    def _apply_initial_geometry(self) -> None:
        """Size the window relative to the active screen, with a sane minimum."""

        screen = self.screen()
        if screen is None:
            self.resize(1600, 960)
            self.setMinimumSize(960, 640)
            return

        available = screen.availableGeometry()
        target_width = max(960, int(available.width() * 0.9))
        target_height = max(640, int(available.height() * 0.9))
        target_width = min(target_width, available.width())
        target_height = min(target_height, available.height())
        self.resize(target_width, target_height)
        self.setMinimumSize(960, 640)

    def _setup_tabs(self) -> None:
        """Initialize the central tab widget and default tabs."""

        central = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self._tab_widget = QTabWidget(central)
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setMovable(False)
        self._tab_widget.setTabsClosable(False)

        self._tab_widget.addTab(
            self._build_tab(
                TrainTab,
                "Training",
                "Training tab is not implemented yet. Step 5 will provide full training controls.",
            ),
            "Training",
        )
        self._tab_widget.addTab(
            self._build_tab(
                DatasetTab,
                "Datasets",
                "Dataset management tab is not implemented yet. Step 7 will add dataset workflows.",
            ),
            "Datasets",
        )
        self._tab_widget.addTab(
            self._build_tab(
                DiscoverTab,
                "Discover",
                "Discover tab is not implemented yet. Step 8 will add Roboflow and Hugging Face integration.",
            ),
            "Discover",
        )
        self._tab_widget.addTab(
            self._build_tab(
                RemoteTab,
                "Remote Devices",
                "Remote tab is not implemented yet. Step 9 will add device management and testing.",
            ),
            "Remote Devices",
        )
        self._tab_widget.addTab(
            self._build_tab(
                EvaluateTab,
                "Evaluate",
                "Evaluate tab is not implemented yet.",
            ),
            "Evaluate",
        )
        self._tab_widget.addTab(
            self._build_tab(
                AnalyticsTab,
                "Analytics",
                "Analytics tab is not implemented yet.",
            ),
            "Analytics",
        )
        self._tab_widget.addTab(
            self._build_tab(
                CameraTab,
                "Camera",
                "Camera tab is not implemented yet.",
            ),
            "Camera",
        )

        self._project_stack = QStackedWidget(central)
        self._welcome_widget = self._build_welcome_widget()
        self._project_stack.addWidget(self._welcome_widget)
        self._project_stack.addWidget(self._tab_widget)

        layout.addWidget(self._project_stack)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self._set_project_scope("none")

    def _build_tab(
        self,
        tab_cls: type[QWidget] | None,
        title: str,
        fallback_message: str,
    ) -> QWidget:
        """Instantiate a tab widget or fallback placeholder.

        Args:
            tab_cls: Tab widget class if available.
            title: Title used for placeholder heading.
            fallback_message: Message shown when fallback is needed.

        Returns:
            QWidget: Instantiated tab widget.
        """

        if tab_cls is None:
            LOGGER.warning("%s tab import unavailable; using placeholder.", title)
            return _wrap_scrollable(PlaceholderTab(title, fallback_message, self))

        try:
            tab = tab_cls(self)
            if hasattr(tab, "set_project_context"):
                tab.set_project_context(self._active_project_id, self._active_project_root())
                self.project_changed.connect(tab.set_project_context)
            return _wrap_scrollable(tab)
        except Exception:
            LOGGER.exception("Failed to initialize %s tab; using placeholder.", title)
            return _wrap_scrollable(PlaceholderTab(title, fallback_message, self))

    def _setup_status_bar(self) -> None:
        """Initialize status bar with persistent summary labels."""

        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(True)

        self._status_summary_label = QLabel("System: initializing", self)
        self._status_summary_label.setProperty("role", "subtle")

        self._status_hint_label = QLabel("Logs: docked at bottom", self)
        self._status_hint_label.setProperty("role", "subtle")

        status_bar.addPermanentWidget(self._status_summary_label)
        status_bar.addPermanentWidget(self._status_hint_label)

    def _setup_log_dock(self) -> None:
        """Create the dockable log output panel."""

        self._log_panel = LogPanel(self)

        self._log_dock = QDockWidget("Logs", self)
        self._log_dock.setObjectName("logDock")
        self._log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._log_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._log_dock.setWidget(self._log_panel)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

    def _setup_menu(self) -> None:
        """Create minimal menu actions for window controls."""

        file_menu = self.menuBar().addMenu("File")
        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self._launch_new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self._open_project_dialog)
        file_menu.addAction(open_project_action)

        manage_projects_action = QAction("Projects", self)
        manage_projects_action.triggered.connect(self._open_project_manager)
        file_menu.addAction(manage_projects_action)

        all_projects_action = QAction("All Projects", self)
        all_projects_action.triggered.connect(lambda: self._set_project_scope("all"))
        file_menu.addAction(all_projects_action)

        close_project_action = QAction("Close Project", self)
        close_project_action.triggered.connect(lambda: self._set_project_scope("none"))
        file_menu.addAction(close_project_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        view_menu = self.menuBar().addMenu("View")

        if self._log_dock is not None:
            toggle_logs_action = QAction("Toggle Logs", self)
            toggle_logs_action.setCheckable(True)
            toggle_logs_action.setChecked(True)
            toggle_logs_action.toggled.connect(self._log_dock.setVisible)
            self._log_dock.visibilityChanged.connect(toggle_logs_action.setChecked)
            view_menu.addAction(toggle_logs_action)

    def _reload_projects(self) -> None:
        current_scope = self._project_scope
        current_project = self._active_project_id

        self._projects.clear()
        session = get_session()
        try:
            projects = session.query(Project).order_by(Project.created_at.desc()).all()
        except Exception:
            LOGGER.exception("Failed to load projects.")
            projects = []
        finally:
            session.close()

        for project in projects:
            self._projects[int(project.id)] = project

        if current_scope == "project" and current_project is not None:
            if current_project not in self._projects:
                self._set_project_scope("none")

    def _set_project_scope(self, scope: str, project_id: int | None = None) -> None:
        self._project_scope = scope
        self._active_project_id = project_id if scope == "project" else None

        if self._project_stack is not None:
            self._project_stack.setCurrentIndex(1 if scope in {"project", "all"} else 0)

        if scope in {"project", "all"}:
            self.project_changed.emit(self._active_project_id, self._active_project_root())
            self._update_recent_projects()

    def _active_project_root(self) -> str | None:
        if self._active_project_id is None:
            return None
        project = self._projects.get(self._active_project_id)
        if project is None:
            return None
        return project.root_dir

    def _build_welcome_widget(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        heading = QLabel("Welcome to YOLO Studio", widget)
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")

        body = QLabel("Select a project to start, or create a new one.", widget)
        body.setWordWrap(True)
        body.setProperty("role", "subtle")

        self._recent_list = QListWidget(widget)
        self._recent_list.itemClicked.connect(self._open_recent_project)

        new_button = QPushButton("New Project", widget)
        new_button.clicked.connect(self._launch_new_project)

        open_button = QPushButton("Open Project", widget)
        open_button.setProperty("secondary", True)
        open_button.clicked.connect(self._open_project_dialog)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(QLabel("Recent Projects", widget))
        layout.addWidget(self._recent_list, stretch=1)
        layout.addWidget(new_button)
        layout.addWidget(open_button)
        widget.setLayout(layout)

        self._load_recent_projects()
        return widget

    def _launch_new_project(self) -> None:
        dialog = ProjectWizardDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        values = dialog.values()
        if not values.name or not values.root_dir:
            QMessageBox.warning(self, "Project", "Name and root directory are required.")
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
                self._import_project_assets(project.id, root)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Project", f"Could not create project: {exc}")
            return
        finally:
            session.close()

        self._reload_projects()
        self._select_project(project.id)

    def _open_project_manager(self) -> None:
        dialog = ProjectManagerDialog(self)
        dialog.exec()
        self._reload_projects()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, dialog.theme_selection())

        if self._tab_widget is None:
            return

        for index in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(index)
            wrapped = widget.widget() if isinstance(widget, QScrollArea) else widget
            if hasattr(wrapped, "refresh_credentials"):
                wrapped.refresh_credentials()

    def _select_project(self, project_id: int) -> None:
        self._set_project_scope("project", project_id)

    def _open_project_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.home()),
            "Project Config (project.yaml)",
        )
        if not selected:
            return
        self._open_project_from_path(Path(selected))

    def _open_project_from_path(self, path: Path) -> None:
        if path.is_dir():
            path = path / "project.yaml"
        if not path.exists():
            QMessageBox.warning(self, "Project", "project.yaml not found.")
            return

        config = load_project_yaml(path)
        if config is None or not config.name or not config.root_dir:
            QMessageBox.warning(self, "Project", "Invalid project.yaml.")
            return

        session = get_session()
        try:
            project = (
                session.query(Project)
                .filter(Project.root_dir == str(Path(config.root_dir).resolve()))
                .first()
            )
            if project is None:
                project = Project(
                    name=config.name,
                    description=config.description,
                    root_dir=str(Path(config.root_dir).resolve()),
                    git_remote=config.git_remote,
                    git_branch=config.git_branch,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(project)
                session.commit()
                session.refresh(project)
                _write_project_yaml(project)
        except Exception as exc:
            session.rollback()
            QMessageBox.critical(self, "Project", f"Could not open project: {exc}")
            return
        finally:
            session.close()

        self._reload_projects()
        self._select_project(project.id)

    def _import_project_assets(self, project_id: int, root: Path) -> None:
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

    def _load_recent_projects(self) -> None:
        if self._recent_list is None:
            return
        settings = QSettings()
        paths = settings.value("recent_projects", []) or []
        self._recent_list.clear()
        for path in paths:
            item = QListWidgetItem(str(path))
            self._recent_list.addItem(item)

    def _update_recent_projects(self) -> None:
        if self._active_project_id is None:
            return
        project = self._projects.get(self._active_project_id)
        if project is None:
            return
        settings = QSettings()
        paths = settings.value("recent_projects", []) or []
        root = project.root_dir
        paths = [p for p in paths if p != root]
        paths.insert(0, root)
        settings.setValue("recent_projects", paths[:10])
        self._load_recent_projects()

    def _open_recent_project(self, item: QListWidgetItem) -> None:
        path = Path(item.text())
        self._open_project_from_path(path)


    def _setup_logging_bridge(self) -> None:
        """Attach a UI log handler so Python logs appear in the dock."""

        if self._log_panel is None:
            return

        root_logger = logging.getLogger()
        self._log_handler = self._log_panel.attach_logger(root_logger, level=logging.INFO)
        self._log_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def show_status_message(self, message: str, timeout_ms: int = 5000) -> None:
        """Display a temporary status bar message.

        Args:
            message: Status text to show.
            timeout_ms: Duration to show the message in milliseconds.
        """

        self.statusBar().showMessage(message, timeout_ms)

        if self._status_summary_label is not None:
            self._status_summary_label.setText(f"System: {message}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cleanup handlers when the window closes.

        Args:
            event: Qt close event.
        """

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

        super().closeEvent(event)


def _write_project_yaml(project: Project) -> None:
    payload = ProjectConfig(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at.isoformat() if project.created_at else now_iso(),
        updated_at=project.updated_at.isoformat() if project.updated_at else now_iso(),
        root_dir=project.root_dir,
        git_remote=project.git_remote,
        git_branch=project.git_branch,
    )
    write_project_yaml(Path(project.root_dir) / "project.yaml", payload)


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


def _wrap_scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll

__all__ = ["MainWindow"]

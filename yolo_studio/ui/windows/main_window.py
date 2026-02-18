"""Main application window for YOLO Studio."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QDockWidget, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

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

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window and UI components.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self.setWindowTitle("YOLO Studio")
        self.resize(1600, 960)

        self._log_handler: QtLogHandler | None = None
        self._log_dock: QDockWidget | None = None
        self._log_panel: LogPanel | None = None
        self._status_summary_label: QLabel | None = None
        self._status_hint_label: QLabel | None = None
        self._tab_widget: QTabWidget | None = None

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

        layout.addWidget(self._tab_widget)
        central.setLayout(layout)
        self.setCentralWidget(central)

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
            return PlaceholderTab(title, fallback_message, self)

        try:
            return tab_cls(self)
        except Exception:
            LOGGER.exception("Failed to initialize %s tab; using placeholder.", title)
            return PlaceholderTab(title, fallback_message, self)

    def _setup_status_bar(self) -> None:
        """Initialize status bar with persistent summary labels."""

        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)

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

        view_menu = self.menuBar().addMenu("View")

        if self._log_dock is not None:
            toggle_logs_action = QAction("Toggle Logs", self)
            toggle_logs_action.setCheckable(True)
            toggle_logs_action.setChecked(True)
            toggle_logs_action.toggled.connect(self._log_dock.setVisible)
            self._log_dock.visibilityChanged.connect(toggle_logs_action.setChecked)
            view_menu.addAction(toggle_logs_action)

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
            self._log_handler = None

        super().closeEvent(event)


__all__ = ["MainWindow"]

"""Drag-and-drop file/folder input widget for dataset path selection."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


class FileDropZone(QFrame):
    """Interactive drop zone that accepts directory paths for dataset splits."""

    path_selected = pyqtSignal(str)

    def __init__(self, title: str, placeholder: str, parent: QWidget | None = None) -> None:
        """Initialize the drop zone.

        Args:
            title: Section title for this zone.
            placeholder: Hint text displayed before selection.
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._path: str | None = None
        self._placeholder = placeholder

        self.setObjectName("panel")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._title_label = QLabel(title, self)
        self._title_label.setStyleSheet("font-weight: 600;")

        self._path_label = QLabel(self._placeholder, self)
        self._path_label.setWordWrap(True)
        self._path_label.setProperty("role", "subtle")

        self._browse_button = QPushButton("Browse Folder", self)
        self._browse_button.clicked.connect(self._select_directory)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._title_label)
        layout.addWidget(self._path_label)
        layout.addWidget(self._browse_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events that contain local directory URLs.

        Args:
            event: Qt drag-enter event.
        """

        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Resolve and apply the first dropped local directory path.

        Args:
            event: Qt drop event.
        """

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue

            local_path = Path(url.toLocalFile())
            if local_path.is_dir():
                self.set_path(str(local_path.resolve()))
                event.acceptProposedAction()
                return

        event.ignore()

    def _select_directory(self) -> None:
        """Open a folder picker dialog and set the selected path."""

        selected = QFileDialog.getExistingDirectory(self, "Select Folder")
        if selected:
            self.set_path(selected)

    def set_path(self, path: str) -> None:
        """Set the selected folder path and emit a change signal.

        Args:
            path: Directory path to assign.
        """

        resolved = str(Path(path).resolve())
        self._path = resolved

        display_name = Path(resolved).name or resolved
        self._path_label.setText(f"{display_name}\n{resolved}")
        self._path_label.setProperty("role", "")
        self.path_selected.emit(resolved)

    def clear(self) -> None:
        """Clear the selected path and reset to placeholder style."""

        self._path = None
        self._path_label.setText(self._placeholder)
        self._path_label.setProperty("role", "subtle")

    def path(self) -> str | None:
        """Return the currently selected directory path.

        Returns:
            str | None: Selected path if available.
        """

        return self._path


__all__ = ["FileDropZone"]

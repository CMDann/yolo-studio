"""Logging widgets for rendering application logs inside the YOLO Studio UI."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class _LogEmitter(QObject):
    """Qt signal emitter used to bridge logging records onto the UI thread."""

    message_emitted = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to a Qt signal."""

    def __init__(self, emitter: _LogEmitter) -> None:
        """Initialize the handler.

        Args:
            emitter: Qt signal emitter that will dispatch log text.
        """

        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        """Format and emit a log record to the UI signal.

        Args:
            record: Log record emitted by the Python logging subsystem.
        """

        try:
            message = self.format(record)
            self._emitter.message_emitted.emit(message)
        except Exception:
            self.handleError(record)


class LogPanel(QWidget):
    """Scrollable panel for displaying real-time application logs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the log panel UI.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._emitter = _LogEmitter()
        self._emitter.message_emitted.connect(self.append_message)

        self._log_view = QTextEdit(self)
        self._log_view.setObjectName("logPanel")
        self._log_view.setReadOnly(True)
        self._log_view.setPlaceholderText("Runtime logs will appear here...")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._log_view)
        self.setLayout(layout)

    def append_message(self, message: str) -> None:
        """Append a log line to the panel.

        Args:
            message: Formatted log message to append.
        """

        self._log_view.append(message)

    def clear(self) -> None:
        """Clear all log output from the panel."""

        self._log_view.clear()

    def attach_logger(self, logger: logging.Logger, level: int = logging.INFO) -> QtLogHandler:
        """Attach a Qt-backed log handler to a logger.

        Args:
            logger: Logger instance that should emit into this panel.
            level: Logging level threshold for the attached handler.

        Returns:
            QtLogHandler: The installed handler for later customization/removal.
        """

        handler = QtLogHandler(self._emitter)
        handler.setLevel(level)
        logger.addHandler(handler)
        return handler


__all__ = ["LogPanel", "QtLogHandler"]

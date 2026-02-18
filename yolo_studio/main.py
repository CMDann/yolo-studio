"""Application entry point for YOLO Studio."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from core.models.database import init_db
from ui.windows.main_window import MainWindow
from ui.styles.theme import apply_theme


def configure_logging() -> None:
    """Configure application-wide logging handlers and formatters."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Launch the YOLO Studio desktop application.

    Returns:
        int: Process exit code.
    """

    configure_logging()
    logger = logging.getLogger(__name__)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Studio")
    app.setOrganizationName("YOLO Studio")

    apply_theme(app)

    try:
        init_db()
    except Exception as exc:
        logger.exception("Failed to initialize database schema.")
        QMessageBox.critical(None, "Database Error", f"Could not initialize database: {exc}")
        return 1

    window = MainWindow()
    window.show()

    logger.info("YOLO Studio started successfully.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

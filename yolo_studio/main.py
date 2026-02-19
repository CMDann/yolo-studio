"""Application entry point for YOLO Studio."""

from __future__ import annotations

import ctypes.util
import logging
import os
import socket
import stat
import sys

from core.models.database import init_db


def configure_qt_platform() -> None:
    """Set a Linux-friendly Qt platform fallback before importing Qt widgets.

    This helps avoid startup failures on Ubuntu systems where the default xcb
    plugin is unavailable but wayland/minimal backends are present.
    """

    if not sys.platform.startswith("linux"):
        return

    if os.getenv("YOLO_STUDIO_DISABLE_QT_PLATFORM_AUTO", "").strip().lower() in {"1", "true", "yes"}:
        return

    if os.getenv("QT_QPA_PLATFORM"):
        return

    session_type = os.getenv("XDG_SESSION_TYPE", "").strip().lower()
    wayland_display = os.getenv("WAYLAND_DISPLAY", "").strip()
    xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR", "").strip()
    display = os.getenv("DISPLAY", "").strip()

    def can_access_socket(path: str) -> bool:
        if not path:
            return False
        try:
            st = os.stat(path)
        except OSError:
            return False
        return stat.S_ISSOCK(st.st_mode) and os.access(path, os.R_OK | os.W_OK)

    def can_connect_wayland(path: str) -> bool:
        if not can_access_socket(path):
            return False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.1)
            sock.connect(path)
            return True
        except OSError:
            return False
        finally:
            sock.close()

    wayland_socket = os.path.join(xdg_runtime_dir, wayland_display) if wayland_display and xdg_runtime_dir else ""
    has_wayland = bool(wayland_display) or session_type == "wayland"
    can_use_wayland = has_wayland and can_connect_wayland(wayland_socket)

    has_x11 = bool(display)
    xauth = os.getenv("XAUTHORITY", "").strip()
    has_xcb_cursor = bool(ctypes.util.find_library("xcb-cursor"))
    can_use_x11 = has_x11 and (not xauth or os.path.exists(xauth)) and has_xcb_cursor

    if can_use_wayland and can_use_x11:
        os.environ["QT_QPA_PLATFORM"] = "wayland;xcb;minimal"
        return

    if can_use_wayland:
        os.environ["QT_QPA_PLATFORM"] = "wayland;minimal"
        return

    if can_use_x11:
        os.environ["QT_QPA_PLATFORM"] = "xcb;minimal"
        return

    # Headless Linux fallback for non-interactive environments.
    os.environ["QT_QPA_PLATFORM"] = "minimal"


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
    configure_qt_platform()
    logger = logging.getLogger(__name__)
    logger.info("QT_QPA_PLATFORM=%s", os.getenv("QT_QPA_PLATFORM", "<default>"))

    from PyQt6.QtWidgets import QApplication, QMessageBox

    from ui.styles.theme import apply_theme, load_theme_preference
    from ui.windows.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Studio")
    app.setOrganizationName("YOLO Studio")

    try:
        init_db()
    except Exception as exc:
        logger.exception("Failed to initialize database schema.")
        QMessageBox.critical(None, "Database Error", f"Could not initialize database: {exc}")
        return 1

    theme_name = load_theme_preference()
    apply_theme(app, theme_name)

    window = MainWindow()
    window.show()

    logger.info("YOLO Studio started successfully.")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

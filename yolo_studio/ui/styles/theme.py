"""Application theme utilities for YOLO Studio.

This module centralizes the visual design tokens and Qt style rules used by the
application. It provides a full dark theme stylesheet and an apply function
that configures palette, fonts, and style hints for a consistent UI.
"""

from __future__ import annotations

from typing import Final

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication


THEME_COLORS: Final[dict[str, str]] = {
    "background": "#0d0d0d",
    "surface": "#161616",
    "surface_raised": "#1e1e1e",
    "border": "#2a2a2a",
    "accent_primary": "#00d4aa",
    "accent_primary_hover": "#1be4bc",
    "accent_secondary": "#6c63ff",
    "text_primary": "#e8e8e8",
    "text_secondary": "#888888",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#ef4444",
}


THEME_QSS: Final[str] = """
/* Global surfaces */
QWidget {
    background-color: #0d0d0d;
    color: #e8e8e8;
    selection-background-color: rgba(0, 212, 170, 0.20);
    selection-color: #e8e8e8;
}

QMainWindow,
QDialog {
    background-color: #0d0d0d;
}

QFrame#panel,
QWidget#panel,
QGroupBox,
QDockWidget {
    background-color: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
}

QGroupBox {
    margin-top: 12px;
    padding: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #e8e8e8;
}

/* Inputs */
QLineEdit,
QTextEdit,
QPlainTextEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox,
QDateTimeEdit,
QListWidget,
QTreeWidget {
    background-color: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 6px 8px;
}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus,
QDateTimeEdit:focus,
QListWidget:focus,
QTreeWidget:focus {
    border: 1px solid #00d4aa;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    selection-background-color: rgba(0, 212, 170, 0.20);
}

/* Buttons */
QPushButton,
QToolButton {
    background-color: #00d4aa;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    min-height: 16px;
}

QPushButton:hover,
QToolButton:hover {
    background-color: #1be4bc;
}

QPushButton:pressed,
QToolButton:pressed {
    background-color: #00bc95;
}

QPushButton:disabled,
QToolButton:disabled {
    background-color: #2a2a2a;
    color: #888888;
}

QPushButton[secondary="true"],
QToolButton[secondary="true"] {
    background-color: #6c63ff;
}

QPushButton[secondary="true"]:hover,
QToolButton[secondary="true"]:hover {
    background-color: #8078ff;
}

QPushButton[danger="true"],
QToolButton[danger="true"] {
    background-color: #ef4444;
}

QPushButton[danger="true"]:hover,
QToolButton[danger="true"]:hover {
    background-color: #f85a5a;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    top: -1px;
    background-color: #161616;
}

QTabBar::tab {
    background: transparent;
    color: #888888;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 14px;
    margin-right: 8px;
}

QTabBar::tab:selected {
    color: #e8e8e8;
    border-bottom: 2px solid #00d4aa;
}

QTabBar::tab:hover:!selected {
    color: #d7d7d7;
    border-bottom: 2px solid rgba(0, 212, 170, 0.40);
}

/* Table and headers */
QTableView,
QTableWidget {
    background-color: #161616;
    alternate-background-color: #1a1a1a;
    gridline-color: #2a2a2a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    selection-background-color: rgba(0, 212, 170, 0.20);
    selection-color: #e8e8e8;
}

QHeaderView::section {
    background-color: #1e1e1e;
    color: #e8e8e8;
    border: none;
    border-right: 1px solid #2a2a2a;
    border-bottom: 1px solid #2a2a2a;
    padding: 8px;
}

/********************
 * Indicators
 ********************/
QProgressBar {
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    text-align: center;
    color: #e8e8e8;
}

QProgressBar::chunk {
    background-color: #00d4aa;
    border-radius: 6px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #2a2a2a;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #00d4aa;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #161616;
    width: 12px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #2a2a2a;
    min-height: 28px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #3a3a3a;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    height: 0;
    background: none;
}

/* Tooltips and status */
QToolTip {
    background-color: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #2a2a2a;
    padding: 4px 8px;
}

QStatusBar {
    background-color: #161616;
    color: #888888;
    border-top: 1px solid #2a2a2a;
}

/* Monospace targets for logs and metric values */
QTextEdit#logPanel,
QPlainTextEdit#logPanel,
QLineEdit#codeField,
QLabel#metricValue,
QLabel#metricBadge {
    font-family: "JetBrains Mono", "Courier New", monospace;
}

/* Optional card classes used by custom widgets */
QWidget[card="true"] {
    background-color: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 8px;
}

QLabel[role="subtle"] {
    color: #888888;
}

QLabel[role="success"] {
    color: #22c55e;
}

QLabel[role="warning"] {
    color: #f59e0b;
}

QLabel[role="error"] {
    color: #ef4444;
}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the YOLO Studio dark theme to a Qt application.

    Args:
        app: The QApplication instance to theme.
    """

    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(THEME_COLORS["background"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(THEME_COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a1a"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(THEME_COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(THEME_COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(THEME_COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(THEME_COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(THEME_COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(THEME_COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(THEME_COLORS["accent_primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(THEME_COLORS["text_secondary"]))

    app.setPalette(palette)

    # Default UI text uses sans-serif while monospace is targeted through QSS object names.
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(THEME_QSS)


__all__ = ["THEME_COLORS", "THEME_QSS", "apply_theme"]

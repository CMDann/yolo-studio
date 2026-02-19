"""Application theme utilities for YOLO Studio.

This module centralizes the visual design tokens and Qt style rules used by the
application. It provides multiple themes plus helpers to persist and apply
user-selected styles.
"""

from __future__ import annotations

from typing import Final

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

from core.models.database import AppSetting, get_session


THEME_TEMPLATE: Final[str] = """
/* Global surfaces */
QWidget {{
    background-color: {background};
    color: {text_primary};
    selection-background-color: rgba(0, 212, 170, 0.20);
    selection-color: {text_primary};
}}

QMainWindow,
QDialog {{
    background-color: {background};
}}

QFrame#panel,
QWidget#panel,
QGroupBox,
QDockWidget {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
}}

QGroupBox {{
    margin-top: 12px;
    padding: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {text_primary};
}}

/* Inputs */
QLineEdit,
QTextEdit,
QPlainTextEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox,
QDateTimeEdit,
QListWidget,
QTreeWidget {{
    background-color: {surface_raised};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 8px;
}}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus,
QDateTimeEdit:focus,
QListWidget:focus,
QTreeWidget:focus {{
    border: 1px solid {accent_primary};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {surface_raised};
    color: {text_primary};
    border: 1px solid {border};
    selection-background-color: rgba(0, 212, 170, 0.20);
}}

/* Buttons */
QPushButton,
QToolButton {{
    background-color: {accent_primary};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    min-height: 16px;
}}

QPushButton:hover,
QToolButton:hover {{
    background-color: {accent_primary_hover};
}}

QPushButton:pressed,
QToolButton:pressed {{
    background-color: {accent_primary};
}}

QPushButton:disabled,
QToolButton:disabled {{
    background-color: {border};
    color: {text_secondary};
}}

QPushButton[secondary="true"],
QToolButton[secondary="true"] {{
    background-color: {accent_secondary};
}}

QPushButton[secondary="true"]:hover,
QToolButton[secondary="true"]:hover {{
    background-color: {accent_secondary};
}}

QPushButton[danger="true"],
QToolButton[danger="true"] {{
    background-color: {error};
}}

QPushButton[danger="true"]:hover,
QToolButton[danger="true"]:hover {{
    background-color: {error};
}}

/* Tabs */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 6px;
    top: -1px;
    background-color: {surface};
}}

QTabBar::tab {{
    background: transparent;
    color: {text_secondary};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 14px;
    margin-right: 8px;
}}

QTabBar::tab:selected {{
    color: {text_primary};
    border-bottom: 2px solid {accent_primary};
}}

QTabBar::tab:hover:!selected {{
    color: {text_primary};
    border-bottom: 2px solid rgba(0, 212, 170, 0.40);
}}

/* Table and headers */
QTableView,
QTableWidget {{
    background-color: {surface};
    alternate-background-color: {surface_raised};
    gridline-color: {border};
    border: 1px solid {border};
    border-radius: 6px;
    selection-background-color: rgba(0, 212, 170, 0.20);
    selection-color: {text_primary};
}}

QHeaderView::section {{
    background-color: {surface_raised};
    color: {text_primary};
    border: none;
    border-right: 1px solid {border};
    border-bottom: 1px solid {border};
    padding: 8px;
}}

/********************
 * Indicators
 ********************/
QProgressBar {{
    background-color: {surface_raised};
    border: 1px solid {border};
    border-radius: 6px;
    text-align: center;
    color: {text_primary};
}}

QProgressBar::chunk {{
    background-color: {accent_primary};
    border-radius: 6px;
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {border};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {accent_primary};
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: {surface};
    width: 12px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {border};
    min-height: 28px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical:hover {{
    background: {surface_raised};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    height: 0;
    background: none;
}}

/* Tooltips and status */
QToolTip {{
    background-color: {surface_raised};
    color: {text_primary};
    border: 1px solid {border};
    padding: 4px 8px;
}}

QStatusBar {{
    background-color: {surface};
    color: {text_secondary};
    border-top: 1px solid {border};
}}

/* Monospace targets for logs and metric values */
QTextEdit#logPanel,
QPlainTextEdit#logPanel,
QLineEdit#codeField,
QLabel#metricValue,
QLabel#metricBadge {{
    font-family: "JetBrains Mono", "Courier New", monospace;
}}

/* Optional card classes used by custom widgets */
QWidget[card="true"] {{
    background-color: {surface};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px;
}}

QLabel[role="subtle"] {{
    color: {text_secondary};
}}

QLabel[role="success"] {{
    color: {success};
}}

QLabel[role="warning"] {{
    color: {warning};
}}

QLabel[role="error"] {{
    color: {error};
}}
"""


THEMES: Final[dict[str, dict[str, str]]] = {
    "Midnight": {
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
    },
    "Dawn": {
        "background": "#f6f4ef",
        "surface": "#ffffff",
        "surface_raised": "#f3f1ec",
        "border": "#d9d5cc",
        "accent_primary": "#2563eb",
        "accent_primary_hover": "#1d4ed8",
        "accent_secondary": "#0f766e",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
        "success": "#16a34a",
        "warning": "#d97706",
        "error": "#dc2626",
    },
    "High Contrast": {
        "background": "#000000",
        "surface": "#0b0b0b",
        "surface_raised": "#141414",
        "border": "#ffffff",
        "accent_primary": "#ffd400",
        "accent_primary_hover": "#ffea61",
        "accent_secondary": "#00c2ff",
        "text_primary": "#ffffff",
        "text_secondary": "#d1d5db",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#f87171",
    },
    "Ocean": {
        "background": "#0c1b2a",
        "surface": "#122338",
        "surface_raised": "#182c46",
        "border": "#263a53",
        "accent_primary": "#38bdf8",
        "accent_primary_hover": "#60d3ff",
        "accent_secondary": "#22c55e",
        "text_primary": "#e2e8f0",
        "text_secondary": "#94a3b8",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#ef4444",
    },
    "Forest": {
        "background": "#0f1b16",
        "surface": "#15231c",
        "surface_raised": "#1c2f26",
        "border": "#2a3f34",
        "accent_primary": "#10b981",
        "accent_primary_hover": "#34d399",
        "accent_secondary": "#f97316",
        "text_primary": "#e7f1ea",
        "text_secondary": "#9aa5a1",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#ef4444",
    },
    "Slate": {
        "background": "#111827",
        "surface": "#1f2937",
        "surface_raised": "#273244",
        "border": "#3b4252",
        "accent_primary": "#f97316",
        "accent_primary_hover": "#fb923c",
        "accent_secondary": "#3b82f6",
        "text_primary": "#f9fafb",
        "text_secondary": "#9ca3af",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "error": "#ef4444",
    },
}

DEFAULT_THEME: Final[str] = "Midnight"


def available_themes() -> list[str]:
    return list(THEMES.keys())


def resolve_theme_name(name: str | None) -> str:
    if name and name in THEMES:
        return name
    return DEFAULT_THEME


def build_qss(name: str) -> str:
    colors = THEMES[resolve_theme_name(name)]
    return THEME_TEMPLATE.format_map(colors)


def apply_theme(app: QApplication, theme_name: str | None = None) -> None:
    """Apply a selected theme to a Qt application."""

    selected = resolve_theme_name(theme_name)
    colors = THEMES[selected]

    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["background"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface_raised"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["surface"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text_primary"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent_primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors["text_secondary"]))

    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(build_qss(selected))


def load_theme_preference() -> str:
    session = get_session()
    try:
        record = session.query(AppSetting).filter(AppSetting.key == "ui_theme").first()
        if record and isinstance(record.value, str):
            return resolve_theme_name(record.value)
    finally:
        session.close()
    return DEFAULT_THEME


def save_theme_preference(name: str) -> None:
    name = resolve_theme_name(name)
    session = get_session()
    try:
        record = session.query(AppSetting).filter(AppSetting.key == "ui_theme").first()
        if record is None:
            record = AppSetting(key="ui_theme", value=name)
            session.add(record)
        else:
            record.value = name
        session.commit()
    finally:
        session.close()


__all__ = [
    "THEMES",
    "DEFAULT_THEME",
    "available_themes",
    "apply_theme",
    "load_theme_preference",
    "save_theme_preference",
]

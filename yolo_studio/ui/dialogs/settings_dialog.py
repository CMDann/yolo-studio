"""Settings dialog for API tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ui.styles.theme import available_themes, load_theme_preference, save_theme_preference

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"


class SettingsDialog(QDialog):
    """Dialog for managing external API tokens."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(480, 240)

        self._roboflow_input = QLineEdit(self)
        self._roboflow_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._roboflow_input.setPlaceholderText("Roboflow API key")

        self._hf_input = QLineEdit(self)
        self._hf_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_input.setPlaceholderText("Hugging Face token (optional)")

        self._kaggle_user_input = QLineEdit(self)
        self._kaggle_user_input.setPlaceholderText("Kaggle username")

        self._kaggle_key_input = QLineEdit(self)
        self._kaggle_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._kaggle_key_input.setPlaceholderText("Kaggle API key")

        self._theme_combo = QComboBox(self)
        self._theme_combo.addItems(available_themes())

        form = QFormLayout()
        form.addRow("Roboflow API Key", self._roboflow_input)
        form.addRow("Hugging Face Token", self._hf_input)
        form.addRow("Kaggle Username", self._kaggle_user_input)
        form.addRow("Kaggle API Key", self._kaggle_key_input)
        form.addRow("Theme", self._theme_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._load()

    def _load(self) -> None:
        payload = _load_config(CONFIG_PATH)
        self._roboflow_input.setText(str(payload.get("roboflow_api_key", "")))
        self._hf_input.setText(str(payload.get("huggingface_token", "")))
        self._kaggle_user_input.setText(str(payload.get("kaggle_username", "")))
        self._kaggle_key_input.setText(str(payload.get("kaggle_api_key", "")))
        theme_name = load_theme_preference()
        index = self._theme_combo.findText(theme_name)
        if index >= 0:
            self._theme_combo.setCurrentIndex(index)

    def _save(self) -> None:
        payload = _load_config(CONFIG_PATH)
        payload["roboflow_api_key"] = self._roboflow_input.text().strip()
        payload["huggingface_token"] = self._hf_input.text().strip()
        payload["kaggle_username"] = self._kaggle_user_input.text().strip()
        payload["kaggle_api_key"] = self._kaggle_key_input.text().strip()
        try:
            _save_config(CONFIG_PATH, payload)
            save_theme_preference(self._theme_combo.currentText())
        except Exception as exc:
            QMessageBox.critical(self, "Settings", f"Could not save settings: {exc}")
            return
        self.accept()

    def theme_selection(self) -> str:
        return self._theme_combo.currentText()


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_config(config_path: Path, payload: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


__all__ = ["SettingsDialog"]

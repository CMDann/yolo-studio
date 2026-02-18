"""Device card widget used by the Remote tab device grid."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.models.database import RemoteDeviceStatus, RemoteDeviceType


class DeviceCard(QFrame):
    """Compact card showing remote-device identity and current status."""

    clicked = pyqtSignal(int)

    def __init__(
        self,
        device_id: int,
        name: str,
        device_type: RemoteDeviceType,
        host: str,
        port: int,
        status: RemoteDeviceStatus,
        last_seen: datetime | None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize a device card.

        Args:
            device_id: Remote device primary key.
            name: Device display name.
            device_type: Device type enum.
            host: Hostname or IP address.
            port: Device service port.
            status: Current status enum.
            last_seen: Last seen timestamp.
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._device_id = int(device_id)
        self._selected = False

        self.setProperty("card", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._name_label = QLabel(name, self)
        self._name_label.setStyleSheet("font-size: 13px; font-weight: 600;")

        self._type_label = QLabel(self._device_type_text(device_type), self)
        self._type_label.setProperty("role", "subtle")

        self._endpoint_label = QLabel(f"{host}:{port}", self)
        self._endpoint_label.setObjectName("metricValue")

        self._status_badge = QLabel("", self)
        self._status_badge.setObjectName("metricBadge")
        self._status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_badge.setMinimumHeight(22)

        self._last_seen_label = QLabel("", self)
        self._last_seen_label.setProperty("role", "subtle")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self._name_label)
        layout.addWidget(self._type_label)
        layout.addWidget(self._endpoint_label)
        layout.addWidget(self._status_badge)
        layout.addWidget(self._last_seen_label)
        layout.addStretch(1)
        self.setLayout(layout)

        self.update_status(status, last_seen)
        self._refresh_border_style()

    @property
    def device_id(self) -> int:
        """Return the associated remote-device primary key.

        Returns:
            int: Device ID.
        """

        return self._device_id

    def set_selected(self, selected: bool) -> None:
        """Update visual selected state for the card.

        Args:
            selected: True when this card is selected.
        """

        self._selected = bool(selected)
        self._refresh_border_style()

    def update_status(self, status: RemoteDeviceStatus, last_seen: datetime | None) -> None:
        """Update status and last-seen labels.

        Args:
            status: Device status enum.
            last_seen: Last seen timestamp.
        """

        status_text = status.value.capitalize()
        self._status_badge.setText(status_text)

        if status == RemoteDeviceStatus.ONLINE:
            self._status_badge.setProperty("role", "success")
        elif status == RemoteDeviceStatus.OFFLINE:
            self._status_badge.setProperty("role", "error")
        else:
            self._status_badge.setProperty("role", "warning")

        if last_seen is None:
            self._last_seen_label.setText("Last seen: -")
        else:
            text = last_seen.isoformat(sep=" ", timespec="seconds")
            self._last_seen_label.setText(f"Last seen: {text}")

        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Emit click signal when user selects the card.

        Args:
            event: Mouse press event.
        """

        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._device_id)
        super().mousePressEvent(event)

    def _refresh_border_style(self) -> None:
        """Apply selected/unselected border style."""

        if self._selected:
            self.setStyleSheet(
                "QFrame { border: 1px solid #00d4aa; border-radius: 6px; background-color: #161616; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { border: 1px solid #2a2a2a; border-radius: 6px; background-color: #161616; }"
            )

    @staticmethod
    def _device_type_text(device_type: RemoteDeviceType) -> str:
        """Map enum to display text.

        Args:
            device_type: Device type enum.

        Returns:
            str: Display label.
        """

        mapping = {
            RemoteDeviceType.JETSON_NANO: "Jetson Nano",
            RemoteDeviceType.XAVIER: "Jetson Xavier",
            RemoteDeviceType.RASPBERRY_PI: "Raspberry Pi",
            RemoteDeviceType.OTHER: "Other Device",
        }
        return mapping.get(device_type, "Other Device")


__all__ = ["DeviceCard"]

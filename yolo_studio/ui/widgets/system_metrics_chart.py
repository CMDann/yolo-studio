"""Live system metrics chart widget built on top of pyqtgraph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional dependency fallback
    pg = None


@dataclass
class _SeriesBuffer:
    x: list[int] = field(default_factory=list)
    y: list[float] = field(default_factory=list)


class SystemMetricsChart(QWidget):
    """Render live CPU, memory, GPU, and temperature metrics."""

    UTIL_KEYS: Final[tuple[str, ...]] = ("cpu_percent", "memory_percent", "gpu_percent")
    TEMP_KEYS: Final[tuple[str, ...]] = ("cpu_temp_c", "gpu_temp_c")

    UTIL_COLORS: Final[dict[str, str]] = {
        "cpu_percent": "#22c55e",
        "memory_percent": "#f59e0b",
        "gpu_percent": "#60a5fa",
    }

    TEMP_COLORS: Final[dict[str, str]] = {
        "cpu_temp_c": "#ef4444",
        "gpu_temp_c": "#a855f7",
    }

    UTIL_LABELS: Final[dict[str, str]] = {
        "cpu_percent": "CPU %",
        "memory_percent": "Memory %",
        "gpu_percent": "GPU %",
    }

    TEMP_LABELS: Final[dict[str, str]] = {
        "cpu_temp_c": "CPU °C",
        "gpu_temp_c": "GPU °C",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._series_data: dict[str, _SeriesBuffer] = {
            key: _SeriesBuffer() for key in (*self.UTIL_KEYS, *self.TEMP_KEYS)
        }
        self._curves: dict[str, object] = {}
        self._max_points = 180

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        if pg is None:
            label = QLabel("pyqtgraph is not installed. System metrics unavailable.", self)
            label.setWordWrap(True)
            label.setProperty("role", "warning")
            layout.addWidget(label)
            self.setLayout(layout)
            self._util_plot = None
            self._temp_plot = None
            return

        self._util_plot = self._build_plot("System Utilization (%)")
        self._temp_plot = self._build_plot("Temperatures (°C)")

        for key in self.UTIL_KEYS:
            curve = self._util_plot.plot(
                [],
                [],
                pen=pg.mkPen(color=self.UTIL_COLORS[key], width=2),
                name=self.UTIL_LABELS[key],
            )
            self._curves[key] = curve

        for key in self.TEMP_KEYS:
            curve = self._temp_plot.plot(
                [],
                [],
                pen=pg.mkPen(color=self.TEMP_COLORS[key], width=2),
                name=self.TEMP_LABELS[key],
            )
            self._curves[key] = curve

        layout.addWidget(self._util_plot)
        layout.addWidget(self._temp_plot)
        self.setLayout(layout)

    def _build_plot(self, title: str) -> "pg.PlotWidget":
        plot = pg.PlotWidget(self)
        plot.setBackground("#161616")
        plot.addLegend(offset=(10, 10))
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setTitle(title, color="#e8e8e8")
        plot.setLabel("left", "Value")
        plot.setLabel("bottom", "Sample")

        axis_pen = pg.mkPen(color="#888888", width=1)
        plot.getAxis("left").setPen(axis_pen)
        plot.getAxis("bottom").setPen(axis_pen)
        plot.getAxis("left").setTextPen("#e8e8e8")
        plot.getAxis("bottom").setTextPen("#e8e8e8")
        return plot

    def reset(self) -> None:
        for key, series in self._series_data.items():
            series.x.clear()
            series.y.clear()
            curve = self._curves.get(key)
            if curve is not None:
                curve.setData([], [])

    def append_sample(self, index: int, metrics: dict[str, float | None]) -> None:
        if pg is None:
            return

        for key, value in metrics.items():
            if key not in self._series_data:
                continue
            if value is None:
                continue

            series = self._series_data[key]
            series.x.append(index)
            series.y.append(float(value))

            if len(series.x) > self._max_points:
                series.x.pop(0)
                series.y.pop(0)

            curve = self._curves.get(key)
            if curve is not None:
                curve.setData(series.x, series.y)


__all__ = ["SystemMetricsChart"]

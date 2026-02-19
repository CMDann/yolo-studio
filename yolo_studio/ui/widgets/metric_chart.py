"""Live training metric chart widget built on top of pyqtgraph."""

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
    """In-memory points buffer for a plotted metric series."""

    x: list[int] = field(default_factory=list)
    y: list[float] = field(default_factory=list)


class MetricChart(QWidget):
    """Render live YOLO training metrics in a multi-series chart."""

    METRIC_KEYS: Final[tuple[str, ...]] = (
        "train/box_loss",
        "train/cls_loss",
        "val/box_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )

    METRIC_COLORS: Final[dict[str, str]] = {
        "train/box_loss": "#00d4aa",
        "train/cls_loss": "#6c63ff",
        "val/box_loss": "#f59e0b",
        "metrics/mAP50(B)": "#22c55e",
        "metrics/mAP50-95(B)": "#ef4444",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the chart widget.

        Args:
            parent: Optional parent widget.
        """

        super().__init__(parent)

        self._series_data: dict[str, _SeriesBuffer] = {
            metric: _SeriesBuffer() for metric in self.METRIC_KEYS
        }
        self._curves: dict[str, object] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if pg is None:
            self._fallback_label = QLabel("pyqtgraph is not installed. Live chart unavailable.", self)
            self._fallback_label.setWordWrap(True)
            self._fallback_label.setProperty("role", "warning")
            layout.addWidget(self._fallback_label)
            self.setLayout(layout)
            return

        self._plot = pg.PlotWidget(self)
        self._plot.setBackground("#161616")
        self._plot.addLegend(offset=(10, 10))
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("left", "Metric")
        self._plot.setLabel("bottom", "Epoch")

        axis_pen = pg.mkPen(color="#888888", width=1)
        self._plot.getAxis("left").setPen(axis_pen)
        self._plot.getAxis("bottom").setPen(axis_pen)
        self._plot.getAxis("left").setTextPen("#e8e8e8")
        self._plot.getAxis("bottom").setTextPen("#e8e8e8")

        for metric in self.METRIC_KEYS:
            curve = self._plot.plot(
                [],
                [],
                pen=pg.mkPen(color=self.METRIC_COLORS[metric], width=2),
                name=metric,
            )
            self._curves[metric] = curve

        layout.addWidget(self._plot)
        self.setLayout(layout)

    def reset(self) -> None:
        """Clear all metric series and reset the chart."""

        for metric in self.METRIC_KEYS:
            self._series_data[metric].x.clear()
            self._series_data[metric].y.clear()
            curve = self._curves.get(metric)
            if curve is not None:
                curve.setData([], [])

    def append_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Append metrics for a single training epoch.

        Args:
            epoch: 1-based epoch index.
            metrics: Mapping from metric key to numeric value.
        """

        if pg is None:
            return

        for metric in self.METRIC_KEYS:
            value = metrics.get(metric)
            if value is None:
                continue

            series = self._series_data[metric]
            # Update-in-place if the same epoch arrives multiple times from different callbacks.
            if series.x and series.x[-1] == epoch:
                series.y[-1] = float(value)
            else:
                series.x.append(epoch)
                series.y.append(float(value))

            curve = self._curves.get(metric)
            if curve is not None:
                curve.setData(
                    series.x,
                    series.y,
                )


__all__ = ["MetricChart"]

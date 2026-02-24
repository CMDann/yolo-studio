"""System metrics and hardware detection helpers."""

from __future__ import annotations

from typing import Any


_NVML_INIT: bool = False


def list_compute_devices() -> list[tuple[str, str]]:
    """Return available compute device labels and values for Ultralytics.

    Values are compatible with Ultralytics device arguments.
    """

    devices: list[tuple[str, str]] = [
        ("Auto (GPU if available)", "auto"),
        ("CPU", "cpu"),
    ]

    try:
        import torch
    except Exception:
        return devices

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(("Apple MPS", "mps"))
    except Exception:
        # MPS availability check can fail in some environments; ignore it.
        pass

    try:
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for idx in range(count):
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = "CUDA"
                devices.append((f"GPU {idx} ({name})", f"cuda:{idx}"))
    except Exception:
        # CUDA detection can fail if torch lacks CUDA bindings.
        pass

    return devices


def resolve_device_value(value: Any) -> str | None:
    """Normalize a raw device selection value into a usable device string."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.lower() == "auto":
        return None

    return text


def read_system_metrics() -> dict[str, float | None]:
    """Collect system utilization metrics for live dashboards.

    Returns a dictionary containing:
    - cpu_percent
    - memory_percent
    - cpu_temp_c
    - gpu_percent
    - gpu_temp_c
    """

    metrics: dict[str, float | None] = {
        "cpu_percent": None,
        "memory_percent": None,
        "cpu_temp_c": None,
        "gpu_percent": None,
        "gpu_temp_c": None,
    }

    try:
        import psutil
    except Exception:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        try:
            metrics["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass

        try:
            metrics["memory_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
            pass

        try:
            temps = psutil.sensors_temperatures(fahrenheit=False)
        except Exception:
            temps = {}

        if temps:
            all_temps: list[float] = []
            for entries in temps.values():
                for entry in entries:
                    value = getattr(entry, "current", None)
                    if isinstance(value, (int, float)):
                        all_temps.append(float(value))
            if all_temps:
                metrics["cpu_temp_c"] = max(all_temps)

    _populate_gpu_metrics(metrics)
    return metrics


def _populate_gpu_metrics(metrics: dict[str, float | None]) -> None:
    """Fill GPU utilization and temperature via NVML if present."""

    global _NVML_INIT

    try:
        import pynvml
    except Exception:
        return

    try:
        if not _NVML_INIT:
            pynvml.nvmlInit()
            _NVML_INIT = True

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count < 1:
            return

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_percent"] = float(util.gpu)

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics["gpu_temp_c"] = float(temp)
        except Exception:
            pass
    except Exception:
        return


__all__ = ["list_compute_devices", "resolve_device_value", "read_system_metrics"]

"""QThread-based YOLO training orchestration for YOLO Studio.

This module provides the `YOLOTrainer` worker used by the Training tab to run
Ultralytics training without blocking the main Qt event loop.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal

from core.models.database import TrainingRun, TrainingRunStatus, get_session


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Normalized configuration payload for a YOLO training run."""

    run_name: str
    notes: str
    project_id: int | None
    dataset_id: int
    model_architecture: str
    task: str
    data_yaml_path: str
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float
    optimizer: str
    warmup_epochs: float
    weight_decay: float
    augmentation: dict[str, bool]
    use_pretrained: bool
    custom_weights_path: str | None
    output_root: str
    device: str | None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TrainingConfig":
        """Build a validated training config from a plain dictionary.

        Args:
            mapping: Raw config map supplied by the UI.

        Returns:
            TrainingConfig: Parsed configuration.

        Raises:
            ValueError: If required values are missing or invalid.
        """

        required_keys = (
            "run_name",
            "dataset_id",
            "model_architecture",
            "task",
            "data_yaml_path",
            "epochs",
            "batch_size",
            "image_size",
            "learning_rate",
            "optimizer",
            "warmup_epochs",
            "weight_decay",
            "use_pretrained",
            "output_root",
        )
        missing = [key for key in required_keys if key not in mapping]
        if missing:
            raise ValueError(f"Training config missing required keys: {', '.join(missing)}")

        run_name = str(mapping["run_name"]).strip()
        if not run_name:
            raise ValueError("run_name must not be empty")

        data_yaml_path = str(mapping["data_yaml_path"]).strip()
        if not data_yaml_path or not Path(data_yaml_path).exists():
            raise ValueError("data_yaml_path is required and must exist")

        custom_weights = mapping.get("custom_weights_path")
        if custom_weights is not None:
            custom_weights = str(custom_weights).strip() or None
            if custom_weights and not Path(custom_weights).exists():
                raise ValueError("custom_weights_path does not exist")

        augmentation_raw = mapping.get("augmentation", {})
        augmentation = {
            "mosaic": bool(augmentation_raw.get("mosaic", True)),
            "mixup": bool(augmentation_raw.get("mixup", False)),
            "copy_paste": bool(augmentation_raw.get("copy_paste", False)),
            "hsv": bool(augmentation_raw.get("hsv", True)),
            "flip": bool(augmentation_raw.get("flip", True)),
        }

        return cls(
            run_name=run_name,
            notes=str(mapping.get("notes", "")).strip(),
            project_id=int(mapping["project_id"]) if mapping.get("project_id") is not None else None,
            dataset_id=int(mapping["dataset_id"]),
            model_architecture=str(mapping["model_architecture"]),
            task=str(mapping["task"]),
            data_yaml_path=data_yaml_path,
            epochs=int(mapping["epochs"]),
            batch_size=int(mapping["batch_size"]),
            image_size=int(mapping["image_size"]),
            learning_rate=float(mapping["learning_rate"]),
            optimizer=str(mapping["optimizer"]),
            warmup_epochs=float(mapping["warmup_epochs"]),
            weight_decay=float(mapping["weight_decay"]),
            augmentation=augmentation,
            use_pretrained=bool(mapping["use_pretrained"]),
            custom_weights_path=custom_weights,
            output_root=str(mapping["output_root"]),
            device=str(mapping["device"]).strip() if mapping.get("device") else None,
        )


class _SignalStream:
    """Text stream used to forward stdout/stderr output into Qt log signals."""

    def __init__(self, emit_line: Callable[[str], None]) -> None:
        """Initialize the stream.

        Args:
            emit_line: Callback to dispatch each processed line.
        """

        self._emit_line = emit_line
        self._buffer = ""

    def write(self, text: str) -> int:
        """Write text and emit complete lines.

        Args:
            text: Stream chunk.

        Returns:
            int: Number of characters consumed.
        """

        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._emit_line(line)
        return len(text)

    def flush(self) -> None:
        """Flush any buffered partial line."""

        if self._buffer.strip():
            self._emit_line(self._buffer.strip())
        self._buffer = ""


class YOLOTrainer(QThread):
    """Background worker that executes Ultralytics YOLO training jobs."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    metrics = pyqtSignal(dict)
    log = pyqtSignal(str)

    def __init__(self, config: dict[str, Any], parent: Any | None = None) -> None:
        """Initialize a training worker instance.

        Args:
            config: Training configuration dictionary from the UI.
            parent: Optional Qt parent object.
        """

        super().__init__(parent)

        self._raw_config = dict(config)
        self._config = TrainingConfig.from_mapping(self._raw_config)

        self._stop_requested = Event()

        self._training_run_id: int | None = None
        self._best_map50: float | None = None
        self._best_map50_95: float | None = None
        self._final_loss: float | None = None
        self._save_dir: Path | None = None
        self._weights_path: Path | None = None
        self._last_emitted_epoch: int = 0

        self._yolo_model: Any | None = None

    def request_stop(self) -> None:
        """Request graceful early stopping at the next callback checkpoint."""

        self._stop_requested.set()
        self.status.emit("Graceful stop requested by user.")
        self.log.emit("Graceful stop requested by user.")

    def run(self) -> None:
        """Execute a full training lifecycle in the worker thread."""

        self.progress.emit(0)
        self.status.emit("Initializing training run...")

        try:
            self._training_run_id = self._create_training_run_record()
            self._set_training_run_status(TrainingRunStatus.RUNNING)
            self.log.emit(f"TrainingRun #{self._training_run_id} created.")

            result = self._train_with_ultralytics()

            self._set_training_run_success(result)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:
            LOGGER.exception("Training worker failed.")
            self._set_training_run_failure(str(exc))
            self.error.emit(str(exc))

    def _train_with_ultralytics(self) -> dict[str, Any]:
        """Run model training and collect final result fields.

        Returns:
            dict[str, Any]: Result payload emitted to UI.

        Raises:
            RuntimeError: If Ultralytics is unavailable or training fails.
        """

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies from requirements.txt"
            ) from exc

        model_source = self._resolve_model_source()
        output_root = Path(self._config.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        self.status.emit(f"Loading model: {model_source}")
        self._yolo_model = YOLO(model_source)

        self._register_callbacks()

        train_kwargs: dict[str, Any] = {
            "data": self._config.data_yaml_path,
            "epochs": self._config.epochs,
            "imgsz": self._config.image_size,
            "batch": self._config.batch_size,
            "task": self._config.task,
            "lr0": self._config.learning_rate,
            "optimizer": self._config.optimizer,
            "warmup_epochs": self._config.warmup_epochs,
            "weight_decay": self._config.weight_decay,
            "project": str(output_root),
            "name": self._build_run_slug(self._config.run_name),
            "exist_ok": False,
            "pretrained": self._config.use_pretrained,
            "mosaic": 1.0 if self._config.augmentation["mosaic"] else 0.0,
            "mixup": 0.2 if self._config.augmentation["mixup"] else 0.0,
            "copy_paste": 0.2 if self._config.augmentation["copy_paste"] else 0.0,
            "hsv_h": 0.015 if self._config.augmentation["hsv"] else 0.0,
            "hsv_s": 0.7 if self._config.augmentation["hsv"] else 0.0,
            "hsv_v": 0.4 if self._config.augmentation["hsv"] else 0.0,
            "fliplr": 0.5 if self._config.augmentation["flip"] else 0.0,
        }

        if self._config.device:
            train_kwargs["device"] = self._config.device

        self.log.emit(f"Training args: {json.dumps(train_kwargs, default=str)}")

        stream = _SignalStream(self.log.emit)
        try:
            # Ultralytics writes rich logs to stdout/stderr; route those lines into the GUI log panel.
            with redirect_stdout(stream), redirect_stderr(stream):
                self._yolo_model.train(**train_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Ultralytics training failed: {exc}") from exc
        finally:
            stream.flush()

        trainer = getattr(self._yolo_model, "trainer", None)
        if trainer is not None:
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir:
                self._save_dir = Path(str(save_dir)).resolve()

        self._weights_path = self._resolve_weights_path(self._save_dir)

        result = {
            "training_run_id": self._training_run_id,
            "run_name": self._config.run_name,
            "model_architecture": self._config.model_architecture,
            "status": "completed",
            "stopped_early": self._stop_requested.is_set(),
            "output_dir": str(self._save_dir) if self._save_dir else None,
            "weights_path": str(self._weights_path) if self._weights_path else None,
            "best_map50": self._best_map50,
            "best_map50_95": self._best_map50_95,
            "final_loss": self._final_loss,
        }

        self.status.emit("Training complete.")
        return result

    def _register_callbacks(self) -> None:
        """Register Ultralytics callbacks used for progress and metrics."""

        if self._yolo_model is None:
            return

        callback_map = {
            "on_train_start": self._on_train_start,
            "on_train_epoch_end": self._on_train_epoch_end,
            "on_fit_epoch_end": self._on_fit_epoch_end,
            "on_train_end": self._on_train_end,
        }

        for event_name, handler in callback_map.items():
            try:
                self._yolo_model.add_callback(event_name, handler)
            except Exception:
                # Some versions expose different callback sets; training still works without optional hooks.
                LOGGER.debug("Ultralytics callback registration skipped: %s", event_name, exc_info=True)

    def _on_train_start(self, trainer: Any) -> None:
        """Handle Ultralytics train-start callback.

        Args:
            trainer: Ultralytics trainer context.
        """

        _ = trainer
        self.status.emit("Training started.")
        self.log.emit("Training started.")

    def _on_fit_epoch_end(self, trainer: Any) -> None:
        """Handle fit-epoch callback to emit progress and metric updates.

        Args:
            trainer: Ultralytics trainer context.
        """

        self._emit_epoch_update(trainer)

    def _on_train_epoch_end(self, trainer: Any) -> None:
        """Handle train-epoch callback and provide metric fallback updates.

        Args:
            trainer: Ultralytics trainer context.
        """

        self._emit_epoch_update(trainer)

    def _on_train_end(self, trainer: Any) -> None:
        """Handle Ultralytics train-end callback.

        Args:
            trainer: Ultralytics trainer context.
        """

        _ = trainer
        if self._stop_requested.is_set():
            self.status.emit("Training stopped early by user request.")
            self.log.emit("Training stopped early by user request.")

    def _extract_metrics(self, trainer: Any) -> dict[str, float]:
        """Extract normalized metrics from an Ultralytics trainer context.

        Args:
            trainer: Ultralytics trainer context.

        Returns:
            dict[str, float]: Metric mapping compatible with UI chart keys.
        """

        payload: dict[str, float] = {}

        # The trainer may provide canonical train loss labels in current versions.
        try:
            label_loss_items = getattr(trainer, "label_loss_items", None)
            tloss = getattr(trainer, "tloss", None)
            if callable(label_loss_items) and tloss is not None:
                labeled = label_loss_items(tloss, prefix="train")
                if isinstance(labeled, dict):
                    for key, value in labeled.items():
                        numeric = self._to_float(value)
                        if numeric is not None:
                            payload[str(key)] = numeric
        except Exception:
            LOGGER.debug("Unable to extract labeled loss items.", exc_info=True)

        # Fallback loss extraction from raw tensor/list values.
        if "train/box_loss" not in payload or "train/cls_loss" not in payload:
            tloss_values = self._to_float_list(getattr(trainer, "tloss", None))
            if tloss_values:
                if "train/box_loss" not in payload and len(tloss_values) >= 1:
                    payload["train/box_loss"] = tloss_values[0]
                if "train/cls_loss" not in payload and len(tloss_values) >= 2:
                    payload["train/cls_loss"] = tloss_values[1]
                if "train/dfl_loss" not in payload and len(tloss_values) >= 3:
                    payload["train/dfl_loss"] = tloss_values[2]

        raw_metrics: dict[str, float] = {}
        trainer_metrics = getattr(trainer, "metrics", None)
        if isinstance(trainer_metrics, dict):
            for key, value in trainer_metrics.items():
                numeric = self._to_float(value)
                if numeric is not None:
                    raw_metrics[str(key)] = numeric

        validator = getattr(trainer, "validator", None)
        validator_metrics = getattr(validator, "metrics", None) if validator is not None else None
        results_dict = getattr(validator_metrics, "results_dict", None) if validator_metrics is not None else None
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                numeric = self._to_float(value)
                if numeric is not None:
                    raw_metrics[str(key)] = numeric

        # Normalize mAP keys to the exact names used by the UI.
        map50_value = self._pick_first_metric(
            raw_metrics,
            ("metrics/mAP50(B)", "mAP50", "map50", "metrics/mAP50"),
        )
        if map50_value is not None:
            payload["metrics/mAP50(B)"] = map50_value

        map50_95_value = self._pick_first_metric(
            raw_metrics,
            ("metrics/mAP50-95(B)", "mAP50-95", "map", "metrics/mAP50-95"),
        )
        if map50_95_value is not None:
            payload["metrics/mAP50-95(B)"] = map50_95_value

        val_box_loss = self._pick_first_metric(
            raw_metrics,
            ("val/box_loss", "metrics/val/box_loss", "box_loss"),
        )
        if val_box_loss is not None:
            payload["val/box_loss"] = val_box_loss

        return payload

    def _emit_epoch_update(self, trainer: Any) -> None:
        """Emit normalized progress/metric payload for the current epoch.

        Args:
            trainer: Ultralytics trainer context.
        """

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", self._config.epochs))
        if epoch <= 0:
            return

        metric_payload = self._extract_metrics(trainer)
        csv_payload = self._extract_metrics_from_results_csv(trainer, expected_epoch=epoch)
        if csv_payload:
            for key, value in csv_payload.items():
                metric_payload.setdefault(key, value)

        # Avoid emitting redundant callbacks when neither epoch nor metric values changed.
        if epoch < self._last_emitted_epoch:
            return
        if epoch == self._last_emitted_epoch and not metric_payload:
            return

        map50 = metric_payload.get("metrics/mAP50(B)")
        if map50 is not None:
            self._best_map50 = map50 if self._best_map50 is None else max(self._best_map50, map50)

        map50_95 = metric_payload.get("metrics/mAP50-95(B)")
        if map50_95 is not None:
            self._best_map50_95 = map50_95 if self._best_map50_95 is None else max(self._best_map50_95, map50_95)

        loss_values = [
            metric_payload.get("train/box_loss"),
            metric_payload.get("train/cls_loss"),
            metric_payload.get("train/dfl_loss"),
        ]
        numeric_losses = [value for value in loss_values if isinstance(value, float)]
        if numeric_losses:
            self._final_loss = sum(numeric_losses)

        if self._stop_requested.is_set():
            setattr(trainer, "stop", True)

        progress_percent = int((epoch / max(total_epochs, 1)) * 100)
        progress_percent = max(0, min(progress_percent, 100))

        self._last_emitted_epoch = max(self._last_emitted_epoch, epoch)
        self.progress.emit(progress_percent)
        self.status.emit(f"Epoch {epoch}/{total_epochs} complete")
        self.metrics.emit(
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "metrics": metric_payload,
            }
        )

    def _extract_metrics_from_results_csv(self, trainer: Any, expected_epoch: int | None = None) -> dict[str, float]:
        """Load latest metric values from trainer `results.csv` as fallback.

        Args:
            trainer: Ultralytics trainer context.
            expected_epoch: Optional 1-based epoch to match against CSV row.

        Returns:
            dict[str, float]: Parsed metric values from the latest CSV row.
        """

        csv_path_raw = getattr(trainer, "csv", None)
        if csv_path_raw is None:
            return {}

        csv_path = Path(str(csv_path_raw))
        if not csv_path.exists() or not csv_path.is_file():
            return {}

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                last_row: dict[str, str] | None = None
                for row in reader:
                    last_row = row
        except Exception:
            LOGGER.debug("Unable to parse Ultralytics results CSV: %s", csv_path, exc_info=True)
            return {}

        if not last_row:
            return {}

        if expected_epoch is not None:
            row_epoch = self._to_float(last_row.get("epoch"))
            if row_epoch is None or int(row_epoch) != int(expected_epoch):
                return {}

        payload: dict[str, float] = {}
        for key in (
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ):
            raw_value = last_row.get(key)
            if raw_value is None:
                continue
            numeric = self._to_float(raw_value)
            if numeric is not None:
                payload[key] = numeric

        return payload

    def _create_training_run_record(self) -> int:
        """Insert the initial TrainingRun row and return its primary key.

        Returns:
            int: Newly created TrainingRun ID.

        Raises:
            RuntimeError: If DB insert fails.
        """

        session = get_session()
        try:
            run = TrainingRun(
                name=self._config.run_name,
                notes=self._config.notes or None,
                project_id=self._config.project_id,
                dataset_id=self._config.dataset_id,
                model_architecture=self._config.model_architecture,
                image_size=self._config.image_size,
                batch_size=self._config.batch_size,
                epochs=self._config.epochs,
                learning_rate=self._config.learning_rate,
                status=TrainingRunStatus.PENDING,
                config_yaml=self._raw_config,
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return int(run.id)
        except Exception as exc:
            session.rollback()
            raise RuntimeError(f"Unable to create training run record: {exc}") from exc
        finally:
            session.close()

    def _set_training_run_status(self, status: TrainingRunStatus) -> None:
        """Set the current TrainingRun lifecycle status.

        Args:
            status: New run status.
        """

        if self._training_run_id is None:
            return

        session = get_session()
        try:
            run = session.get(TrainingRun, self._training_run_id)
            if run is None:
                return
            run.status = status
            session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to update training run status to %s", status.value)
        finally:
            session.close()

    def _set_training_run_success(self, result: dict[str, Any]) -> None:
        """Persist final successful run values into SQLite.

        Args:
            result: Final training result payload.
        """

        if self._training_run_id is None:
            return

        session = get_session()
        try:
            run = session.get(TrainingRun, self._training_run_id)
            if run is None:
                return

            run.status = TrainingRunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)
            run.best_map50 = self._best_map50
            run.best_map50_95 = self._best_map50_95
            run.final_loss = self._final_loss
            run.output_dir = result.get("output_dir")
            run.weights_path = result.get("weights_path")
            run.config_yaml = self._raw_config

            session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to persist successful training run state.")
        finally:
            session.close()

    def _set_training_run_failure(self, error_message: str) -> None:
        """Persist failure state and attach diagnostic message to notes.

        Args:
            error_message: Human-readable failure reason.
        """

        if self._training_run_id is None:
            return

        session = get_session()
        try:
            run = session.get(TrainingRun, self._training_run_id)
            if run is None:
                return

            run.status = TrainingRunStatus.FAILED
            run.completed_at = datetime.now(timezone.utc)

            error_note = f"\n[ERROR] {error_message}".strip()
            if run.notes:
                run.notes = f"{run.notes}\n{error_note}".strip()
            else:
                run.notes = error_note

            session.commit()
        except Exception:
            session.rollback()
            LOGGER.exception("Failed to persist failed training run state.")
        finally:
            session.close()

    def _resolve_model_source(self) -> str:
        """Resolve model source path/name for Ultralytics YOLO initialization.

        Returns:
            str: Weight file path or architecture spec.
        """

        if self._config.custom_weights_path:
            return str(Path(self._config.custom_weights_path).resolve())

        architecture = str(self._config.model_architecture).strip()
        if architecture:
            candidate = Path(architecture)
            if candidate.exists():
                return str(candidate.resolve())
            if architecture.endswith((".pt", ".yaml", ".yml")):
                return architecture

        if self._config.use_pretrained:
            return f"{self._config.model_architecture}.pt"

        return f"{self._config.model_architecture}.yaml"

    @staticmethod
    def _resolve_weights_path(save_dir: Path | None) -> Path | None:
        """Resolve the best available weights path from an Ultralytics run directory.

        Args:
            save_dir: Run save directory.

        Returns:
            Path | None: Existing weights path if found.
        """

        if save_dir is None:
            return None

        candidate_best = save_dir / "weights" / "best.pt"
        if candidate_best.exists():
            return candidate_best.resolve()

        candidate_last = save_dir / "weights" / "last.pt"
        if candidate_last.exists():
            return candidate_last.resolve()

        return None

    @staticmethod
    def _build_run_slug(run_name: str) -> str:
        """Create filesystem-safe slug used as Ultralytics run folder name.

        Args:
            run_name: Original run name.

        Returns:
            str: Sanitized run folder slug.
        """

        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", run_name).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{cleaned or 'run'}_{timestamp}"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert scalar-like values to float when possible.

        Args:
            value: Candidate scalar value.

        Returns:
            float | None: Parsed float, otherwise None.
        """

        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return None

    @classmethod
    def _to_float_list(cls, value: Any) -> list[float]:
        """Convert sequence-like values to a list of floats.

        Args:
            value: Candidate sequence object.

        Returns:
            list[float]: Parsed numeric list.
        """

        if value is None:
            return []

        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().tolist()
            elif hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            return []

        if not isinstance(value, (list, tuple)):
            return []

        result: list[float] = []
        for item in value:
            numeric = cls._to_float(item)
            if numeric is not None:
                result.append(numeric)
        return result

    @staticmethod
    def _pick_first_metric(metrics: dict[str, float], candidates: tuple[str, ...]) -> float | None:
        """Pick the first matching metric from an exact-or-substring candidate list.

        Args:
            metrics: Available metric dictionary.
            candidates: Ordered preferred metric keys.

        Returns:
            float | None: Selected metric value if found.
        """

        for candidate in candidates:
            if candidate in metrics:
                return metrics[candidate]

        lower_lookup = {key.lower(): value for key, value in metrics.items()}
        for candidate in candidates:
            candidate_lower = candidate.lower()
            for key, value in lower_lookup.items():
                if candidate_lower in key:
                    return value

        return None


__all__ = ["YOLOTrainer", "TrainingConfig"]

"""YOLO Studio edge-device WebSocket agent.

# systemd service example (`/etc/systemd/system/yolo-studio-agent.service`)
# [Unit]
# Description=YOLO Studio Edge Agent
# After=network-online.target
# Wants=network-online.target
#
# [Service]
# Type=simple
# WorkingDirectory=/opt/yolo_studio/edge
# ExecStart=/usr/bin/python3 /opt/yolo_studio/edge/jetson_agent.py --config /opt/yolo_studio/edge/agent_config.yaml
# Restart=always
# RestartSec=3
# User=ubuntu
# Group=ubuntu
#
# [Install]
# WantedBy=multi-user.target

This standalone script runs on Jetson Nano/Xavier/Raspberry Pi targets and
handles remote deployment/testing requests from the YOLO Studio desktop app.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import platform
import shutil
import socket
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


try:
    import websockets
except Exception as exc:  # pragma: no cover - dependency import guard
    raise RuntimeError("The 'websockets' package is required on the edge agent.") from exc


LOGGER = logging.getLogger("yolo_studio.edge.agent")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("agent_config.yaml")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class AgentConfig:
    """Runtime configuration loaded from `agent_config.yaml`."""

    host: str
    port: int
    auth_token: str
    model_cache_dir: Path
    output_dir: Path
    chunk_size: int
    log_file: Path

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AgentConfig":
        """Load and validate agent config from a YAML file.

        Args:
            config_path: Path to YAML config.

        Returns:
            AgentConfig: Parsed configuration instance.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If required fields are invalid.
        """

        if not config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("agent_config.yaml must contain a top-level mapping.")

        base_dir = config_path.resolve().parent

        host = str(raw.get("host") or "0.0.0.0").strip()
        port = int(raw.get("port") or 8765)
        auth_token = str(raw.get("auth_token") or "").strip()

        model_cache_dir = _resolve_path(base_dir, raw.get("model_cache_dir") or "./model_cache")
        output_dir = _resolve_path(base_dir, raw.get("output_dir") or "./results")
        log_file = _resolve_path(base_dir, raw.get("log_file") or "./agent.log")
        chunk_size = int(raw.get("chunk_size") or 200_000)

        if not host:
            raise ValueError("Config 'host' must be set.")
        if port <= 0 or port > 65535:
            raise ValueError("Config 'port' must be between 1 and 65535.")
        if not auth_token:
            raise ValueError("Config 'auth_token' must be set.")
        if chunk_size < 1_024:
            raise ValueError("Config 'chunk_size' must be at least 1024 bytes.")

        model_cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        return cls(
            host=host,
            port=port,
            auth_token=auth_token,
            model_cache_dir=model_cache_dir,
            output_dir=output_dir,
            chunk_size=chunk_size,
            log_file=log_file,
        )


@dataclass(slots=True)
class InferenceArtifact:
    """Stored inference outputs retrievable through `GET_RESULTS`."""

    result_id: str
    model_name: str
    dataset_path: str
    metrics: dict[str, float | None]
    num_images_tested: int
    output_dir: Path
    image_paths: list[Path]
    created_at: datetime


class EdgeAgent:
    """WebSocket server that executes remote model deployment and inference."""

    def __init__(self, config: AgentConfig) -> None:
        """Initialize server state.

        Args:
            config: Agent runtime configuration.
        """

        self._config = config
        self._shutdown_event = asyncio.Event()
        self._results: dict[str, InferenceArtifact] = {}

    async def serve(self) -> None:
        """Start and run the WebSocket server until shutdown is requested."""

        self._print_startup_banner()

        async with websockets.serve(
            self._handle_connection,
            self._config.host,
            self._config.port,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ):
            LOGGER.info("Edge agent listening on ws://%s:%d", self._config.host, self._config.port)
            await self._shutdown_event.wait()

        LOGGER.info("Edge agent stopped.")

    async def _handle_connection(self, websocket: Any, _path: str | None = None) -> None:
        """Handle incoming client messages for a connected WebSocket.

        Args:
            websocket: Client WebSocket connection.
            _path: Optional route path provided by some websockets versions.
        """

        peer = _format_peer(websocket)
        LOGGER.info("Client connected: %s", peer)

        try:
            async for raw_message in websocket:
                await self._handle_raw_message(websocket, raw_message)
        except websockets.exceptions.ConnectionClosed:
            LOGGER.info("Client disconnected: %s", peer)
        except Exception:
            LOGGER.exception("Unexpected connection error (%s).", peer)

    async def _handle_raw_message(self, websocket: Any, raw_message: Any) -> None:
        """Parse one raw frame and dispatch to a typed message handler.

        Args:
            websocket: Client WebSocket connection.
            raw_message: Raw frame payload.
        """

        request_id = None

        try:
            message = _parse_json_message(raw_message)
            request_id = str(message.get("request_id") or uuid.uuid4())

            if not self._is_authorized(message):
                await self._send_auth_failed(websocket, request_id)
                return

            message_type = str(message.get("type") or "").strip().upper()
            if not message_type:
                await self._send_error(websocket, request_id, "BAD_REQUEST", "Message type is required.")
                return

            if message_type == "PING":
                await self._handle_ping(websocket, request_id)
                return

            if message_type == "DEPLOY_MODEL":
                await self._handle_deploy_model(websocket, request_id, message)
                return

            if message_type == "RUN_INFERENCE":
                await self._handle_run_inference(websocket, request_id, message)
                return

            if message_type == "GET_RESULTS":
                await self._handle_get_results(websocket, request_id, message)
                return

            if message_type == "SHUTDOWN":
                await self._handle_shutdown(websocket, request_id)
                return

            await self._send_error(
                websocket,
                request_id,
                "UNSUPPORTED_MESSAGE",
                f"Unsupported message type: {message_type}",
            )
        except ValueError as exc:
            await self._send_error(websocket, request_id, "BAD_REQUEST", str(exc))
        except Exception as exc:
            LOGGER.exception("Unhandled request error.")
            await self._send_error(
                websocket,
                request_id,
                "INTERNAL_ERROR",
                str(exc),
                details={"traceback": traceback.format_exc(limit=5)},
            )

    async def _handle_ping(self, websocket: Any, request_id: str) -> None:
        """Handle PING request and return runtime/device diagnostics.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
        """

        device_info = _collect_device_info(self._config)

        await self._send_json(
            websocket,
            {
                "type": "PONG",
                "request_id": request_id,
                "status": "ok",
                "device_info": device_info,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _handle_deploy_model(self, websocket: Any, request_id: str, message: dict[str, Any]) -> None:
        """Handle DEPLOY_MODEL request by writing decoded model bytes to cache.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            message: Parsed request payload.
        """

        model_name = str(message.get("model_name") or "deployed_model.pt").strip()
        model_name = _ensure_model_filename(model_name)

        model_bytes_b64 = str(message.get("model_bytes") or "").strip()
        if not model_bytes_b64:
            raise ValueError("DEPLOY_MODEL requires 'model_bytes'.")

        try:
            model_bytes = base64.b64decode(model_bytes_b64, validate=False)
        except Exception as exc:
            raise ValueError("DEPLOY_MODEL 'model_bytes' is not valid base64 data.") from exc

        if not model_bytes:
            raise ValueError("Decoded model payload is empty.")

        # Model names are sanitized to avoid path traversal in cache output.
        destination = self._config.model_cache_dir / Path(model_name).name
        destination.write_bytes(model_bytes)

        LOGGER.info("Model deployed: %s (%d bytes)", destination.name, len(model_bytes))

        await self._send_json(
            websocket,
            {
                "type": "ACK",
                "request_id": request_id,
                "action": "DEPLOY_MODEL",
                "status": "Model deployed.",
                "model_name": destination.name,
                "model_path": str(destination.resolve()),
                "size_bytes": len(model_bytes),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _handle_run_inference(self, websocket: Any, request_id: str, message: dict[str, Any]) -> None:
        """Handle RUN_INFERENCE by evaluating a deployed model on dataset input.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            message: Parsed request payload.
        """

        raw_model_name = str(message.get("model_name") or "").strip()
        if not raw_model_name:
            raise ValueError("RUN_INFERENCE requires 'model_name'.")
        model_name = _ensure_model_filename(raw_model_name)

        dataset_path = str(message.get("dataset_path") or "").strip()
        if not dataset_path:
            raise ValueError("RUN_INFERENCE requires 'dataset_path'.")

        conf = _to_float(message.get("conf"), default=0.25)
        iou = _to_float(message.get("iou"), default=0.45)

        model_path = self._config.model_cache_dir / model_name
        if not model_path.exists():
            raise ValueError(f"Model is not deployed on this agent: {model_name}")

        dataset = Path(dataset_path).expanduser().resolve()
        if not dataset.exists():
            raise ValueError(f"Dataset path does not exist on device: {dataset}")

        result_id = uuid.uuid4().hex
        run_output_dir = self._config.output_dir / result_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        await self._send_progress(websocket, request_id, 3, "Loading model...", result_id=result_id)

        artifact = await self._run_inference_job(
            websocket=websocket,
            request_id=request_id,
            result_id=result_id,
            model_path=model_path,
            model_name=model_name,
            dataset_path=dataset,
            conf=conf,
            iou=iou,
            output_dir=run_output_dir,
        )

        self._results[result_id] = artifact

        await self._send_progress(websocket, request_id, 100, "Inference complete.", result_id=result_id)

        await self._send_json(
            websocket,
            {
                "type": "INFERENCE_COMPLETE",
                "request_id": request_id,
                "result_id": artifact.result_id,
                "status": "Inference completed.",
                "model_name": artifact.model_name,
                "dataset_path": artifact.dataset_path,
                "num_images_tested": artifact.num_images_tested,
                "metrics": artifact.metrics,
                "output_images_dir": str(artifact.output_dir.resolve()),
                "timestamp": artifact.created_at.isoformat(),
            },
        )

    async def _run_inference_job(
        self,
        websocket: Any,
        request_id: str,
        result_id: str,
        model_path: Path,
        model_name: str,
        dataset_path: Path,
        conf: float,
        iou: float,
        output_dir: Path,
    ) -> InferenceArtifact:
        """Execute YOLO validation/prediction workflow.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            result_id: Inference result identifier.
            model_path: Cached local model path.
            model_name: Deployed model name.
            dataset_path: Validation dataset path.
            conf: Confidence threshold.
            iou: IoU threshold.
            output_dir: Directory for output artifacts.

        Returns:
            InferenceArtifact: Stored inference outputs/metadata.

        Raises:
            RuntimeError: If ultralytics is unavailable.
        """

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - dependency import guard
            raise RuntimeError("Ultralytics is not installed on this edge device.") from exc

        model = YOLO(str(model_path))

        image_paths: list[Path] = []
        metrics: dict[str, float | None] = {
            "map50": None,
            "map50_95": None,
            "precision": None,
            "recall": None,
            "speed_ms": None,
        }

        num_images_tested = 0

        is_yaml_dataset = dataset_path.is_file() and dataset_path.suffix.lower() in {".yaml", ".yml"}

        if is_yaml_dataset:
            await self._send_progress(websocket, request_id, 12, "Running validation...", result_id=result_id)

            val_result = await asyncio.to_thread(
                model.val,
                data=str(dataset_path),
                conf=conf,
                iou=iou,
                project=str(output_dir),
                name="val",
                exist_ok=True,
                verbose=False,
            )

            metrics = _extract_val_metrics(val_result)

            # Validation metrics are authoritative for model quality.
            num_images_tested = _count_images_from_dataset_yaml(dataset_path)

            await self._send_progress(websocket, request_id, 70, "Validation complete. Building previews...", result_id=result_id)

            preview_source = _resolve_preview_source_from_yaml(dataset_path)
            if preview_source is not None and preview_source.exists():
                image_paths, predict_speed = await self._run_predict_preview(
                    websocket=websocket,
                    request_id=request_id,
                    result_id=result_id,
                    model=model,
                    source=preview_source,
                    conf=conf,
                    iou=iou,
                    output_dir=output_dir,
                    max_images=48,
                    start_progress=72,
                    end_progress=95,
                    output_name="preview",
                )
                if metrics.get("speed_ms") is None:
                    metrics["speed_ms"] = predict_speed
        else:
            await self._send_progress(websocket, request_id, 12, "Running prediction...", result_id=result_id)

            image_paths, predict_speed = await self._run_predict_preview(
                websocket=websocket,
                request_id=request_id,
                result_id=result_id,
                model=model,
                source=dataset_path,
                conf=conf,
                iou=iou,
                output_dir=output_dir,
                max_images=0,
                start_progress=18,
                end_progress=95,
                output_name="predict",
            )

            num_images_tested = len(_collect_image_files(dataset_path))
            if num_images_tested == 0:
                num_images_tested = len(image_paths)

            metrics["speed_ms"] = predict_speed

        if not image_paths:
            image_paths = _collect_image_files(output_dir)

        return InferenceArtifact(
            result_id=result_id,
            model_name=model_name,
            dataset_path=str(dataset_path),
            metrics=metrics,
            num_images_tested=num_images_tested,
            output_dir=output_dir,
            image_paths=image_paths,
            created_at=datetime.now(timezone.utc),
        )

    async def _run_predict_preview(
        self,
        websocket: Any,
        request_id: str,
        result_id: str,
        model: Any,
        source: Path,
        conf: float,
        iou: float,
        output_dir: Path,
        max_images: int,
        start_progress: int,
        end_progress: int,
        output_name: str,
    ) -> tuple[list[Path], float | None]:
        """Run YOLO prediction and stream incremental progress updates.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            result_id: Inference result identifier.
            model: Ultralytics YOLO model instance.
            source: Source image directory/file.
            conf: Confidence threshold.
            iou: IoU threshold.
            output_dir: Directory for output artifacts.
            max_images: Max image count for prediction (0 means no limit).
            start_progress: Lower progress bound.
            end_progress: Upper progress bound.
            output_name: Subfolder name under output_dir.

        Returns:
            tuple[list[Path], float | None]: Output image paths and average inference speed.
        """

        source_images = _collect_image_files(source)
        if max_images > 0:
            source_images = source_images[:max_images]

        if source_images:
            predict_source: list[str] | str = [str(path) for path in source_images]
            total = len(source_images)
        else:
            # If no direct images were discovered, let ultralytics resolve source itself.
            predict_source = str(source)
            total = 0

        results_iter = model.predict(
            source=predict_source,
            conf=conf,
            iou=iou,
            save=True,
            project=str(output_dir),
            name=output_name,
            exist_ok=True,
            stream=True,
            verbose=False,
        )

        processed = 0
        speed_values: list[float] = []

        for result in results_iter:
            processed += 1

            # Progress is bounded between caller-provided stage values.
            if total > 0:
                ratio = processed / total
                progress = start_progress + int((end_progress - start_progress) * ratio)
                await self._send_progress(
                    websocket,
                    request_id,
                    progress,
                    f"Processing images {processed}/{total}...",
                    result_id=result_id,
                )

            speed_map = getattr(result, "speed", None)
            if isinstance(speed_map, dict):
                inference_speed = _to_float(speed_map.get("inference"), default=None)
                if inference_speed is not None:
                    speed_values.append(inference_speed)

            await asyncio.sleep(0)

        if total == 0:
            await self._send_progress(
                websocket,
                request_id,
                end_progress,
                "Prediction pass complete.",
                result_id=result_id,
            )

        image_paths = _collect_image_files(output_dir / output_name)
        avg_speed = sum(speed_values) / len(speed_values) if speed_values else None

        return image_paths, avg_speed

    async def _handle_get_results(self, websocket: Any, request_id: str, message: dict[str, Any]) -> None:
        """Handle GET_RESULTS request by streaming chunked base64 images.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            message: Parsed request payload.
        """

        result_id = str(message.get("result_id") or "").strip()
        if not result_id:
            raise ValueError("GET_RESULTS requires 'result_id'.")

        artifact = self._results.get(result_id)
        if artifact is None:
            await self._send_error(
                websocket,
                request_id,
                "RESULT_NOT_FOUND",
                f"No stored result found for result_id={result_id}",
            )
            return

        images = [path for path in artifact.image_paths if path.exists()]
        total_images = len(images)

        if total_images == 0:
            await self._send_json(
                websocket,
                {
                    "type": "RESULTS_COMPLETE",
                    "request_id": request_id,
                    "result_id": result_id,
                    "status": "No result images available.",
                    "num_images": 0,
                    "output_images_dir": str(artifact.output_dir.resolve()),
                    "metrics": artifact.metrics,
                },
            )
            return

        for image_index, image_path in enumerate(images, start=1):
            data = image_path.read_bytes()
            b64_data = base64.b64encode(data).decode("utf-8")
            chunks = [
                b64_data[i : i + self._config.chunk_size]
                for i in range(0, len(b64_data), self._config.chunk_size)
            ]
            if not chunks:
                chunks = [""]

            for chunk_index, chunk in enumerate(chunks, start=1):
                progress = int(((image_index - 1) / total_images) * 100)
                await self._send_json(
                    websocket,
                    {
                        "type": "RESULT_CHUNK",
                        "request_id": request_id,
                        "result_id": result_id,
                        "image_name": image_path.name,
                        "image_index": image_index,
                        "num_images": total_images,
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks),
                        "chunk": chunk,
                        "is_last": chunk_index == len(chunks),
                        "progress": progress,
                        "status": f"Transferring {image_path.name} ({chunk_index}/{len(chunks)})",
                    },
                )

                await asyncio.sleep(0)

        await self._send_json(
            websocket,
            {
                "type": "RESULTS_COMPLETE",
                "request_id": request_id,
                "result_id": result_id,
                "status": "Result transfer complete.",
                "num_images": total_images,
                "output_images_dir": str(artifact.output_dir.resolve()),
                "metrics": artifact.metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _handle_shutdown(self, websocket: Any, request_id: str) -> None:
        """Handle SHUTDOWN request and stop the server loop.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
        """

        await self._send_json(
            websocket,
            {
                "type": "ACK",
                "request_id": request_id,
                "action": "SHUTDOWN",
                "status": "Shutdown signal accepted.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        LOGGER.info("Shutdown requested by remote client.")
        self._shutdown_event.set()

    def _is_authorized(self, message: dict[str, Any]) -> bool:
        """Return whether request payload carries a valid auth token.

        Args:
            message: Parsed request payload.

        Returns:
            bool: True when request auth token matches configured token.
        """

        token = str(message.get("auth_token") or "").strip()
        return bool(token) and token == self._config.auth_token

    async def _send_progress(
        self,
        websocket: Any,
        request_id: str,
        progress: int,
        status: str,
        *,
        result_id: str,
    ) -> None:
        """Send progress event for long-running inference operations.

        Args:
            websocket: Client WebSocket connection.
            request_id: Request correlation ID.
            progress: Progress percentage.
            status: Human-readable status message.
            result_id: Inference result identifier.
        """

        await self._send_json(
            websocket,
            {
                "type": "INFERENCE_PROGRESS",
                "request_id": request_id,
                "result_id": result_id,
                "progress": max(0, min(100, int(progress))),
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _send_auth_failed(self, websocket: Any, request_id: str | None) -> None:
        """Send standardized auth-failure response.

        Args:
            websocket: Client WebSocket connection.
            request_id: Optional request correlation ID.
        """

        await self._send_json(
            websocket,
            {
                "type": "AUTH_FAILED",
                "request_id": request_id,
                "error": "Invalid or missing auth token.",
                "message": "Invalid or missing auth token.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _send_error(
        self,
        websocket: Any,
        request_id: str | None,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send standardized error response payload.

        Args:
            websocket: Client WebSocket connection.
            request_id: Optional request correlation ID.
            code: Stable machine-readable error code.
            message: Human-readable error summary.
            details: Optional details map.
        """

        payload: dict[str, Any] = {
            "type": "ERROR",
            "request_id": request_id,
            "code": code,
            "error": message,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if details:
            payload["details"] = details

        LOGGER.warning("Agent error [%s]: %s", code, message)
        await self._send_json(websocket, payload)

    async def _send_json(self, websocket: Any, payload: dict[str, Any]) -> None:
        """Serialize and send a JSON payload on active WebSocket.

        Args:
            websocket: Client WebSocket connection.
            payload: Outgoing message payload.
        """

        text = json.dumps(payload, ensure_ascii=True, default=str)
        LOGGER.debug("SEND: %s", text)
        await websocket.send(text)

    def _print_startup_banner(self) -> None:
        """Print human-readable startup diagnostics to stdout/log."""

        info = _collect_device_info(self._config)

        lines = [
            "=" * 66,
            " YOLO Studio Edge Agent",
            "=" * 66,
            f" Listen Address : ws://{self._config.host}:{self._config.port}",
            f" Hostname       : {info.get('hostname')}",
            f" Platform       : {info.get('platform')}",
            f" Python         : {info.get('python_version')}",
            f" CUDA Available : {info.get('cuda_available')}",
            f" Disk Free      : {info.get('disk_free_human')}",
            f" RAM Total      : {info.get('ram_total_human')}",
            f" Model Cache    : {self._config.model_cache_dir.resolve()}",
            f" Output Dir     : {self._config.output_dir.resolve()}",
            f" Log File       : {self._config.log_file.resolve()}",
            "=" * 66,
        ]

        banner = "\n".join(lines)
        print(banner)
        LOGGER.info("\n%s", banner)


def _resolve_path(base_dir: Path, candidate: str) -> Path:
    """Resolve path relative to config directory when not absolute.

    Args:
        base_dir: Base directory for relative paths.
        candidate: Candidate path text.

    Returns:
        Path: Resolved path.
    """

    path = Path(str(candidate)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _parse_json_message(raw_message: Any) -> dict[str, Any]:
    """Parse one WebSocket frame payload into dict.

    Args:
        raw_message: Raw frame payload.

    Returns:
        dict[str, Any]: Parsed JSON dictionary.

    Raises:
        ValueError: If payload is invalid or not an object.
    """

    if isinstance(raw_message, (bytes, bytearray)):
        text = raw_message.decode("utf-8", errors="replace")
    else:
        text = str(raw_message)

    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError("Invalid JSON payload.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    LOGGER.debug("RECV: %s", payload)
    return payload


def _ensure_model_filename(name: str) -> str:
    """Normalize untrusted model names to a safe `.pt` file name.

    Args:
        name: Requested model name.

    Returns:
        str: Sanitized model file name.
    """

    raw = Path(name or "deployed_model.pt").name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw).strip("._")
    if not safe:
        safe = "deployed_model"
    if not safe.endswith(".pt"):
        safe = f"{safe}.pt"
    return safe


def _collect_device_info(config: AgentConfig) -> dict[str, Any]:
    """Collect edge-device diagnostics for heartbeat and startup banners.

    Args:
        config: Agent configuration.

    Returns:
        dict[str, Any]: Device diagnostics payload.
    """

    disk = shutil.disk_usage(config.output_dir)
    ram_total = _get_total_ram_bytes()

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": _is_cuda_available(),
        "ip_address": _get_primary_ip(),
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes": int(disk.free),
        "disk_total_human": _format_bytes(int(disk.total)),
        "disk_free_human": _format_bytes(int(disk.free)),
        "ram_total_bytes": ram_total,
        "ram_total_human": _format_bytes(ram_total) if ram_total is not None else "unknown",
    }


def _is_cuda_available() -> bool:
    """Return whether CUDA is currently available on device.

    Returns:
        bool: True when torch CUDA backend is available.
    """

    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_total_ram_bytes() -> int | None:
    """Best-effort total RAM size detection.

    Returns:
        int | None: Total RAM bytes if available.
    """

    # Linux-first path for Jetson/Raspberry Pi targets.
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    # Portable fallback using POSIX sysconf where available.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except Exception:
        return None


def _format_peer(websocket: Any) -> str:
    """Return best-effort remote peer address string.

    Args:
        websocket: Client WebSocket connection.

    Returns:
        str: Formatted peer name.
    """

    address = getattr(websocket, "remote_address", None)
    if not address:
        return "unknown"
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


def _format_bytes(value: int | None) -> str:
    """Format bytes into concise human-readable units.

    Args:
        value: Byte count.

    Returns:
        str: Human-readable text.
    """

    if value is None:
        return "unknown"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    scaled = float(value)
    for unit in units:
        if scaled < 1024.0:
            return f"{scaled:.2f} {unit}"
        scaled /= 1024.0
    return f"{scaled:.2f} EB"


def _to_float(value: Any, default: float | None) -> float | None:
    """Convert arbitrary scalar-like value to float.

    Args:
        value: Candidate value.
        default: Fallback value.

    Returns:
        float | None: Converted float when possible, else default.
    """

    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def _collect_image_files(path: Path) -> list[Path]:
    """Collect image files from a file or directory path.

    Args:
        path: File or directory path.

    Returns:
        list[Path]: Sorted image file list.
    """

    if not path.exists():
        return []

    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        return [path]

    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        resolved = _resolve_preview_source_from_yaml(path)
        if resolved is not None:
            return _collect_image_files(resolved)
        return []

    if not path.is_dir():
        return []

    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _resolve_preview_source_from_yaml(data_yaml_path: Path) -> Path | None:
    """Resolve best preview image source from YOLO dataset YAML.

    Args:
        data_yaml_path: Dataset YAML path.

    Returns:
        Path | None: Resolved preview source path.
    """

    try:
        payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    base_dir = data_yaml_path.resolve().parent
    for key in ("test", "val", "train"):
        value = payload.get(key)

        if isinstance(value, list) and value:
            candidate = value[0]
        else:
            candidate = value

        if not isinstance(candidate, str) or not candidate.strip():
            continue

        source = Path(candidate.strip())
        if not source.is_absolute():
            source = (base_dir / source).resolve()

        if source.exists():
            return source

    return None


def _count_images_from_dataset_yaml(data_yaml_path: Path) -> int:
    """Estimate number of images referenced by dataset YAML.

    Args:
        data_yaml_path: Dataset YAML path.

    Returns:
        int: Count of images found under test/val/train references.
    """

    total = 0

    try:
        payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return total

    if not isinstance(payload, dict):
        return total

    base_dir = data_yaml_path.resolve().parent

    for key in ("test", "val", "train"):
        value = payload.get(key)

        candidates: list[str] = []
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [item for item in value if isinstance(item, str)]

        for candidate in candidates:
            source = Path(candidate)
            if not source.is_absolute():
                source = (base_dir / source).resolve()
            total += len(_collect_image_files(source))

    return total


def _extract_val_metrics(val_result: Any) -> dict[str, float | None]:
    """Extract standard metric values from Ultralytics validation output.

    Args:
        val_result: Return value from `model.val(...)`.

    Returns:
        dict[str, float | None]: Normalized metric payload.
    """

    metrics: dict[str, float | None] = {
        "map50": None,
        "map50_95": None,
        "precision": None,
        "recall": None,
        "speed_ms": None,
    }

    if val_result is None:
        return metrics

    if isinstance(val_result, dict):
        metrics["map50"] = _pick_metric(val_result, ("map50", "mAP50", "metrics/mAP50(B)"))
        metrics["map50_95"] = _pick_metric(val_result, ("map50_95", "mAP50-95", "metrics/mAP50-95(B)"))
        metrics["precision"] = _pick_metric(val_result, ("precision", "metrics/precision(B)", "mp"))
        metrics["recall"] = _pick_metric(val_result, ("recall", "metrics/recall(B)", "mr"))
        metrics["speed_ms"] = _pick_metric(val_result, ("speed", "inference_ms", "speed_ms"))
        return metrics

    box = getattr(val_result, "box", None)
    if box is not None:
        metrics["map50"] = _to_float(getattr(box, "map50", None), default=None)
        metrics["map50_95"] = _to_float(getattr(box, "map", None), default=None)
        metrics["precision"] = _to_float(getattr(box, "mp", None), default=None)
        metrics["recall"] = _to_float(getattr(box, "mr", None), default=None)

    speed = getattr(val_result, "speed", None)
    if isinstance(speed, dict):
        metrics["speed_ms"] = _pick_metric(speed, ("inference", "inference_ms", "speed"))

    results_dict = getattr(val_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics["map50"] = metrics["map50"] or _pick_metric(results_dict, ("metrics/mAP50(B)", "map50"))
        metrics["map50_95"] = metrics["map50_95"] or _pick_metric(
            results_dict,
            ("metrics/mAP50-95(B)", "map50_95"),
        )
        metrics["precision"] = metrics["precision"] or _pick_metric(
            results_dict,
            ("metrics/precision(B)", "precision"),
        )
        metrics["recall"] = metrics["recall"] or _pick_metric(
            results_dict,
            ("metrics/recall(B)", "recall"),
        )

    return metrics


def _pick_metric(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Pick first numeric value for preferred metric keys.

    Args:
        source: Metric source mapping.
        keys: Candidate keys in priority order.

    Returns:
        float | None: Parsed metric value.
    """

    for key in keys:
        value = source.get(key)
        numeric = _to_float(value, default=None)
        if numeric is not None:
            return numeric

    lowered = {str(key).lower(): value for key, value in source.items()}
    for key in keys:
        target = key.lower()
        for existing_key, value in lowered.items():
            if target in existing_key:
                numeric = _to_float(value, default=None)
                if numeric is not None:
                    return numeric

    return None


def _get_primary_ip() -> str:
    """Return best-effort primary device IP address.

    Returns:
        str: Detected IP address or loopback fallback.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def _configure_logging(log_file: Path, verbose: bool) -> None:
    """Configure file and console logging handlers.

    Args:
        log_file: Path to log output file.
        verbose: Whether to enable DEBUG-level logging.
    """

    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="YOLO Studio edge WebSocket agent")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to agent_config.yaml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser.parse_args()


def main() -> int:
    """Run edge agent process entrypoint.

    Returns:
        int: Process exit code.
    """

    args = _parse_args()

    try:
        config = AgentConfig.from_yaml(Path(args.config).expanduser().resolve())
    except Exception as exc:
        print(f"Failed to load agent config: {exc}")
        return 1

    _configure_logging(config.log_file, verbose=bool(args.verbose))

    LOGGER.info("Starting YOLO Studio edge agent...")

    agent = EdgeAgent(config=config)

    try:
        asyncio.run(agent.serve())
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Edge agent interrupted by keyboard signal.")
        return 0
    except Exception:
        LOGGER.exception("Edge agent failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

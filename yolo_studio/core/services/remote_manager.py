"""WebSocket client manager for remote edge-device operations.

This module provides a production-oriented client used by YOLO Studio to
communicate with Jetson/Raspberry Pi agents over WebSockets. It supports
heartbeat checks, model deployment, inference requests, and result retrieval.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "remote_results"


class RemoteManagerError(RuntimeError):
    """Base exception raised for remote-manager request failures."""


@dataclass(slots=True)
class RemoteDeviceConfig:
    """Normalized remote-device connection payload."""

    host: str
    port: int
    auth_token: str
    name: str = "device"
    device_id: int | None = None
    device_type: str | None = None

    @property
    def ws_url(self) -> str:
        """Return full WebSocket URL for this device.

        Returns:
            str: WebSocket URL.
        """

        return f"ws://{self.host}:{self.port}"

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
    ) -> "RemoteDeviceConfig":
        """Build device config from dictionary-like payload.

        Args:
            mapping: Device dictionary.

        Returns:
            RemoteDeviceConfig: Normalized config.

        Raises:
            ValueError: If required fields are missing.
        """

        host = str(mapping.get("host") or "").strip()
        port = int(mapping.get("port") or 0)
        auth_token = str(mapping.get("auth_token") or "").strip()

        if not host:
            raise ValueError("Device host is required.")
        if port <= 0:
            raise ValueError("Device port is required.")
        if not auth_token:
            raise ValueError("Device auth_token is required.")

        name = str(mapping.get("name") or "device").strip() or "device"
        device_id = _to_int(mapping.get("id"))
        device_type = str(mapping.get("device_type") or "").strip() or None

        return cls(
            host=host,
            port=port,
            auth_token=auth_token,
            name=name,
            device_id=device_id,
            device_type=device_type,
        )


class RemoteManager:
    """Client manager for remote-agent WebSocket interactions."""

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        output_root: str | Path | None = None,
    ) -> None:
        """Initialize remote manager settings.

        Args:
            timeout_seconds: Default request timeout in seconds.
            output_root: Root directory where retrieved images are stored.
        """

        self._timeout_seconds = float(timeout_seconds)
        self._output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        self._output_root.mkdir(parents=True, exist_ok=True)

    async def ping_device(
        self,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        device: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send PING to a remote device.

        Args:
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            device: Optional full device mapping.
            timeout: Optional per-call timeout.

        Returns:
            dict[str, Any]: Ping response payload with `ok` field.
        """

        try:
            cfg = self._resolve_device(
                device=device,
                host=host,
                port=port,
                auth_token=auth_token,
            )
            message = self._build_message(message_type="PING", auth_token=cfg.auth_token)
            response = await self._request_once(cfg=cfg, message=message, timeout=timeout)

            msg_type = _message_type(response)
            ok = msg_type == "PONG"

            return {
                "ok": ok,
                "type": msg_type,
                "device": {
                    "id": cfg.device_id,
                    "name": cfg.name,
                    "host": cfg.host,
                    "port": cfg.port,
                    "device_type": cfg.device_type,
                },
                "info": response.get("device_info")
                or response.get("info")
                or response.get("payload")
                or response,
                "raw": response,
            }
        except Exception as exc:
            LOGGER.debug("ping_device failed", exc_info=True)
            return {
                "ok": False,
                "type": "ERROR",
                "error": str(exc),
                "device": {
                    "host": host,
                    "port": port,
                },
            }

    async def deploy_model(
        self,
        device: dict[str, Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        weights_path: str | Path | None = None,
        model_path: str | Path | None = None,
        model_name: str | None = None,
        timeout: float | None = None,
        progress_callback: Callable[[int], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        """Deploy model bytes to remote agent.

        Args:
            device: Optional device mapping.
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            weights_path: Local weight file path.
            model_path: Alias for `weights_path`.
            model_name: Optional model identifier used on the agent.
            timeout: Optional per-call timeout.
            progress_callback: Optional callback receiving integer progress.
            status_callback: Optional callback receiving status text.

        Returns:
            dict[str, Any]: Agent ACK payload.

        Raises:
            RemoteManagerError: If deployment fails.
        """

        cfg = self._resolve_device(
            device=device,
            host=host,
            port=port,
            auth_token=auth_token,
        )

        path = Path(model_path or weights_path or "").resolve()
        if not path.exists():
            raise RemoteManagerError(f"Model path does not exist: {path}")

        await _emit_status(status_callback, f"Reading model file: {path.name}")
        await _emit_progress(progress_callback, 5)

        model_bytes = path.read_bytes()
        model_b64 = base64.b64encode(model_bytes).decode("utf-8")
        resolved_name = model_name or path.name

        message = self._build_message(
            message_type="DEPLOY_MODEL",
            auth_token=cfg.auth_token,
            model_name=resolved_name,
            model_bytes=model_b64,
            model_size=len(model_bytes),
        )

        await _emit_status(status_callback, "Sending model to remote device...")
        await _emit_progress(progress_callback, 20)

        response = await self._request_once(cfg=cfg, message=message, timeout=timeout)
        response_type = _message_type(response)

        if response_type not in {"ACK", "DEPLOY_ACK", "DEPLOY_COMPLETE", "SUCCESS", "OK"}:
            raise RemoteManagerError(f"Unexpected deploy response type: {response_type}")

        await _emit_progress(progress_callback, 40)
        await _emit_status(status_callback, "Model deployment acknowledged by remote device.")

        return response

    async def run_inference(
        self,
        device: dict[str, Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        model_name: str | None = None,
        dataset_path: str | Path | None = None,
        test_dataset_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        timeout: float | None = None,
        progress_callback: Callable[[int], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        """Run remote inference request and stream progress events.

        Args:
            device: Optional device mapping.
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            model_name: Model name already cached on agent.
            dataset_path: Dataset path.
            test_dataset_path: Alias for `dataset_path`.
            conf: Confidence threshold.
            iou: IoU threshold.
            timeout: Optional per-call timeout.
            progress_callback: Optional callback receiving integer progress.
            status_callback: Optional callback receiving status text.

        Returns:
            dict[str, Any]: Final inference completion payload.

        Raises:
            RemoteManagerError: If inference fails or no terminal event is returned.
        """

        cfg = self._resolve_device(
            device=device,
            host=host,
            port=port,
            auth_token=auth_token,
        )

        resolved_dataset = Path(test_dataset_path or dataset_path or "").resolve()
        if not resolved_dataset.exists():
            raise RemoteManagerError(f"Dataset path does not exist: {resolved_dataset}")

        message = self._build_message(
            message_type="RUN_INFERENCE",
            auth_token=cfg.auth_token,
            model_name=model_name,
            dataset_path=str(resolved_dataset),
            conf=float(conf),
            iou=float(iou),
        )

        await _emit_status(status_callback, "Running remote inference...")
        await _emit_progress(progress_callback, 45)

        terminal = await self._stream_request(
            cfg=cfg,
            message=message,
            timeout=timeout,
            progress_callback=progress_callback,
            status_callback=status_callback,
            terminal_types={"INFERENCE_COMPLETE", "COMPLETE", "DONE", "SUCCESS"},
        )

        if terminal is None:
            raise RemoteManagerError("Remote inference did not return a completion event.")

        await _emit_progress(progress_callback, 80)
        await _emit_status(status_callback, "Remote inference completed.")
        return terminal

    async def get_results(
        self,
        result_id: str,
        device: dict[str, Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        output_dir: str | Path | None = None,
        timeout: float | None = None,
        progress_callback: Callable[[int], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        """Request annotated result images for an inference run.

        Args:
            result_id: Inference result identifier returned by the agent.
            device: Optional device mapping.
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            output_dir: Optional output directory for decoded images.
            timeout: Optional per-call timeout.
            progress_callback: Optional callback receiving integer progress.
            status_callback: Optional callback receiving status text.

        Returns:
            dict[str, Any]: Result payload containing `output_images`.

        Raises:
            RemoteManagerError: If retrieval fails.
        """

        cfg = self._resolve_device(
            device=device,
            host=host,
            port=port,
            auth_token=auth_token,
        )

        resolved_output_dir = Path(output_dir) if output_dir is not None else self._build_output_dir(cfg, "results")
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        message = self._build_message(
            message_type="GET_RESULTS",
            auth_token=cfg.auth_token,
            result_id=result_id,
        )

        await _emit_status(status_callback, f"Requesting results for result_id={result_id}...")
        await _emit_progress(progress_callback, 82)

        terminal, output_images = await self._stream_result_images(
            cfg=cfg,
            message=message,
            output_dir=resolved_output_dir,
            timeout=timeout,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

        await _emit_progress(progress_callback, 95)
        await _emit_status(status_callback, "Result image retrieval complete.")

        return {
            "terminal": terminal,
            "output_images": output_images,
            "output_images_dir": str(resolved_output_dir.resolve()),
        }

    async def deploy_and_test(
        self,
        device: dict[str, Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        weights_path: str | Path | None = None,
        model_path: str | Path | None = None,
        dataset_path: str | Path | None = None,
        test_dataset_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        timeout: float | None = None,
        progress_callback: Callable[[int], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Deploy a model and execute a remote test run.

        Args:
            device: Optional device mapping.
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            weights_path: Local model file path.
            model_path: Alias for `weights_path`.
            dataset_path: Dataset path.
            test_dataset_path: Alias for `dataset_path`.
            conf: Confidence threshold.
            iou: IoU threshold.
            timeout: Optional per-call timeout.
            progress_callback: Optional callback receiving integer progress.
            status_callback: Optional callback receiving status text.
            **_: Additional kwargs ignored for compatibility.

        Returns:
            dict[str, Any]: Normalized deploy/test result.

        Raises:
            RemoteManagerError: If deployment or inference fails.
        """

        cfg = self._resolve_device(
            device=device,
            host=host,
            port=port,
            auth_token=auth_token,
        )

        path = Path(model_path or weights_path or "").resolve()
        if not path.exists():
            raise RemoteManagerError(f"Weights path does not exist: {path}")

        run_output_dir = self._build_output_dir(cfg, "run")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        await _emit_status(status_callback, f"Deploy/test started for device '{cfg.name}'.")
        await _emit_progress(progress_callback, 2)

        deploy_response = await self.deploy_model(
            device={
                "id": cfg.device_id,
                "name": cfg.name,
                "host": cfg.host,
                "port": cfg.port,
                "auth_token": cfg.auth_token,
                "device_type": cfg.device_type,
            },
            weights_path=path,
            model_name=path.name,
            timeout=timeout,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

        inference_response = await self.run_inference(
            device={
                "id": cfg.device_id,
                "name": cfg.name,
                "host": cfg.host,
                "port": cfg.port,
                "auth_token": cfg.auth_token,
                "device_type": cfg.device_type,
            },
            model_name=path.name,
            dataset_path=test_dataset_path or dataset_path,
            conf=conf,
            iou=iou,
            timeout=timeout,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

        metrics = _extract_metrics(inference_response)

        output_images: list[str] = []
        output_images_dir: str | None = None

        result_id = str(
            inference_response.get("result_id")
            or inference_response.get("id")
            or inference_response.get("inference_id")
            or ""
        ).strip()

        if result_id:
            results_payload = await self.get_results(
                result_id=result_id,
                device={
                    "id": cfg.device_id,
                    "name": cfg.name,
                    "host": cfg.host,
                    "port": cfg.port,
                    "auth_token": cfg.auth_token,
                    "device_type": cfg.device_type,
                },
                output_dir=run_output_dir / "images",
                timeout=timeout,
                progress_callback=progress_callback,
                status_callback=status_callback,
            )
            output_images = list(results_payload.get("output_images") or [])
            output_images_dir = str(results_payload.get("output_images_dir") or "") or None
        else:
            inline_images = _extract_inline_image_payloads(inference_response)
            if inline_images:
                output_dir = run_output_dir / "images"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_images = _decode_inline_images(inline_images, output_dir)
                output_images_dir = str(output_dir.resolve())

        await _emit_progress(progress_callback, 100)
        await _emit_status(status_callback, "Deploy/test completed.")

        return {
            "device": {
                "id": cfg.device_id,
                "name": cfg.name,
                "host": cfg.host,
                "port": cfg.port,
                "device_type": cfg.device_type,
            },
            "model_name": path.name,
            "result_id": result_id or None,
            "metrics": metrics,
            "output_images": output_images,
            "output_images_dir": output_images_dir,
            "num_images_tested": _to_int(
                inference_response.get("num_images_tested")
                or inference_response.get("images_tested")
                or len(output_images)
            )
            or 0,
            "deploy_response": deploy_response,
            "inference_response": inference_response,
            "notes": "Remote deploy/test completed via WebSocket manager.",
        }

    async def run_remote_test(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for deploy-and-test behavior.

        Args:
            **kwargs: Same kwargs accepted by `deploy_and_test`.

        Returns:
            dict[str, Any]: Deploy/test result payload.
        """

        return await self.deploy_and_test(**kwargs)

    async def shutdown_device(
        self,
        device: dict[str, Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        auth_token: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send SHUTDOWN command to remote agent.

        Args:
            device: Optional device mapping.
            host: Device host.
            port: Device port.
            auth_token: Device auth token.
            timeout: Optional per-call timeout.

        Returns:
            dict[str, Any]: Agent response payload.
        """

        cfg = self._resolve_device(
            device=device,
            host=host,
            port=port,
            auth_token=auth_token,
        )

        message = self._build_message(message_type="SHUTDOWN", auth_token=cfg.auth_token)
        return await self._request_once(cfg=cfg, message=message, timeout=timeout)

    async def _request_once(
        self,
        cfg: RemoteDeviceConfig,
        message: dict[str, Any],
        timeout: float | None,
    ) -> dict[str, Any]:
        """Send one request and await a single response frame.

        Args:
            cfg: Device connection config.
            message: Outbound JSON message.
            timeout: Per-call timeout.

        Returns:
            dict[str, Any]: Parsed JSON response.

        Raises:
            RemoteManagerError: If transport or response parsing fails.
        """

        websocket_module = _import_websockets()
        resolved_timeout = self._resolve_timeout(timeout)

        try:
            async with websocket_module.connect(
                cfg.ws_url,
                open_timeout=resolved_timeout,
                close_timeout=resolved_timeout,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                await ws.send(json.dumps(message))
                raw = await asyncio.wait_for(ws.recv(), timeout=resolved_timeout)
        except Exception as exc:
            raise RemoteManagerError(f"WebSocket request failed: {exc}") from exc

        response = _parse_message(raw)
        self._raise_for_error(response)
        return response

    async def _stream_request(
        self,
        cfg: RemoteDeviceConfig,
        message: dict[str, Any],
        timeout: float | None,
        progress_callback: Callable[[int], Any] | None,
        status_callback: Callable[[str], Any] | None,
        terminal_types: set[str],
    ) -> dict[str, Any] | None:
        """Send request and stream response frames until terminal event.

        Args:
            cfg: Device config.
            message: Outbound message.
            timeout: Timeout seconds.
            progress_callback: Optional progress callback.
            status_callback: Optional status callback.
            terminal_types: Set of terminal message types.

        Returns:
            dict[str, Any] | None: Terminal response payload.

        Raises:
            RemoteManagerError: If transport fails.
        """

        websocket_module = _import_websockets()
        resolved_timeout = self._resolve_timeout(timeout)

        try:
            async with websocket_module.connect(
                cfg.ws_url,
                open_timeout=resolved_timeout,
                close_timeout=resolved_timeout,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                await ws.send(json.dumps(message))

                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=resolved_timeout)
                    payload = _parse_message(raw)
                    self._raise_for_error(payload)

                    msg_type = _message_type(payload)

                    progress_value = _extract_progress(payload)
                    if progress_value is not None:
                        await _emit_progress(progress_callback, progress_value)

                    status_text = _extract_status(payload)
                    if status_text:
                        await _emit_status(status_callback, status_text)

                    if msg_type in terminal_types:
                        return payload
        except Exception as exc:
            raise RemoteManagerError(f"WebSocket stream request failed: {exc}") from exc

    async def _stream_result_images(
        self,
        cfg: RemoteDeviceConfig,
        message: dict[str, Any],
        output_dir: Path,
        timeout: float | None,
        progress_callback: Callable[[int], Any] | None,
        status_callback: Callable[[str], Any] | None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Request result images and decode chunked responses.

        Args:
            cfg: Device config.
            message: Outbound GET_RESULTS message.
            output_dir: Directory where decoded files are written.
            timeout: Timeout seconds.
            progress_callback: Optional progress callback.
            status_callback: Optional status callback.

        Returns:
            tuple[dict[str, Any] | None, list[str]]: Terminal payload and output image paths.

        Raises:
            RemoteManagerError: If transport fails.
        """

        websocket_module = _import_websockets()
        resolved_timeout = self._resolve_timeout(timeout)

        chunk_buffers: dict[str, list[str]] = {}
        output_paths: list[str] = []
        terminal_payload: dict[str, Any] | None = None

        terminal_types = {
            "RESULTS_COMPLETE",
            "GET_RESULTS_COMPLETE",
            "INFERENCE_RESULTS_COMPLETE",
            "COMPLETE",
            "DONE",
            "SUCCESS",
        }

        chunk_types = {
            "RESULT_CHUNK",
            "IMAGE_CHUNK",
            "RESULT_IMAGE_CHUNK",
            "CHUNK",
        }

        image_types = {
            "RESULT_IMAGE",
            "IMAGE",
            "RESULT",
        }

        try:
            async with websocket_module.connect(
                cfg.ws_url,
                open_timeout=resolved_timeout,
                close_timeout=resolved_timeout,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                await ws.send(json.dumps(message))

                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=resolved_timeout)
                    payload = _parse_message(raw)
                    self._raise_for_error(payload)

                    msg_type = _message_type(payload)

                    progress_value = _extract_progress(payload)
                    if progress_value is not None:
                        await _emit_progress(progress_callback, progress_value)

                    status_text = _extract_status(payload)
                    if status_text:
                        await _emit_status(status_callback, status_text)

                    if msg_type in chunk_types:
                        key = str(
                            payload.get("image_name")
                            or payload.get("image_id")
                            or payload.get("result_id")
                            or "result"
                        )
                        chunk = _extract_base64_field(payload)
                        if chunk:
                            chunk_buffers.setdefault(key, []).append(chunk)

                        if bool(payload.get("is_last")):
                            written = _flush_chunk_buffer(
                                key=key,
                                buffers=chunk_buffers,
                                output_dir=output_dir,
                            )
                            if written is not None:
                                output_paths.append(str(written))
                        continue

                    if msg_type in image_types:
                        b64_data = _extract_base64_field(payload)
                        if b64_data:
                            image_name = str(payload.get("image_name") or payload.get("name") or "result")
                            written = _write_b64_image(
                                b64_data=b64_data,
                                output_dir=output_dir,
                                base_name=image_name,
                            )
                            if written is not None:
                                output_paths.append(str(written))
                        continue

                    if msg_type in terminal_types:
                        terminal_payload = payload
                        inline_images = _extract_inline_image_payloads(payload)
                        if inline_images:
                            output_paths.extend(_decode_inline_images(inline_images, output_dir))
                        break
        except Exception as exc:
            raise RemoteManagerError(f"WebSocket result stream failed: {exc}") from exc

        # Flush any incomplete chunk buffers on stream termination.
        for key in list(chunk_buffers.keys()):
            written = _flush_chunk_buffer(key=key, buffers=chunk_buffers, output_dir=output_dir)
            if written is not None:
                output_paths.append(str(written))

        deduped = _dedupe_preserve_order(output_paths)
        return terminal_payload, deduped

    def _resolve_device(
        self,
        device: dict[str, Any] | None,
        host: str | None,
        port: int | None,
        auth_token: str | None,
    ) -> RemoteDeviceConfig:
        """Resolve device config from explicit args and optional mapping.

        Args:
            device: Optional device mapping.
            host: Explicit host override.
            port: Explicit port override.
            auth_token: Explicit auth token override.

        Returns:
            RemoteDeviceConfig: Normalized device config.

        Raises:
            ValueError: If required device fields are missing.
        """

        payload = dict(device or {})

        if host is not None:
            payload["host"] = host
        if port is not None:
            payload["port"] = port
        if auth_token is not None:
            payload["auth_token"] = auth_token

        return RemoteDeviceConfig.from_mapping(payload)

    def _build_message(self, message_type: str, auth_token: str, **payload: Any) -> dict[str, Any]:
        """Build outbound message envelope.

        Args:
            message_type: Message type keyword.
            auth_token: Device auth token.
            **payload: Additional payload fields.

        Returns:
            dict[str, Any]: Outbound message body.
        """

        body = {
            "request_id": str(uuid.uuid4()),
            "type": str(message_type).upper(),
            "auth_token": auth_token,
        }
        body.update(payload)
        return body

    def _resolve_timeout(self, timeout: float | None) -> float:
        """Resolve effective timeout value.

        Args:
            timeout: Optional override timeout.

        Returns:
            float: Effective timeout.
        """

        value = self._timeout_seconds if timeout is None else float(timeout)
        return max(1.0, value)

    def _build_output_dir(self, cfg: RemoteDeviceConfig, suffix: str) -> Path:
        """Build timestamped output directory for a device run.

        Args:
            cfg: Device config.
            suffix: Folder suffix.

        Returns:
            Path: Output directory path.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _slugify(cfg.name or f"device_{cfg.host}_{cfg.port}")
        return self._output_root / f"{safe_name}_{timestamp}_{suffix}"

    @staticmethod
    def _raise_for_error(payload: dict[str, Any]) -> None:
        """Raise manager error for error-type responses.

        Args:
            payload: Parsed response payload.

        Raises:
            RemoteManagerError: If payload indicates error.
        """

        msg_type = _message_type(payload)

        if msg_type in {"ERROR", "AUTH_FAILED", "FAILED", "EXCEPTION"}:
            message = (
                payload.get("message")
                or payload.get("error")
                or payload.get("detail")
                or f"Remote agent returned {msg_type}"
            )
            raise RemoteManagerError(str(message))

        error_value = payload.get("error")
        if error_value:
            raise RemoteManagerError(str(error_value))


async def _emit_progress(callback: Callable[[int], Any] | None, value: int) -> None:
    """Invoke progress callback safely.

    Args:
        callback: Progress callback.
        value: Progress value.
    """

    if callback is None:
        return

    try:
        result = callback(int(max(0, min(value, 100))))
        if inspect.isawaitable(result):
            await result
    except Exception:
        LOGGER.debug("Progress callback failed.", exc_info=True)


async def _emit_status(callback: Callable[[str], Any] | None, text: str) -> None:
    """Invoke status callback safely.

    Args:
        callback: Status callback.
        text: Status text.
    """

    if callback is None:
        return

    try:
        result = callback(str(text))
        if inspect.isawaitable(result):
            await result
    except Exception:
        LOGGER.debug("Status callback failed.", exc_info=True)


def _import_websockets() -> Any:
    """Import websockets module with a clear error message.

    Returns:
        Any: Imported websockets module.

    Raises:
        RemoteManagerError: If dependency is missing.
    """

    try:
        import websockets  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RemoteManagerError("websockets package is not installed.") from exc

    return websockets


def _parse_message(raw: Any) -> dict[str, Any]:
    """Parse WebSocket frame payload as JSON object.

    Args:
        raw: Raw frame payload.

    Returns:
        dict[str, Any]: Parsed JSON object.

    Raises:
        RemoteManagerError: If payload is not valid JSON object.
    """

    text: str
    if isinstance(raw, (bytes, bytearray)):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)

    try:
        payload = json.loads(text)
    except Exception as exc:
        raise RemoteManagerError(f"Invalid JSON response from remote agent: {text[:300]}") from exc

    if not isinstance(payload, dict):
        raise RemoteManagerError("Remote agent response must be a JSON object.")

    return payload


def _message_type(payload: dict[str, Any]) -> str:
    """Return normalized uppercase message type.

    Args:
        payload: Message payload.

    Returns:
        str: Uppercase message type.
    """

    for key in ("type", "message_type", "event", "kind"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _extract_progress(payload: dict[str, Any]) -> int | None:
    """Extract progress integer from payload.

    Args:
        payload: Message payload.

    Returns:
        int | None: Progress value if present.
    """

    for key in ("progress", "percent", "percentage"):
        value = payload.get(key)
        numeric = _to_int(value)
        if numeric is not None:
            return max(0, min(numeric, 100))

    return None


def _extract_status(payload: dict[str, Any]) -> str | None:
    """Extract status text from payload.

    Args:
        payload: Message payload.

    Returns:
        str | None: Status text.
    """

    for key in ("status", "message", "log", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_base64_field(payload: dict[str, Any]) -> str | None:
    """Extract likely base64 chunk/image field from payload.

    Args:
        payload: Message payload.

    Returns:
        str | None: Base64 text if found.
    """

    for key in (
        "chunk",
        "data",
        "chunk_data",
        "chunk_b64",
        "image",
        "image_bytes",
        "image_b64",
        "bytes",
        "payload",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if text.startswith("data:"):
                text = text.split(",", 1)[-1]
            return text

    return None


def _extract_inline_image_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract inline image payload records from response.

    Args:
        payload: Response payload.

    Returns:
        list[dict[str, Any]]: Image payload entries.
    """

    records: list[dict[str, Any]] = []

    images = payload.get("images") or payload.get("output_images")
    if isinstance(images, list):
        for index, item in enumerate(images, start=1):
            if isinstance(item, dict):
                b64_data = _extract_base64_field(item)
                if b64_data:
                    records.append(
                        {
                            "name": str(item.get("name") or item.get("image_name") or f"result_{index:03d}"),
                            "b64": b64_data,
                        }
                    )
            elif isinstance(item, str) and item.strip():
                maybe_b64 = item.strip()
                if maybe_b64.startswith("data:"):
                    maybe_b64 = maybe_b64.split(",", 1)[-1]
                records.append({"name": f"result_{index:03d}", "b64": maybe_b64})

    single = _extract_base64_field(payload)
    if single and not records:
        records.append({"name": "result_001", "b64": single})

    return records


def _decode_inline_images(records: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Decode inline base64 image records into files.

    Args:
        records: Image payload records.
        output_dir: Output directory.

    Returns:
        list[str]: Written image paths.
    """

    output_paths: list[str] = []
    for record in records:
        b64_data = str(record.get("b64") or "").strip()
        if not b64_data:
            continue

        base_name = str(record.get("name") or "result").strip() or "result"
        written = _write_b64_image(b64_data=b64_data, output_dir=output_dir, base_name=base_name)
        if written is not None:
            output_paths.append(str(written))

    return _dedupe_preserve_order(output_paths)


def _flush_chunk_buffer(key: str, buffers: dict[str, list[str]], output_dir: Path) -> Path | None:
    """Flush chunk buffer for one image key into a decoded image file.

    Args:
        key: Image buffer key.
        buffers: Chunk buffer dict.
        output_dir: Output directory.

    Returns:
        Path | None: Written image path.
    """

    chunks = buffers.pop(key, [])
    if not chunks:
        return None

    combined = "".join(chunks)
    return _write_b64_image(b64_data=combined, output_dir=output_dir, base_name=key)


def _write_b64_image(b64_data: str, output_dir: Path, base_name: str) -> Path | None:
    """Decode base64 image bytes and write to disk.

    Args:
        b64_data: Base64 encoded image bytes.
        output_dir: Output directory.
        base_name: Base file name prefix.

    Returns:
        Path | None: Path to written image.
    """

    try:
        raw = base64.b64decode(b64_data, validate=False)
    except Exception:
        LOGGER.debug("Failed decoding base64 image payload.", exc_info=True)
        return None

    if not raw:
        return None

    extension = _guess_image_extension(raw)
    safe_name = _slugify(Path(base_name).stem)
    file_path = _make_unique_path(output_dir / f"{safe_name}{extension}")

    try:
        file_path.write_bytes(raw)
    except Exception:
        LOGGER.debug("Failed writing decoded image to disk.", exc_info=True)
        return None

    return file_path.resolve()


def _make_unique_path(path: Path) -> Path:
    """Return unique path by adding numeric suffix when needed.

    Args:
        path: Desired file path.

    Returns:
        Path: Unique path.
    """

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for index in range(2, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate

    # Fallback with UUID for extreme collision cases.
    return parent / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def _guess_image_extension(raw: bytes) -> str:
    """Guess image extension from file signature.

    Args:
        raw: Decoded image bytes.

    Returns:
        str: File extension beginning with dot.
    """

    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return ".gif"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
        return ".webp"
    if raw.startswith(b"II*\x00") or raw.startswith(b"MM\x00*"):
        return ".tif"
    return ".bin"


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract normalized metric fields from inference completion payload.

    Args:
        payload: Inference completion payload.

    Returns:
        dict[str, float | None]: Normalized metrics.
    """

    metrics_source = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload

    return {
        "map50": _pick_metric(metrics_source, ("map50", "mAP50", "metrics/mAP50(B)")),
        "map50_95": _pick_metric(metrics_source, ("map50_95", "mAP50-95", "metrics/mAP50-95(B)")),
        "precision": _pick_metric(metrics_source, ("precision", "metrics/precision(B)", "mp")),
        "recall": _pick_metric(metrics_source, ("recall", "metrics/recall(B)", "mr")),
        "speed_ms": _pick_metric(metrics_source, ("speed_ms", "inference_ms", "speed")),
    }


def _pick_metric(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Pick first numeric metric value from preferred key candidates.

    Args:
        source: Metric mapping.
        keys: Preferred keys.

    Returns:
        float | None: Metric value if found.
    """

    for key in keys:
        value = source.get(key)
        numeric = _to_float(value)
        if numeric is not None:
            return numeric

    lower_source = {str(key).lower(): value for key, value in source.items()}
    for key in keys:
        key_lower = key.lower()
        for existing_key, value in lower_source.items():
            if key_lower in existing_key:
                numeric = _to_float(value)
                if numeric is not None:
                    return numeric

    return None


def _to_int(value: Any) -> int | None:
    """Convert scalar-like value to int.

    Args:
        value: Candidate value.

    Returns:
        int | None: Converted integer.
    """

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _to_float(value: Any) -> float | None:
    """Convert scalar-like value to float.

    Args:
        value: Candidate value.

    Returns:
        float | None: Converted float.
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


def _slugify(text: str) -> str:
    """Create filesystem-safe slug from arbitrary text.

    Args:
        text: Input text.

    Returns:
        str: Slug text.
    """

    raw = str(text or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or "asset"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate list preserving first-seen order.

    Args:
        values: Value list.

    Returns:
        list[str]: Deduplicated list.
    """

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "RemoteManager",
    "RemoteManagerError",
    "RemoteDeviceConfig",
]

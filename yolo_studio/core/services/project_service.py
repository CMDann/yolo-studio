"""Project persistence utilities for YOLO Studio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProjectConfig:
    id: int | None
    name: str
    description: str | None
    created_at: str
    updated_at: str
    root_dir: str
    git_remote: str | None
    git_branch: str | None
    version: int = 1


def write_project_yaml(path: Path, payload: ProjectConfig) -> None:
    data = {
        "version": payload.version,
        "id": payload.id,
        "name": payload.name,
        "description": payload.description,
        "created_at": payload.created_at,
        "updated_at": payload.updated_at,
        "root_dir": payload.root_dir,
        "git_remote": payload.git_remote,
        "git_branch": payload.git_branch,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_project_yaml(path: Path) -> ProjectConfig | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    return ProjectConfig(
        id=payload.get("id"),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip() or None,
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        root_dir=str(payload.get("root_dir") or "").strip(),
        git_remote=str(payload.get("git_remote") or "").strip() or None,
        git_branch=str(payload.get("git_branch") or "").strip() or None,
        version=int(payload.get("version") or 1),
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_for_datasets(root_dir: Path) -> list[Path]:
    candidates = []
    for candidate in root_dir.rglob("*.yaml"):
        if candidate.name not in {"data.yaml", "dataset.yaml"}:
            continue
        candidates.append(candidate)
    return candidates


def scan_for_weights(root_dir: Path) -> list[Path]:
    weights = []
    for ext in ("*.pt", "*.onnx", "*.engine", "*.tflite"):
        weights.extend(root_dir.rglob(ext))
    return weights


__all__ = [
    "ProjectConfig",
    "write_project_yaml",
    "load_project_yaml",
    "scan_for_datasets",
    "scan_for_weights",
    "now_iso",
]

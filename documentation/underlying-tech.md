# Underlying Tech

This document summarizes the technical stack and architecture used by YOLO Studio.

## Core Stack

Declared in `yolo_studio/requirements.txt`:

- `PyQt6`: desktop UI framework
- `pyqtgraph`: live metric chart rendering
- `SQLAlchemy`: ORM + persistence layer
- `ultralytics`: YOLO training/inference engine
- `opencv-python`: annotation/image preview rendering support
- `Pillow`: image IO/render helpers (notably remote preview generation)
- `websockets`: remote device protocol transport
- `roboflow`: Roboflow catalog + dataset download integration
- `huggingface_hub`: Hugging Face search/download/snapshot import
- `PyYAML`: YAML parsing/generation (`data.yaml`, configs)
- `python-socketio`: included dependency (not currently wired in core paths)

## Application Layers

1. UI Layer (`yolo_studio/ui`)
   - Tabs, widgets, theme, main window
2. Worker/Service Layer (`yolo_studio/core/workers`, `yolo_studio/core/services`)
   - Training orchestration, remote manager, dataset operations
3. Persistence Layer (`yolo_studio/core/models/database.py`)
   - SQLAlchemy models + session utilities
4. Edge Runtime (`yolo_studio/edge/jetson_agent.py`)
   - WebSocket server for deploy/inference/result streaming

## Data and Persistence

Default DB:

- SQLite file: `yolo_studio/yolo_studio.db`
- URL default: `sqlite:///.../yolo_studio.db`
- `check_same_thread=False` enabled for QThread worker compatibility

Key tables:

- `Dataset`
- `TrainingRun`
- `RemoteDevice`
- `RemoteTestResult`
- `BaseWeight`

ORM model source:

- `yolo_studio/core/models/database.py`

Compatibility wrappers exist in:

- `yolo_studio/core/database.py`
- `yolo_studio/core/trainer.py`
- `yolo_studio/core/remote_manager.py`

## Concurrency Model

UI thread:

- All Qt widget creation and updates.

Background workers:

- `QThread` classes per workflow (training, search, download, deploy/test).

Async networking:

- `RemoteManager` uses `asyncio` + `websockets`.
- Remote tab can invoke async calls and bridge callbacks into Qt progress/status UI.

Design goal:

- Keep long-running compute/network tasks off the main thread.

## Training Runtime

Engine:

- Ultralytics YOLO via `YOLOTrainer`.

Signals:

- `progress`, `status`, `metrics`, `log`, `finished`, `error`

Persistence:

- Creates/updates `TrainingRun` rows during lifecycle.

Metric extraction strategy:

- Ultralytics callback fields first
- CSV fallback from `results.csv` when needed

## Remote Runtime

Desktop client:

- `RemoteManager` in `core/services/remote_manager.py`
- Protocol transport: JSON-over-WebSocket
- Handles deploy, inference, streamed progress, chunked image retrieval

Edge server:

- `edge/jetson_agent.py`
- Handles:
  - auth validation
  - model deployment to cache
  - inference execution
  - chunked result transfer
  - shutdown commands

Config file:

- `yolo_studio/edge/agent_config.yaml`

## External Integrations

Roboflow:

- Search + download workers in `ui/tabs/discover_tab.py`

Hugging Face:

- Search/model download/dataset import workers in `ui/tabs/discover_tab.py`
- Base weight metadata saved to `BaseWeight`
- Dataset imports saved to `Dataset`

## Filesystem Conventions

Common output locations:

- Generated train YAMLs: `yolo_studio/datasets/generated/`
- Built datasets: `yolo_studio/datasets/library/`
- Roboflow downloads: `yolo_studio/datasets/roboflow_downloads/`
- Hugging Face model cache: `yolo_studio/models/huggingface/`
- Hugging Face dataset cache: `yolo_studio/datasets/huggingface/`
- Training runs: `yolo_studio/runs/train/`
- Saved model copies: `yolo_studio/saved_models/`
- Remote fallback runs: `yolo_studio/runs/remote_fallback/`
- Remote result outputs: `yolo_studio/runs/remote_results/`

## UI Theme and Logging

Theme system:

- `yolo_studio/ui/styles/theme.py`
- Global QSS + palette + tokenized colors

In-app logging:

- `LogPanel` + `QtLogHandler` in `yolo_studio/ui/widgets/log_panel.py`
- Root logger bridge allows worker logs to appear in the docked log UI.

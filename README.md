![gif](./ref/screen.gif)


# YOLO Studio

YOLO Studio is a desktop developer tool for training, organizing, and remotely deploying YOLO computer vision models.

## Project Overview

This project provides:
- Local dataset management and dataset-building workflows
- Configurable YOLO training with live metrics and logs
- Discovery integrations for Roboflow Universe, Hugging Face Hub, Kaggle, and Open Images
- Remote deployment and testing against Jetson/Raspberry Pi edge devices
- Real-time camera inference with session capture
- Offline evaluation with analytics dashboards
- Cross-run analytics and experiment comparison
- Project system for organizing assets under named roots

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
python main.py
```

## Architecture

```text
+-------------------------- YOLO Studio (PyQt6) ---------------------------+
|                                                                          |
|  +---------------- UI Layer ----------------+    +------ Widgets -------+ |
|  | train | dataset | discover | remote      |    | log | chart | cards | |
|  | camera | evaluate | analytics            |    +----------------------+ |
|  +------------------------------------------+                           |
|                 |                                                    |
|                 v                                                    |
|  +------------------ Core Layers ----------------------------+         |
|  | models | services | workers | projects                   |         |
|  +----------------------------------------------------------+         |
|                 |                                                    |
|                 v                                                    |
|  +------------------------- Integrations ---------------------------+  |
|  | ultralytics | roboflow | huggingface_hub | websocket devices    |  |
|  +------------------------------------------------------------------+  |
+--------------------------------------------------------------------------+

Edge Device Agent (Jetson/RPi): edge/jetson_agent.py
```

## Code Layout

- `core/models/`: SQLAlchemy entities and DB session/bootstrap logic
- `core/services/`: dataset and remote-device service classes
- `core/workers/`: QThread background worker classes
- `ui/windows/`: top-level window classes
- `ui/styles/`: theme and styling modules
- `ui/tabs/`, `ui/widgets/`: feature tabs and reusable UI components
- Compatibility shims remain at legacy module paths (`core/database.py`, etc.)
  for safer import migration.

## Usage Guide

### Training Tab
- Configure model architecture, dataset, and hyperparameters
- Start/stop training jobs with live metrics and logs
- Save successful runs and manage trained weights

### Dataset Tab
- Browse datasets stored in SQLite
- Build datasets from local images and generate YOLO metadata
- Review and export saved models
Note: Project scope filters the dataset library and saved models.

### Discover Tab
- Search and download datasets from Roboflow Universe
- Search and download models/datasets from Hugging Face Hub
- Search Kaggle datasets/competitions and download datasets
- Download Open Images subsets with class filters
Note: Imported assets can be attached to the active project.

### Remote Devices Tab
- Register edge devices
- Deploy model weights remotely
- Run remote inference tests and store result metrics

### Camera Tab
- Run real-time webcam inference with overlays
- Record frames, save snapshots, export detections CSV
- Session metadata stored in the database

### Evaluate Tab
- Offline evaluation for datasets or image folders
- Confusion matrix, PR curves, and confidence curves
- Export reports as PDF + JSON

### Analytics Tab
- Cross-run history, loss overlays, and per-class heatmaps
- Compare runs and export dashboard ZIP

### Project System
- Create/open/duplicate/delete projects
- Store project config in `project.yaml`
- Project selector filters all tabs

## Edge Agent Setup

1. Copy `edge/jetson_agent.py` and `edge/agent_config.yaml` to the device.
2. Install Python dependencies on the device.
3. Configure `agent_config.yaml`.
4. Start the agent and verify connectivity from YOLO Studio.

## Troubleshooting

- Ensure Python version is 3.10+.
- Confirm `config.json` contains valid API tokens when using external services.
- Verify CUDA/device drivers for GPU-enabled edge inference.
- Check application and edge agent logs for runtime errors.
- On Linux desktops without working Wayland or XCB dependencies (e.g. missing `libxcb-cursor0`),
  YOLO Studio falls back to Qt's `minimal` platform. You can override with `QT_QPA_PLATFORM`
  if you want to force a specific backend.

## Current Status

The application scaffold and core feature modules are implemented, including:
- SQLAlchemy-backed metadata models and persistence helpers
- Training, dataset, discover, and remote-device GUI tabs
- QThread workers for long-running operations
- WebSocket remote manager and edge agent protocol implementation

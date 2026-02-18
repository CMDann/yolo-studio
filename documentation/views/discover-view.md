# Discover View

The Discover tab integrates external model/dataset catalogs and registers imported assets into local storage + SQLite.

Main class: `DiscoverTab` in `yolo_studio/ui/tabs/discover_tab.py`

## Section Layout

The tab uses a segmented top control with two stacked sections:

- Roboflow Universe
- Hugging Face Hub

Credentials are persisted in:

- `yolo_studio/config.json`

## Roboflow Section

Workers:

- `RoboflowSearchWorker`
- `RoboflowDownloadWorker`

Capabilities:

- Search projects (paged)
- Download dataset versions in YOLO format
- Infer class names when metadata is incomplete
- Register downloaded dataset in local `Dataset` table

Local download root:

- `yolo_studio/datasets/roboflow_downloads/`

## Hugging Face Section

Workers:

- `HuggingFaceSearchWorker`
- `HuggingFaceModelDownloadWorker`
- `HuggingFaceDatasetImportWorker`

Capabilities:

- Search models (filtered toward YOLO-related results)
- Download model weight assets (`.pt`, `.onnx`, `.engine`, `.tflite`)
- Register weights into `BaseWeight` table
- Import dataset repos and map snapshots to YOLO-compatible structure
- Register imported datasets in `Dataset` table

Local storage:

- Model cache: `yolo_studio/models/huggingface/`
- Dataset cache: `yolo_studio/datasets/huggingface/`

## Dataset Import Mapping Strategy

For Hugging Face dataset snapshots:

1. If `data.yaml` exists, use that YOLO root directly.
2. Otherwise, build `yolo_mapped/` with copied images and label placeholders.
3. Infer classes from labels, fallback to `["object"]` if none.
4. Emit generated `data.yaml` and register as dataset.

## Async/UI Model

All network-heavy operations run in `QThread` workers and update the UI with:

- `progress` signals
- `status` signals
- `finished` / `error` signals

This prevents UI blocking while integrating external services.

## Related Files

- `yolo_studio/ui/tabs/discover_tab.py`
- `yolo_studio/core/models/database.py`
- `yolo_studio/requirements.txt` (Roboflow/HF dependencies)

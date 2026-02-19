# Discover View

The Discover tab integrates external model/dataset catalogs and registers imported assets into local storage + SQLite.

Main class: `DiscoverTab` in `yolo_studio/ui/tabs/discover_tab.py`

## Section Layout

The tab uses a segmented top control with four stacked sections:

- Roboflow Universe
- Hugging Face Hub
- Kaggle
- Open Images

Credentials are persisted in:

- `yolo_studio/config.json`

Credentials are managed from File > Settings (Roboflow, Hugging Face, Kaggle).

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

- `<project_root>/datasets/roboflow_downloads/`

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

- Model cache: `<project_root>/models/huggingface/`
- Dataset cache: `<project_root>/datasets/huggingface/`

## Kaggle Section

Workers:

- `KaggleSearchWorker`
- `KaggleDownloadWorker`

Capabilities:

- Search datasets with filters (type, sort, tag)
- List active computer-vision competitions with “View on Kaggle”
- Download datasets and auto-register YOLO layouts
- Optional CSV-to-YOLO conversion prompt when no YOLO structure is detected

Local download root:

- `<project_root>/datasets/kaggle/`

## Open Images Section

Workers:

- `OpenImagesDownloadWorker`

Capabilities:

- Select classes with autocomplete and multi-select list
- Choose splits (train/validation/test) and max images per class
- Download subsets via FiftyOne and export to YOLO format
- Auto-generate `data.yaml` and register dataset

Local download root:

- `<project_root>/datasets/open_images/`

Dependency note:

- Open Images downloads require the optional `fiftyone` package.

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
- `yolo_studio/requirements.txt` (Roboflow/HF/Kaggle dependencies)

## Project Scope

- Imported datasets and base weights are tagged with the active project.

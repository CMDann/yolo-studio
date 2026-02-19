# Datasets View

The Datasets tab handles dataset library operations, dataset building, and saved model management.

Main class: `DatasetTab` in `yolo_studio/ui/tabs/dataset_tab.py`

## Library Panel

Capabilities:

- Create dataset metadata rows manually
- Import existing dataset folders
- Edit/delete dataset rows
- Export selected dataset as ZIP
- Open dataset folder in system file explorer
- View structured metadata details (classes, tags, path, counts, description)

Persistence:

- Reads/writes `Dataset` rows in SQLite (`core.models.database`)

## Dataset Builder Sub-View

Purpose:

- Build YOLO-style datasets from selected images and label files.

Key features:

- Drag-drop or picker for image files
- Class list add/remove/rename
- Split sliders (train/val/test)
- Annotation preview with bounding-box overlay
- `data.yaml` generation preview
- Save built dataset into managed library path

Disk output:

- Root: `<project_root>/datasets/library/<slug_timestamp>/`
- Creates:
  - `images/train|val|test`
  - `labels/train|val|test`
  - `data.yaml`

Preview tech:

- Uses OpenCV (`cv2`) when available to draw boxes and labels.
- Falls back gracefully if optional image dependencies are missing.

## Saved Models Sub-View

Shows saved `TrainingRun` rows where `is_saved = true`.

Columns:

- Run name, architecture, dataset, mAP metrics, saved date, notes

Actions:

- Refresh
- Export weights to chosen path
- Push to device (currently queues UI flow and prompts device; full transfer is handled via Remote tab/manager path)
- Remove from saved list (optional file deletion)

## Data Sources

- Dataset rows: `Dataset`
- Saved model rows: `TrainingRun`
- Device list for "Push to Device": `RemoteDevice`

All read/write activity uses SQLAlchemy sessions from `get_session()`.

## Project Scope

- Dataset and saved model lists are filtered to the active project.
- New datasets are tagged with `project_id` if a project is selected.

## Related Files

- `yolo_studio/ui/tabs/dataset_tab.py`
- `yolo_studio/core/models/database.py`
- `yolo_studio/core/services/dataset_manager.py`

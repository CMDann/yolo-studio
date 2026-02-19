# Camera View

The Camera tab runs real-time webcam inference with annotations and session capture.

Main class: `CameraTab` in `yolo_studio/ui/tabs/camera_tab.py`

## Key Features

- Enumerates local camera devices via OpenCV.
- Selects a saved model (`TrainingRun.is_saved`).
- Configures confidence + IoU thresholds.
- Starts/stops a streaming worker (`CameraStreamWorker`).
- Draws boxes/labels on frames with class-consistent colors.

## Session Controls

- Record toggle saves annotated frames to:
  - `<project_root>/output/camera_sessions/<session_id>/frames/`
- Snapshot button saves a single frame immediately.
- Save CSV exports detections to:
  - `<project_root>/output/camera_sessions/<session_id>/detections.csv`

## Persistence

- `CameraSession` table tracks session metadata.
- All records are tagged with `project_id` when a project is active.

## Related Files

- `yolo_studio/ui/tabs/camera_tab.py`
- `yolo_studio/core/models/database.py`

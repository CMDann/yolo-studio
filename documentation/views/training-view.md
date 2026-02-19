# Training View

The Training tab configures and executes Ultralytics training runs in a background thread.

## UI Structure

Main class: `TrainTab` in `yolo_studio/ui/tabs/train_tab.py`

Left side controls:

- Model architecture selector (`yolov8*`, `yolov11*`)
- Dataset selector (from SQLite)
- Train/val/test folder drop zones
- `data.yaml` generate/select/clear controls
- Hyperparameters (epochs, batch size, image size, LR, optimizer, warmup, weight decay)
- Augmentation toggles (mosaic/mixup/copy-paste/hsv/flip)
- Pretrained/custom-weights controls
- Run metadata (name, notes)
- Start/Stop buttons

Right side monitoring:

- Live metrics chart (`MetricChart`)
- Epoch progress bar and labels
- Best mAP badge
- Training log stream
- "Save This Run" button

## Dataset YAML Logic

`TrainTab` can auto-generate `data.yaml` files under:

- `<project_root>/datasets/generated/`

Class-name resolution priority during generation:

1. Selected dataset record class names from DB.
2. Nearby YAML files (`data.yaml`, `dataset.yaml`, `.yml`) around selected split paths.
3. Inferred class IDs from YOLO label `.txt` files in split folders.

Before training starts, it validates that the selected YAML resolves to a non-zero class count.

## Training Execution Path

Worker class: `YOLOTrainer` in `yolo_studio/core/workers/trainer.py`

Flow:

1. Validate config (`TrainingConfig.from_mapping`).
2. Create `TrainingRun` row (`PENDING` -> `RUNNING`).
3. Load Ultralytics model (`.pt` or `.yaml` depending on settings).
4. Run `model.train(...)` with UI-selected args.
5. Emit status/progress/metrics over Qt signals.
6. Persist success/failure fields back to DB.

## Live Monitoring Tech

Metrics pipeline:

- Ultralytics callbacks (`on_train_epoch_end`, `on_fit_epoch_end`)
- Metric extraction from trainer fields
- CSV fallback parsing from Ultralytics `results.csv`
- Qt signal to `TrainTab._on_metrics(...)`
- Chart update via `MetricChart.append_epoch_metrics(...)`

Chart widget:

- File: `yolo_studio/ui/widgets/metric_chart.py`
- Backend: `pyqtgraph`
- Series rendered:
  - `train/box_loss`
  - `train/cls_loss`
  - `val/box_loss`
  - `metrics/mAP50(B)`
  - `metrics/mAP50-95(B)`

## Saving Runs

"Save This Run" behavior:

- Copies best weights into `<project_root>/saved_models/*.pt`
- Marks `TrainingRun.is_saved = True`
- Updates `TrainingRun.weights_path` to copied file
- Optionally stores run notes

Saved runs then appear in:

- Datasets view (`Saved Models` sub-tab)
- Remote Devices model selector

## Project Scope

- Dataset selector and saved model outputs are scoped to the active project.
- New `TrainingRun` rows store `project_id`.

## Related Files

- `yolo_studio/ui/tabs/train_tab.py`
- `yolo_studio/core/workers/trainer.py`
- `yolo_studio/ui/widgets/file_drop_zone.py`
- `yolo_studio/ui/widgets/metric_chart.py`
- `yolo_studio/ui/widgets/log_panel.py`

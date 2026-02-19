# Evaluate View

The Evaluate tab runs offline evaluation against datasets or image folders and renders analytics.

Main class: `EvaluateTab` in `yolo_studio/ui/tabs/evaluate_tab.py`

## Modes

- Dataset evaluation uses `model.val(...)` with a YOLO `data.yaml`.
- Folder evaluation uses `model.predict(...)` per image and computes metrics when label files exist.

## Outputs

- Confusion matrices (raw + normalized)
- PR curves and confidence curves
- Class distribution and speed histogram
- Summary metrics cards (mAP50, mAP50-95, Precision, Recall, F1)

## Export

- PDF report (Matplotlib backend)
- Raw JSON metrics
- Files written to `<project_root>/output/evaluations/`

## Persistence

- Stores summary in `RemoteTestResult` with `source_type` and `source_path`.
- Writes evaluation snapshot `run_<id>.json` for analytics heatmaps.

## Project Scope

- Model and dataset selectors are filtered to the active project.

## Related Files

- `yolo_studio/ui/tabs/evaluate_tab.py`
- `yolo_studio/core/models/database.py`

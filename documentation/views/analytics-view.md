# Analytics View

The Analytics tab provides cross-run historical analysis and experiment comparison.

Main class: `AnalyticsTab` in `yolo_studio/ui/tabs/analytics_tab.py`

## Panels

- Training History: mAP50 and mAP50-95 over time, selectable points
- Run Comparison: multi-select comparison table with CSV export
- Loss Overlay: up to 5 runs plotted from `results.csv`
- Class Performance Heatmap: per-class mAP across models
- Dataset Statistics: image count, class distribution, annotation density
- Model Leaderboard: saved models ranked by mAP50

## Exports

- Dashboard ZIP containing PNGs and CSVs

## Data Sources

- `TrainingRun` (metrics + config YAML)
- Dataset metadata
- Evaluation snapshots (`<project_root>/output/evaluations/run_<id>.json`)

## Project Scope

- All views are filtered to the active project.

## Related Files

- `yolo_studio/ui/tabs/analytics_tab.py`
- `yolo_studio/core/models/database.py`

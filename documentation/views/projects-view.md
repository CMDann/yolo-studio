# Projects View

Projects allow grouping datasets, models, runs, and sessions under named roots.

## Project Selector

- Located in the main window toolbar.
- Options:
  - Select Project...
  - All Projects
  - Individual projects

## Project Wizard

Accessible via File > New Project:

- Name, description, root directory
- Optional import scan for datasets and weight files
- Writes `project.yaml` into the project root

## Project Manager

Accessible via File > Projects:

- Create, rename, duplicate, delete projects
- Set project root directory
- View project statistics

## Persistence

- SQLite table: `Project`
- Config file: `project.yaml` in project root
- `project_id` columns on Dataset, TrainingRun, BaseWeight, CameraSession

## Related Files

- `yolo_studio/core/models/database.py`
- `yolo_studio/core/services/project_service.py`
- `yolo_studio/ui/dialogs/project_dialogs.py`
- `yolo_studio/ui/windows/main_window.py`

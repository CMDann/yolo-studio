# Remote Devices View

The Remote Devices tab manages edge device records and runs deploy/test jobs against them.

Main class: `RemoteTab` in `yolo_studio/ui/tabs/remote_tab.py`

## Device Manager Section

Capabilities:

- Add remote device records (name/type/host/port/auth token)
- Remove selected device
- Ping all devices asynchronously
- Render device cards with status badges and last-seen metadata

Persistence:

- `RemoteDevice` table (SQLite via SQLAlchemy)

Ping path:

- Preferred: `core.services.remote_manager.RemoteManager.ping_device(...)`
- Fallback: plain TCP probe on host:port

## Test Runner Section

Inputs:

- Device selector
- Saved model selector (saved `TrainingRun` rows)
- Dataset selector or custom dataset folder
- Confidence and IoU sliders

Outputs:

- Progress/status
- Metrics (`mAP50`, `mAP50_95`, `precision`, `recall`, `speed_ms`)
- Thumbnail strip of result images
- Save Results action to persist `RemoteTestResult`

## Deploy/Test Execution Paths

Worker class: `DeployTestWorker`

Primary path:

- Uses `RemoteManager` APIs:
  - `deploy_model`
  - `run_inference`
  - `get_results`
  - combined `deploy_and_test`

Fallback path:

- If remote manager route is unavailable/incompatible:
  - Runs local Ultralytics validation/prediction
  - Generates preview images
  - Returns normalized metrics payload

## Result Persistence

"Save Results" inserts a `RemoteTestResult` row with:

- Device and training run IDs
- Dataset path
- Number of tested images
- Metrics
- Output images directory
- Notes

## Remote Protocol Compatibility

The tab expects a WebSocket agent that supports message types such as:

- `PING` / `PONG`
- `DEPLOY_MODEL`
- `RUN_INFERENCE`
- `INFERENCE_PROGRESS`
- `INFERENCE_COMPLETE`
- `GET_RESULTS`
- `RESULT_CHUNK`
- `RESULTS_COMPLETE`

Reference implementation:

- `yolo_studio/edge/jetson_agent.py`

## Related Files

- `yolo_studio/ui/tabs/remote_tab.py`
- `yolo_studio/core/services/remote_manager.py`
- `yolo_studio/edge/jetson_agent.py`
- `yolo_studio/core/models/database.py`

## Project Scope

- Saved model and dataset selectors are filtered to the active project.

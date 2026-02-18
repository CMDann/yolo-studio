# Main Shell

The application shell is the top-level Qt window that hosts all tabs, status widgets, and the log dock.

## Entry and Boot Flow

- App entrypoint: `yolo_studio/main.py`
- Main window class: `yolo_studio/ui/windows/main_window.py`
- Theme application: `yolo_studio/ui/styles/theme.py`
- Database schema init on startup: `core.models.database.init_db()`

Startup sequence in plain terms:

1. Configure Python logging.
2. Create `QApplication`.
3. Apply global Qt theme and palette.
4. Initialize SQLite tables (if missing).
5. Create and show `MainWindow`.

## Visible Shell Components

`MainWindow` contains:

- A central `QTabWidget` with:
  - Training
  - Datasets
  - Discover
  - Remote Devices
- A status bar with persistent summary labels.
- A dockable log panel (`LogPanel`) at the bottom by default.
- A minimal `View` menu for toggling the log dock.

## Tab Loading Behavior

Each tab is created through `_build_tab(...)`:

- If import fails, the shell shows a fallback `PlaceholderTab`.
- If construction throws at runtime, it also falls back to `PlaceholderTab`.

This fallback mechanism is why "Step X" placeholder text can still appear even if a real tab exists in source: a runtime import/init exception will trigger placeholder rendering.

## Logging Bridge

`MainWindow` attaches `QtLogHandler` to the root logger:

- Source: `yolo_studio/ui/widgets/log_panel.py`
- Transport: Python logging record -> Qt signal -> `QTextEdit`
- Benefit: background worker logs appear in-app without blocking the UI thread.

## Related Files

- `yolo_studio/main.py`
- `yolo_studio/ui/windows/main_window.py`
- `yolo_studio/ui/widgets/log_panel.py`
- `yolo_studio/ui/styles/theme.py`

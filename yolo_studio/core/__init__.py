"""Core services package for YOLO Studio.

This package intentionally avoids eager imports so CLI/database-only tooling can
import `core` without requiring optional GUI/runtime dependencies.
"""

__all__ = [
    "database",
    "dataset_manager",
    "models",
    "remote_manager",
    "services",
    "trainer",
    "workers",
]

"""Database models and session utilities for YOLO Studio.

This module defines the SQLAlchemy ORM models for datasets, training runs,
remote devices, and remote test results. It also exposes helpers for creating
database sessions and initializing the SQLite schema.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum as SQLEnum, Float, ForeignKey
from sqlalchemy import Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class DatasetSource(str, Enum):
    """Enum of supported dataset origins."""

    MANUAL = "manual"
    ROBOFLOW = "roboflow"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    OPEN_IMAGES = "open_images"


class TrainingRunStatus(str, Enum):
    """Enum of training run lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class RemoteDeviceType(str, Enum):
    """Enum of supported remote edge device types."""

    JETSON_NANO = "jetson_nano"
    XAVIER = "xavier"
    RASPBERRY_PI = "raspberry_pi"
    OTHER = "other"


class RemoteDeviceStatus(str, Enum):
    """Enum of remote device connectivity states."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class BaseWeightSource(str, Enum):
    """Enum of base-weight provenance sources."""

    HUGGINGFACE = "huggingface"
    ROBOFLOW = "roboflow"
    LOCAL = "local"
    OTHER = "other"


def utc_now() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        datetime: Timezone-aware UTC datetime.
    """

    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


class Project(Base):
    """Project metadata for organizing YOLO Studio assets."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    root_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    git_remote: Mapped[str | None] = mapped_column(String(512), nullable=True)
    git_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Dataset(Base):
    """Dataset metadata stored in the YOLO Studio database."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    source: Mapped[DatasetSource] = mapped_column(
        SQLEnum(DatasetSource, name="dataset_source"),
        default=DatasetSource.MANUAL,
        nullable=False,
    )
    roboflow_project_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hf_repo_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    class_names: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    num_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    num_classes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class TrainingRun(Base):
    """Training run metadata and metrics for a YOLO experiment."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    model_architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[TrainingRunStatus] = mapped_column(
        SQLEnum(TrainingRunStatus, name="training_run_status"),
        default=TrainingRunStatus.PENDING,
        nullable=False,
        index=True,
    )

    best_map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_loss: Mapped[float | None] = mapped_column(Float, nullable=True)

    output_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    weights_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    is_saved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    config_yaml: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="training_runs")
    remote_test_results: Mapped[list["RemoteTestResult"]] = relationship(
        back_populates="training_run",
        cascade="all, delete-orphan",
    )


class RemoteDevice(Base):
    """Remote edge device metadata used for deployment and testing."""

    __tablename__ = "remote_devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    device_type: Mapped[RemoteDeviceType] = mapped_column(
        SQLEnum(RemoteDeviceType, name="remote_device_type"),
        default=RemoteDeviceType.OTHER,
        nullable=False,
    )
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    auth_token: Mapped[str] = mapped_column(String(255), nullable=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[RemoteDeviceStatus] = mapped_column(
        SQLEnum(RemoteDeviceStatus, name="remote_device_status"),
        default=RemoteDeviceStatus.UNKNOWN,
        nullable=False,
        index=True,
    )

    test_results: Mapped[list["RemoteTestResult"]] = relationship(
        back_populates="device",
        cascade="all, delete-orphan",
    )


class RemoteTestResult(Base):
    """Metrics and artifacts produced by remote inference test runs."""

    __tablename__ = "remote_test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("remote_devices.id"), nullable=False, index=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    test_dataset_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    num_images_tested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)

    output_images_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    device: Mapped[RemoteDevice] = relationship(back_populates="test_results")
    training_run: Mapped[TrainingRun] = relationship(back_populates="remote_test_results")


class AppSetting(Base):
    """Simple key/value settings stored in the database."""

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False, default=dict)


class CameraSession(Base):
    """Metadata for a real-time camera inference session."""

    __tablename__ = "camera_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), nullable=False, index=True)
    device_index: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    output_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    detection_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    training_run: Mapped[TrainingRun] = relationship()


class BaseWeight(Base):
    """Model registry entry for reusable base weights."""

    __tablename__ = "base_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source: Mapped[BaseWeightSource] = mapped_column(
        SQLEnum(BaseWeightSource, name="base_weight_source"),
        default=BaseWeightSource.OTHER,
        nullable=False,
    )
    repo_id: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    task: Mapped[str | None] = mapped_column(String(255), nullable=True)
    downloads: Mapped[int | None] = mapped_column(Integer, nullable=True)
    likes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "yolo_studio.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
DATABASE_URL = os.getenv("YOLO_STUDIO_DB_URL", DEFAULT_DATABASE_URL)

# SQLite is used by default; check_same_thread=False is required for QThread workers.
SQLITE_CONNECT_ARGS: dict[str, Any] = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    future=True,
    echo=False,
    connect_args=SQLITE_CONNECT_ARGS,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_session() -> Session:
    """Create a new SQLAlchemy session.

    Returns:
        Session: New database session bound to the global engine.
    """

    return SessionLocal()


def init_db() -> None:
    """Create all tables for YOLO Studio if they do not exist."""

    Base.metadata.create_all(bind=engine)
    _apply_sqlite_migrations()


def _apply_sqlite_migrations() -> None:
    """Apply lightweight SQLite schema updates when needed."""

    if not str(engine.url).startswith("sqlite"):
        return

    with engine.connect() as conn:
        result = conn.exec_driver_sql("PRAGMA table_info(remote_test_results)")
        columns = {row[1] for row in result}

        if "source_type" not in columns:
            conn.exec_driver_sql("ALTER TABLE remote_test_results ADD COLUMN source_type TEXT")
        if "source_path" not in columns:
            conn.exec_driver_sql("ALTER TABLE remote_test_results ADD COLUMN source_path TEXT")

        _ensure_column(conn, "datasets", "project_id", "INTEGER")
        _ensure_column(conn, "training_runs", "project_id", "INTEGER")
        _ensure_column(conn, "base_weights", "project_id", "INTEGER")
        _ensure_column(conn, "camera_sessions", "project_id", "INTEGER")


def _ensure_column(conn: Any, table: str, column: str, ddl_type: str) -> None:
    exists = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not exists:
        return

    result = conn.exec_driver_sql(f"PRAGMA table_info({table})")
    columns = {row[1] for row in result}
    if column not in columns:
        conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


__all__ = [
    "Base",
    "Project",
    "Dataset",
    "DatasetSource",
    "TrainingRun",
    "TrainingRunStatus",
    "RemoteDevice",
    "RemoteDeviceType",
    "RemoteDeviceStatus",
    "RemoteTestResult",
    "AppSetting",
    "CameraSession",
    "BaseWeight",
    "BaseWeightSource",
    "DATABASE_URL",
    "engine",
    "get_session",
    "init_db",
]

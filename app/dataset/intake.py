"""Licensed source-video registration with immutable content identity."""

from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path
from typing import Any

from app.benchmark.fingerprints import streaming_file_sha256
from app.dataset.models import (
    IntakeRegistry,
    PermissionStatus,
    SourceType,
    VehicleClass,
    VideoIntakeRecord,
    VideoResolution,
)


class DuplicateVideoError(ValueError):
    pass


def inspect_video(path: str | Path) -> tuple[float, VideoResolution, float]:
    cv2: Any = importlib.import_module("cv2")
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open intake video: {source}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            "video metadata is invalid: "
            f"frames={frames}, fps={fps}, width={width}, height={height}"
        )
    return frames / fps, VideoResolution(width=width, height=height), fps


def register_video(
    registry: IntakeRegistry,
    video_path: str | Path,
    *,
    video_id: str,
    source_group_id: str,
    source_type: SourceType,
    source_reference: str,
    acquisition_date: date,
    permission_status: PermissionStatus,
    redistribution_allowed: bool,
    benchmark_use_allowed: bool,
    notes: str | None = None,
    scenario_tags: list[str] | None = None,
    vehicle_classes: list[VehicleClass] | None = None,
    allow_duplicate_content: bool = False,
    replace_existing: bool = False,
) -> tuple[IntakeRegistry, VideoIntakeRecord, list[str]]:
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"intake video not found: {source}")
    sha256 = streaming_file_sha256(source)
    duplicate_ids = sorted(
        record.video_id
        for record in registry.videos
        if record.source_video_sha256 == sha256 and record.video_id != video_id
    )
    warnings = []
    if duplicate_ids:
        message = (
            "DUPLICATE_VIDEO_CONTENT: source SHA-256 is already registered as "
            + ", ".join(duplicate_ids)
        )
        if not allow_duplicate_content:
            raise DuplicateVideoError(message)
        warnings.append(message)
    existing = next(
        (record for record in registry.videos if record.video_id == video_id), None
    )
    if existing is not None and not replace_existing:
        raise ValueError(
            f"video_id {video_id!r} already exists; use explicit update mode"
        )
    duration, resolution, fps = inspect_video(source)
    record = VideoIntakeRecord(
        video_id=video_id,
        source_group_id=source_group_id,
        source_type=source_type,
        source_reference=source_reference,
        acquisition_date=acquisition_date,
        license_or_permission_status=permission_status,
        redistribution_allowed=redistribution_allowed,
        benchmark_use_allowed=benchmark_use_allowed,
        notes=notes,
        source_video_sha256=sha256,
        source_video_size_bytes=source.stat().st_size,
        source_identity_verified=True,
        duration_seconds=duration,
        resolution=resolution,
        fps=fps,
        original_filename=source.name,
        scenario_tags=scenario_tags or [],
        vehicle_classes=vehicle_classes or [],
    )
    videos = [item for item in registry.videos if item.video_id != video_id]
    videos.append(record)
    updated = registry.model_copy(
        update={"videos": sorted(videos, key=lambda item: item.video_id)}
    )
    return updated, record, warnings

"""Normalize Phase 3 runtime event records into stable benchmark predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.benchmark.models import (
    PredictedEvent,
    PredictionDocument,
    RuntimePerformance,
    VersionMetadata,
)


def load_prediction_document(path: str | Path) -> PredictionDocument:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction cache not found: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8") as stream:
        return PredictionDocument.model_validate(json.load(stream))


def write_prediction_document(document: PredictionDocument, path: str | Path) -> None:
    prediction_path = Path(path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as stream:
        json.dump(
            document.model_dump(mode="json"),
            stream,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        stream.write("\n")


def adapt_runtime_event(
    record: dict[str, Any],
    *,
    video_id: str,
    run_directory: Path,
) -> PredictedEvent:
    start = float(record["event_start_timestamp_seconds"])
    end_value = record.get("event_end_timestamp_seconds")
    end = (
        float(end_value)
        if end_value is not None
        else start + float(record.get("duration_seconds", 0.0))
    )
    if end <= start:
        end = start + max(float(record.get("duration_seconds", 0.0)), 1e-6)

    geometry = record.get("geometry_integrity") or {}
    overtaking = record.get("overtaking_assessment")
    traffic = record.get("traffic_context") or {}
    lifecycle = record.get("candidate_lifecycle") or {}
    event_id = str(record["event_id"])
    event_directory = run_directory / "events" / event_id
    metadata_path = event_directory / "metadata.json"
    reason_codes = _unique_strings(
        [
            *record.get("review_reason_codes", []),
            *geometry.get("reason_codes", []),
        ]
    )
    overtaking_status = (
        str(overtaking.get("status", "unavailable"))
        if isinstance(overtaking, dict)
        else "unavailable"
    )
    evidence_confidence = record.get("evidence_confidence_score")
    confidence_value = (
        evidence_confidence
        if evidence_confidence is not None
        else record.get("confidence_score", 0.0)
    )
    if confidence_value is None:
        confidence_value = 0.0
    return PredictedEvent(
        event_id=event_id,
        video_id=video_id,
        track_id=record.get("track_id"),
        start_seconds=start,
        end_seconds=end,
        event_type=str(record.get("event_type", "left_lane_review_candidate")),
        confidence=float(confidence_value),
        reason_codes=reason_codes,
        geometry_status=str(geometry.get("status", "unavailable")),
        overtaking_status=overtaking_status,
        review_status=str(record.get("review_status", "unavailable")),
        congestion_level=traffic.get("congestion_level"),
        right_lane_available=traffic.get("right_lane_available"),
        lifecycle_state=lifecycle.get("state"),
        source_metadata_path=str(metadata_path) if metadata_path.exists() else None,
        representative_frame_path=_artifact_path(
            event_directory, record.get("representative_frame")
        ),
        event_video_clip_path=_artifact_path(
            event_directory, record.get("event_video_clip")
        ),
    )


def prediction_document_from_run(
    run_directory: str | Path,
    *,
    video_id: str,
    source_file: str | None,
    source_video_sha256: str | None = None,
    source_video_size_bytes: int | None = None,
    performance: RuntimePerformance | None = None,
    versions: VersionMetadata | None = None,
) -> PredictionDocument:
    run_path = Path(run_directory).resolve()
    events_path = run_path / "events.jsonl"
    predictions: list[PredictedEvent] = []
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid runtime event JSON at {events_path}:{line_number}"
                    ) from exc
                prediction = adapt_runtime_event(
                    record, video_id=video_id, run_directory=run_path
                )
                if prediction.review_status == "pending_human_review":
                    predictions.append(prediction)
    predictions.sort(key=lambda event: (event.start_seconds, event.event_id))
    return PredictionDocument(
        video_id=video_id,
        source_file=source_file,
        source_video_sha256=source_video_sha256,
        source_video_size_bytes=source_video_size_bytes,
        predictions=predictions,
        cancelled_event_count=_count_jsonl(run_path / "cancelled_events.jsonl"),
        performance=performance,
        versions=versions or VersionMetadata(),
    )


def _artifact_path(directory: Path, name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    path = directory / name
    return str(path) if path.is_file() else None


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))

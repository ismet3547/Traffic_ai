"""Collect and reconcile video-duration evidence before rate calculation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Literal

from app.benchmark.annotations import resolve_manifest_path
from app.benchmark.models import (
    AnnotationDocument,
    DurationEvidence,
    DurationValidationConfig,
    DurationValidationResult,
    ManifestVideo,
    PredictionDocument,
)


def resolve_video_duration(
    manifest_path: str | Path,
    video: ManifestVideo,
    annotation: AnnotationDocument,
    prediction: PredictionDocument,
    config: DurationValidationConfig,
) -> DurationValidationResult:
    evidence: list[DurationEvidence] = []
    if video.duration_seconds is not None:
        evidence.append(
            DurationEvidence(
                source="manifest",
                seconds=video.duration_seconds,
                confidence="low",
            )
        )
    if annotation.video_duration_seconds is not None:
        evidence.append(
            DurationEvidence(
                source="annotation",
                seconds=annotation.video_duration_seconds,
                confidence="medium",
            )
        )
    if prediction.performance is not None:
        evidence.append(
            DurationEvidence(
                source="prediction_cache",
                seconds=prediction.performance.video_duration_seconds,
                confidence="medium",
            )
        )
    if video.path is not None:
        video_path = resolve_manifest_path(manifest_path, video.path)
        if video_path.is_file():
            evidence.append(
                DurationEvidence(
                    source="video_metadata",
                    seconds=probe_video_duration(video_path),
                    confidence="high",
                )
            )
    return validate_duration_evidence(video.id, evidence, config)


def validate_duration_evidence(
    video_id: str,
    evidence: list[DurationEvidence],
    config: DurationValidationConfig,
) -> DurationValidationResult:
    if not evidence:
        raise ValueError(
            f"video {video_id!r} has no duration evidence; provide manifest, "
            "annotation, video metadata, or prediction-cache duration"
        )
    sources = [item.source for item in evidence]
    if len(sources) != len(set(sources)):
        raise ValueError(f"video {video_id!r} has duplicate duration evidence sources")
    for first, second in combinations(evidence, 2):
        tolerance = max(
            config.absolute_tolerance_seconds,
            config.relative_tolerance * max(first.seconds, second.seconds),
        )
        difference = abs(first.seconds - second.seconds)
        if difference > tolerance + 1e-12:
            raise ValueError(
                "DURATION_EVIDENCE_MISMATCH: "
                f"video {video_id!r} {first.source}={first.seconds:.6f}s vs "
                f"{second.source}={second.seconds:.6f}s; difference "
                f"{difference:.6f}s exceeds tolerance {tolerance:.6f}s"
            )

    precedence = {
        "video_metadata": 0,
        "prediction_cache": 1,
        "annotation": 2,
        "manifest": 3,
    }
    ordered = sorted(evidence, key=lambda item: precedence[item.source])
    selected = ordered[0]
    has_video_metadata = any(item.source == "video_metadata" for item in evidence)
    if has_video_metadata:
        status: Literal[
            "verified_video_metadata",
            "consistent_multiple_sources",
            "single_source_unverified",
        ] = "verified_video_metadata"
        confidence: Literal["high", "medium", "low"] = "high"
    elif len(evidence) >= 2:
        status = "consistent_multiple_sources"
        confidence = "medium"
    else:
        status = "single_source_unverified"
        confidence = "low"
    return DurationValidationResult(
        duration_seconds_used=selected.seconds,
        duration_source=selected.source,
        duration_validation_status=status,
        denominator_confidence=confidence,
        evidence=sorted(evidence, key=lambda item: item.source),
    )


def probe_video_duration(path: str | Path) -> float:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - normal project dependency
        raise RuntimeError("OpenCV is required to validate video duration") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open benchmark video for duration: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(
            f"video metadata has impossible duration inputs: frames={frames}, fps={fps}"
        )
    duration = frames / fps
    if duration <= 0:
        raise ValueError(f"video metadata produced impossible duration: {duration}")
    return duration

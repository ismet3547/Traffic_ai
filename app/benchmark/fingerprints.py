"""Canonical hashes that define dataset and evaluation comparability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.benchmark.annotations import resolve_manifest_path
from app.benchmark.models import (
    ANNOTATION_ROLE_BY_LABEL,
    BenchmarkSettings,
    ManifestVideo,
)


def canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_fingerprint_payload(
    manifest_path: str | Path,
    videos: list[ManifestVideo],
    annotation_hashes: dict[str, str],
    annotation_versions: list[str],
) -> dict[str, Any]:
    ordered_videos = []
    for video in sorted(videos, key=lambda item: item.id):
        annotations = []
        for index, value in enumerate(video.annotation_paths):
            path = resolve_manifest_path(manifest_path, value)
            key = f"{video.id}:{index}:{path.name}"
            annotations.append(
                {
                    "position": index,
                    "file_name": path.name,
                    "sha256": annotation_hashes.get(key),
                }
            )
        ordered_videos.append(
            {
                "video_id": video.id,
                "split": video.split.value,
                "duration_seconds_declared": video.duration_seconds,
                "annotations": annotations,
            }
        )
    return {
        "videos": ordered_videos,
        "annotation_schema_versions": sorted(annotation_versions),
    }


def evaluation_fingerprint_payload(settings: BenchmarkSettings) -> dict[str, Any]:
    return {
        "matching": settings.matching.model_dump(mode="json"),
        "ignored_regions": settings.ignored_regions.model_dump(mode="json"),
        "control_events": settings.control_events.model_dump(mode="json"),
        "duration_validation": settings.duration_validation.model_dump(mode="json"),
        "headline_confidences": sorted(
            confidence.value for confidence in settings.headline_confidences
        ),
        "minimum_prediction_confidence": settings.minimum_prediction_confidence,
        "annotation_roles": {
            label.value: role.value
            for label, role in sorted(
                ANNOTATION_ROLE_BY_LABEL.items(), key=lambda item: item[0].value
            )
        },
        "positive_label_set": ["unnecessary_left_lane_occupation"],
    }

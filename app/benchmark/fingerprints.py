"""Canonical dataset content and evaluation-protocol identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.benchmark.annotations import resolve_manifest_path
from app.benchmark.models import (
    ANNOTATION_ROLE_BY_LABEL,
    BenchmarkSettings,
    DatasetIdentityStatus,
    EvaluationProtocolIdentity,
    ManifestVideo,
    PredictionDocument,
    VideoIdentity,
    VideoIdentityMode,
)
from app.benchmark.protocol import current_evaluation_protocol

HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def streaming_file_sha256(path: str | Path) -> str:
    """Hash a file in bounded chunks so large source videos are not loaded at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_video_identities(
    manifest_path: str | Path,
    videos: list[ManifestVideo],
    predictions: Mapping[str, PredictionDocument],
) -> dict[str, VideoIdentity]:
    """Resolve raw or cache-preserved source identity and verify their relationship."""

    identities: dict[str, VideoIdentity] = {}
    for video in sorted(videos, key=lambda item: item.id):
        prediction = predictions.get(video.id)
        if prediction is None:
            raise ValueError(f"prediction cache missing video ID: {video.id}")
        if prediction.video_id != video.id:
            raise ValueError(
                f"prediction cache video_id {prediction.video_id!r} does not match "
                f"manifest video ID {video.id!r}"
            )
        raw_path = (
            resolve_manifest_path(manifest_path, video.path)
            if video.path is not None
            else None
        )
        cached_sha256 = prediction.source_video_sha256
        cached_size = prediction.source_video_size_bytes

        if raw_path is not None and raw_path.is_file():
            actual_size = raw_path.stat().st_size
            actual_sha256 = streaming_file_sha256(raw_path)
            if cached_sha256 is not None and (
                cached_sha256 != actual_sha256 or cached_size != actual_size
            ):
                raise ValueError(
                    "PREDICTION_CACHE_SOURCE_MISMATCH: "
                    f"video {video.id!r} cache sha256={cached_sha256} "
                    f"size={cached_size} does not match current source "
                    f"sha256={actual_sha256} size={actual_size}"
                )
            if cached_sha256 is None:
                identities[video.id] = VideoIdentity(
                    video_id=video.id,
                    source_path=str(raw_path),
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                    identity_mode=VideoIdentityMode.FULL_SHA256,
                    verified=False,
                    reason_codes=["PREDICTION_CACHE_SOURCE_IDENTITY_MISSING"],
                )
            else:
                identities[video.id] = VideoIdentity(
                    video_id=video.id,
                    source_path=str(raw_path),
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                    identity_mode=VideoIdentityMode.FULL_SHA256,
                    verified=True,
                )
            continue

        if cached_sha256 is not None and cached_size is not None:
            identities[video.id] = VideoIdentity(
                video_id=video.id,
                source_path=str(raw_path) if raw_path is not None else None,
                sha256=cached_sha256,
                size_bytes=cached_size,
                identity_mode=VideoIdentityMode.CACHED_FULL_SHA256,
                verified=True,
                reason_codes=["SOURCE_VIDEO_UNAVAILABLE_IDENTITY_FROM_CACHE"],
            )
            continue

        reasons = ["PREDICTION_CACHE_SOURCE_IDENTITY_MISSING"]
        if raw_path is None:
            reasons.append("SOURCE_VIDEO_PATH_NOT_CONFIGURED")
        else:
            reasons.append("SOURCE_VIDEO_UNAVAILABLE")
        identities[video.id] = VideoIdentity(
            video_id=video.id,
            source_path=str(raw_path) if raw_path is not None else None,
            identity_mode=VideoIdentityMode.UNVERIFIED,
            verified=False,
            reason_codes=reasons,
        )
    return identities


def dataset_identity_status(
    identities: Mapping[str, VideoIdentity],
) -> DatasetIdentityStatus:
    return (
        DatasetIdentityStatus.VERIFIED
        if identities and all(identity.verified for identity in identities.values())
        else DatasetIdentityStatus.UNVERIFIED
    )


def dataset_fingerprint_payload(
    videos: list[ManifestVideo],
    annotation_identities: Mapping[str, Mapping[str, str]],
    video_identities: Mapping[str, VideoIdentity],
    predictions: Mapping[str, PredictionDocument],
) -> dict[str, Any]:
    """Build path-independent, deterministic evidence identity.

    Raw and cache-preserved full SHA-256 evidence normalize to the same
    cryptographic verification mode. Acquisition provenance remains available in
    the report's per-video identity metadata but does not make byte-identical
    footage incomparable merely because one local file was renamed or removed.
    """

    ordered_videos = []
    for video in sorted(videos, key=lambda item: item.id):
        identity = video_identities[video.id]
        prediction = predictions[video.id]
        annotations = []
        for index, _ in enumerate(video.annotation_paths):
            key = f"{video.id}:{index}"
            annotation = annotation_identities.get(key, {})
            annotations.append(
                {
                    "position": index,
                    "sha256": annotation.get("sha256"),
                    "schema_version": annotation.get("schema_version"),
                }
            )
        cache_duration = (
            prediction.performance.video_duration_seconds
            if prediction.performance is not None
            else None
        )
        ordered_videos.append(
            {
                "video_id": video.id,
                "split": video.split.value,
                "source_video": {
                    "sha256": identity.sha256,
                    "size_bytes": identity.size_bytes,
                    "verification_mode": (
                        "full_sha256" if identity.verified else "unverified"
                    ),
                    "verified": identity.verified,
                },
                "duration_identity": {
                    "manifest_seconds": video.duration_seconds,
                    "prediction_cache_seconds": cache_duration,
                },
                "annotations": annotations,
            }
        )
    return {"videos": ordered_videos}


def evaluation_fingerprint_payload(
    settings: BenchmarkSettings,
    protocol: EvaluationProtocolIdentity | None = None,
) -> dict[str, Any]:
    active_protocol = protocol or current_evaluation_protocol()
    return {
        "evaluation_protocol": active_protocol.model_dump(mode="json"),
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

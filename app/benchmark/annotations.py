"""Load and validate benchmark manifests and ground-truth annotations."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.benchmark.models import AnnotationDocument, BenchmarkManifest, ManifestVideo


def load_annotation(path: str | Path) -> AnnotationDocument:
    annotation_path = Path(path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"annotation file not found: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return AnnotationDocument.model_validate(raw)


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return BenchmarkManifest.model_validate(raw)


def resolve_manifest_path(manifest_path: str | Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(manifest_path).resolve().parent / path).resolve()


def load_video_annotations(
    manifest_path: str | Path, video: ManifestVideo
) -> list[AnnotationDocument]:
    documents = [
        load_annotation(resolve_manifest_path(manifest_path, path))
        for path in video.annotation_paths
    ]
    for document in documents:
        if document.video_id != video.id:
            raise ValueError(
                f"annotation video_id {document.video_id!r} does not match "
                f"manifest video ID {video.id!r}"
            )
        if (
            video.path is not None
            and Path(document.source_file).name != Path(video.path).name
        ):
            raise ValueError(
                f"annotation source_file {document.source_file!r} does not match "
                f"manifest video file {Path(video.path).name!r}"
            )
        duration = video.duration_seconds or document.video_duration_seconds
        if duration is not None:
            outside = [
                event.event_id
                for event in document.events
                if event.end_seconds > duration + 1e-9
            ]
            if outside:
                raise ValueError(
                    f"annotation timestamps exceed duration for {video.id}: "
                    + ", ".join(sorted(outside))
                )
    return documents


def validate_manifest_references(
    manifest_path: str | Path,
    manifest: BenchmarkManifest,
    *,
    require_videos: bool,
    require_configs: bool,
) -> list[str]:
    """Return human-readable reference errors without stopping at the first one."""

    errors: list[str] = []
    for video in manifest.videos:
        if not video.enabled:
            continue
        for annotation in video.annotation_paths:
            annotation_path = resolve_manifest_path(manifest_path, annotation)
            if not annotation_path.is_file():
                errors.append(f"{video.id}: annotation not found: {annotation_path}")
                continue
            try:
                load_video_annotations(manifest_path, video)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{video.id}: {exc}")
                break
        if require_videos:
            if video.path is None:
                errors.append(f"{video.id}: video path is required")
            else:
                video_path = resolve_manifest_path(manifest_path, video.path)
                if not video_path.is_file():
                    errors.append(f"{video.id}: video not found: {video_path}")
        if require_configs:
            if video.config is None:
                errors.append(f"{video.id}: production config is required")
            else:
                config_path = resolve_manifest_path(manifest_path, video.config)
                if not config_path.is_file():
                    errors.append(f"{video.id}: config not found: {config_path}")
    return sorted(set(errors))

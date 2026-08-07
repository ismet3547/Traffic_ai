"""Inference orchestration, cache management, and reproducibility metadata."""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from app.benchmark.adapter import prediction_document_from_run
from app.benchmark.annotations import resolve_manifest_path
from app.benchmark.fingerprints import (
    canonical_sha256,
    dataset_fingerprint_payload,
    dataset_identity_status,
    evaluation_fingerprint_payload,
    resolve_video_identities,
    streaming_file_sha256,
)
from app.benchmark.models import (
    BenchmarkManifest,
    ManifestVideo,
    PredictionDocument,
    RuntimePerformance,
    VersionMetadata,
)
from app.benchmark.protocol import current_evaluation_protocol

LOGGER = logging.getLogger(__name__)


def select_videos(manifest: BenchmarkManifest, split: str) -> list[ManifestVideo]:
    videos = [video for video in manifest.videos if video.enabled]
    if split != "all":
        videos = [video for video in videos if video.split.value == split]
    if not videos:
        raise ValueError(f"manifest has no enabled videos for split {split!r}")
    return sorted(videos, key=lambda video: video.id)


def run_video_inference(
    manifest_path: str | Path,
    video: ManifestVideo,
    output_directory: str | Path,
    *,
    git_commit: str | None,
) -> PredictionDocument:
    if video.path is None or video.config is None:
        raise ValueError(
            f"video {video.id!r} requires path and config when inference is enabled"
        )
    source = resolve_manifest_path(manifest_path, video.path)
    config = resolve_manifest_path(manifest_path, video.config)
    if not source.is_file():
        raise FileNotFoundError(f"benchmark video not found: {source}")
    if not config.is_file():
        raise FileNotFoundError(f"production config not found: {config}")
    source_video_sha256 = streaming_file_sha256(source)
    source_video_size_bytes = source.stat().st_size

    inference_base = Path(output_directory) / "inference_runs" / video.id
    inference_base.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in inference_base.iterdir() if path.is_dir()}
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--config",
        str(config),
        "--input",
        str(source),
        "--output-dir",
        str(inference_base),
        "--log-level",
        "WARNING",
    ]
    LOGGER.info("Running inference for %s", video.id)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path = inference_base / "benchmark_inference.log"
    log_path.write_text(
        "$ "
        + subprocess.list2cmdline(command)
        + "\n\n"
        + completed.stdout
        + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"inference failed for {video.id} with exit code {completed.returncode}:\n"
            + completed.stderr.strip()
        )
    after = {path.resolve() for path in inference_base.iterdir() if path.is_dir()}
    new_directories = sorted(after - before, key=lambda path: path.stat().st_mtime_ns)
    if not new_directories:
        raise RuntimeError(f"inference for {video.id} did not create a run directory")
    run_directory = new_directories[-1]
    frames, duration = probe_video(source)
    identifiers = production_identifiers(config)
    performance = RuntimePerformance(
        total_processing_time_seconds=elapsed,
        video_duration_seconds=duration,
        frames_processed=frames,
        processing_fps=frames / elapsed if elapsed else 0.0,
        real_time_factor=duration / elapsed if elapsed else 0.0,
        average_frame_processing_time_ms=(elapsed * 1000.0 / frames if frames else 0.0),
        hardware=hardware_metadata(identifiers.get("detector_device")),
    )
    versions = VersionMetadata(
        git_commit=git_commit,
        policy_version=identifiers.get("policy_version"),
        detector_model_identifier=identifiers.get("detector_model_identifier"),
        tracker_identifier=identifiers.get("tracker_identifier"),
    )
    return prediction_document_from_run(
        run_directory,
        video_id=video.id,
        source_file=source.name,
        source_video_sha256=source_video_sha256,
        source_video_size_bytes=source_video_size_bytes,
        performance=performance,
        versions=versions,
    )


def probe_video(path: str | Path) -> tuple[int, float]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - normal project dependency
        raise RuntimeError("OpenCV is required to inspect benchmark videos") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open benchmark video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return frames, duration


def git_commit_hash(repository: str | Path | None = None) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        LOGGER.warning(
            "Git commit could not be determined: %s", completed.stderr.strip()
        )
        return None
    return completed.stdout.strip() or None


def git_worktree_dirty(repository: str | Path | None = None) -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        LOGGER.warning("Git worktree state could not be determined")
        return None
    return bool(completed.stdout.strip())


def production_identifiers(config_path: str | Path) -> dict[str, str | None]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return {
        "policy_version": (
            raw.get("rules", {}).get("left_lane", {}).get("policy_version")
        ),
        "detector_model_identifier": raw.get("detector", {}).get("model_path"),
        "detector_device": raw.get("detector", {}).get("device"),
        "tracker_identifier": "ByteTrack",
    }


def build_reproducibility_snapshot(
    manifest_path: str | Path,
    manifest: BenchmarkManifest,
    videos: list[ManifestVideo],
    predictions: dict[str, PredictionDocument],
    output_directory: str | Path,
    *,
    git_commit: str | None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    production_configs: dict[str, Any] = {}
    identifiers: dict[str, Any] = {}
    annotation_versions: set[str] = set()
    annotation_hashes: dict[str, str] = {}
    annotation_identities: dict[str, dict[str, str]] = {}
    for video in videos:
        if video.config is not None:
            config_path = resolve_manifest_path(manifest_path, video.config)
            if config_path.is_file():
                with config_path.open("r", encoding="utf-8") as stream:
                    production_configs[video.id] = yaml.safe_load(stream) or {}
                identifiers[video.id] = production_identifiers(config_path)
        for annotation_index, annotation_path in enumerate(video.annotation_paths):
            path = resolve_manifest_path(manifest_path, annotation_path)
            if path.is_file():
                with path.open("r", encoding="utf-8") as stream:
                    schema_version = str(json.load(stream).get("schema_version"))
                annotation_versions.add(schema_version)
                annotation_sha256 = file_sha256(path)
                annotation_hashes[f"{video.id}:{annotation_index}:{path.name}"] = (
                    annotation_sha256
                )
                annotation_identities[f"{video.id}:{annotation_index}"] = {
                    "sha256": annotation_sha256,
                    "schema_version": schema_version,
                }
    snapshot = {
        "benchmark_manifest": manifest.model_dump(mode="json"),
        "production_configs": dict(sorted(production_configs.items())),
    }
    resolved_config_hash = canonical_sha256(snapshot)
    video_identities = resolve_video_identities(manifest_path, videos, predictions)
    identity_status = dataset_identity_status(video_identities)
    dataset_payload = dataset_fingerprint_payload(
        videos,
        annotation_identities,
        video_identities,
        predictions,
    )
    evaluation_protocol = current_evaluation_protocol()
    evaluation_payload = evaluation_fingerprint_payload(
        manifest.benchmark, evaluation_protocol
    )
    production_config_hash = canonical_sha256(dict(sorted(production_configs.items())))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=True, allow_unicode=True)
    return {
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "config_hash_sha256": resolved_config_hash,
        "resolved_config_hash_sha256": resolved_config_hash,
        "production_config_hash_sha256": production_config_hash,
        "dataset_fingerprint": canonical_sha256(dataset_payload),
        "dataset_fingerprint_payload": dataset_payload,
        "dataset_identity_status": identity_status.value,
        "source_video_identities": {
            video_id: identity.model_dump(mode="json")
            for video_id, identity in sorted(video_identities.items())
        },
        "evaluation_fingerprint": canonical_sha256(evaluation_payload),
        "evaluation_fingerprint_payload": evaluation_payload,
        "evaluation_protocol": evaluation_protocol.model_dump(mode="json"),
        "benchmark_schema_version": manifest.schema_version,
        "annotation_schema_versions": sorted(annotation_versions),
        "annotation_hashes_sha256": dict(sorted(annotation_hashes.items())),
        "production_identifiers": identifiers,
        "resolved_config": "resolved_config.yaml",
    }


def file_sha256(path: str | Path) -> str:
    return streaming_file_sha256(path)


def hardware_metadata(configured_device: str | None = None) -> dict[str, Any]:
    device = str(configured_device).lower() if configured_device is not None else None
    if device == "cpu":
        execution_class = "cpu"
    elif device is not None and (device.startswith("cuda") or device.isdigit()):
        execution_class = "gpu_configured"
    else:
        execution_class = "auto_unverified"
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "configured_detector_device": configured_device,
        "execution_device_class": execution_class,
        "gpu_name": None,
    }
    try:
        torch: Any = importlib.import_module("torch")
        if execution_class == "gpu_configured" and torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - ultralytics normally provides torch
        pass
    return result

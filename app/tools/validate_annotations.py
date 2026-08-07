"""Validate annotation schemas, intervals, durations, and manifest references."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.benchmark.annotations import (
    load_annotation,
    load_manifest,
    load_video_annotations,
    resolve_manifest_path,
    validate_manifest_references,
)
from app.benchmark.duration import (
    probe_video_duration,
    resolve_video_duration,
    validate_duration_evidence,
)
from app.benchmark.models import (
    DurationEvidence,
    DurationValidationConfig,
    PredictionDocument,
)
from app.dataset.io import load_annotation as load_dataset_annotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate traffic benchmark annotations."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--annotation", nargs="+")
    group.add_argument(
        "--dataset-annotation",
        nargs="+",
        help="Validate Phase 4.2 prediction-free annotation documents and locks.",
    )
    parser.add_argument(
        "--video", help="Optional video for standalone duration validation"
    )
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="Validate schemas and annotation references without local video files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    checked = 0
    if args.manifest:
        path = Path(args.manifest).resolve()
        try:
            manifest = load_manifest(path)
            errors.extend(
                validate_manifest_references(
                    path,
                    manifest,
                    require_videos=not args.allow_missing_videos,
                    require_configs=False,
                )
            )
            if not args.allow_missing_videos:
                for video in manifest.videos:
                    if not video.enabled or video.path is None:
                        continue
                    video_path = resolve_manifest_path(path, video.path)
                    if not video_path.is_file():
                        continue
                    for document in load_video_annotations(path, video):
                        duration_validation = resolve_video_duration(
                            path,
                            video,
                            document,
                            PredictionDocument(video_id=video.id),
                            manifest.benchmark.duration_validation,
                        )
                        outside = [
                            event.event_id
                            for event in document.events
                            if event.end_seconds
                            > duration_validation.duration_seconds_used + 1e-9
                        ]
                        if outside:
                            errors.append(
                                f"{video.id}: timestamps exceed actual video duration "
                                f"{duration_validation.duration_seconds_used:.3f}s: "
                                + ", ".join(sorted(outside))
                            )
            checked = sum(video.enabled for video in manifest.videos)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    elif args.dataset_annotation:
        for value in args.dataset_annotation:
            try:
                dataset_document = load_dataset_annotation(value)
                checked += 1
                if args.video:
                    actual = probe_video_duration(args.video)
                    if abs(actual - dataset_document.video_duration_seconds) > 1.0:
                        errors.append(
                            f"{value}: annotation/video duration differs by more than 1 second"
                        )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                errors.append(f"{value}: {exc}")
    else:
        video_duration_seconds: float | None = None
        if args.video:
            try:
                video_duration_seconds = probe_video_duration(args.video)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                errors.append(str(exc))
        for value in args.annotation:
            try:
                document = load_annotation(value)
                checked += 1
                if video_duration_seconds is not None:
                    evidence = [
                        DurationEvidence(
                            source="video_metadata",
                            seconds=video_duration_seconds,
                            confidence="high",
                        )
                    ]
                    if document.video_duration_seconds is not None:
                        evidence.append(
                            DurationEvidence(
                                source="annotation",
                                seconds=document.video_duration_seconds,
                                confidence="medium",
                            )
                        )
                    validate_duration_evidence(
                        document.video_id,
                        evidence,
                        DurationValidationConfig(),
                    )
                    outside = [
                        event.event_id
                        for event in document.events
                        if event.end_seconds > video_duration_seconds + 1e-9
                    ]
                    if outside:
                        errors.append(
                            f"{value}: timestamps exceed video duration: "
                            + ", ".join(sorted(outside))
                        )
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{value}: {exc}")
    if errors:
        print("Annotation validation failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1
    print(f"Annotation validation passed ({checked} document/video entry(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from app.benchmark.runner import probe_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate traffic benchmark annotations."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--annotation", nargs="+")
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
                    _, actual_duration = probe_video(video_path)
                    for document in load_video_annotations(path, video):
                        outside = [
                            event.event_id
                            for event in document.events
                            if event.end_seconds > actual_duration + 1e-9
                        ]
                        if outside:
                            errors.append(
                                f"{video.id}: timestamps exceed actual video duration "
                                f"{actual_duration:.3f}s: " + ", ".join(sorted(outside))
                            )
            checked = sum(video.enabled for video in manifest.videos)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        duration = None
        if args.video:
            try:
                _, duration = probe_video(args.video)
            except (FileNotFoundError, RuntimeError) as exc:
                errors.append(str(exc))
        for value in args.annotation:
            try:
                document = load_annotation(value)
                checked += 1
                if duration is not None:
                    outside = [
                        event.event_id
                        for event in document.events
                        if event.end_seconds > duration + 1e-9
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

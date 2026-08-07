"""Create or extend a blinded Phase 4.2 annotation from explicit event JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.benchmark.fingerprints import streaming_file_sha256
from app.dataset.intake import inspect_video
from app.dataset.io import load_annotation, lock_annotation, save_annotation
from app.dataset.models import DatasetAnnotation, DatasetEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--video-id")
    parser.add_argument(
        "--event-json",
        action="append",
        default=[],
        help="Inline event JSON or path to a JSON event/list. Repeatable.",
    )
    parser.add_argument("--override-lock", action="store_true")
    parser.add_argument("--override-reason")
    parser.add_argument(
        "--lock",
        action="store_true",
        help="Lock the completed independent pass with a content hash.",
    )
    return parser


def _events(values: list[str]) -> list[DatasetEvent]:
    result: list[DatasetEvent] = []
    for value in values:
        path = Path(value)
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else json.loads(value)
        )
        items = payload if isinstance(payload, list) else [payload]
        result.extend(DatasetEvent.model_validate(item) for item in items)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video).resolve()
    duration, _, fps = inspect_video(video)
    digest = streaming_file_sha256(video)
    output = Path(args.output)
    additions = _events(args.event_json)
    if output.is_file():
        document = load_annotation(output)
        if (
            document.annotator_id != args.annotator_id
            or document.source_video_sha256 != digest
        ):
            raise ValueError(
                "existing annotation identity does not match annotator/video"
            )
        document = document.model_copy(
            update={"events": [*document.events, *additions]}
        )
    else:
        document = DatasetAnnotation(
            video_id=args.video_id or video.stem,
            source_video_sha256=digest,
            source_video_size_bytes=video.stat().st_size,
            source_file=video.name,
            fps=fps,
            video_duration_seconds=duration,
            annotator_id=args.annotator_id,
            created_at=datetime.now(timezone.utc),
            events=additions,
        )
    document = DatasetAnnotation.model_validate(document.model_dump(mode="python"))
    if args.lock:
        document = lock_annotation(document)
    saved = save_annotation(
        document,
        output,
        override_lock=args.override_lock,
        override_reason=args.override_reason,
    )
    print(f"Saved {len(saved.events)} blinded event(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

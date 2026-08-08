"""Create the official exact-coverage review for a frozen pilot baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.dataset.pilot import load_pilot_manifest
from app.dataset.pilot_review import (
    FailureReviewEntry,
    assess_failure_review,
    build_failure_review_document,
    derive_required_failures,
    load_baseline_review_identity,
    render_failure_review_summary,
    save_failure_review,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-manifest",
        default="data/benchmark/pilot/mini_pilot_manifest.json",
    )
    parser.add_argument("--baseline")
    parser.add_argument(
        "--reviews",
        help="JSON list (or object with a reviews list) of structured review entries",
    )
    parser.add_argument("--output")
    parser.add_argument("--summary-output")
    args = parser.parse_args(argv)

    manifest_path = Path(args.pilot_manifest)
    manifest = load_pilot_manifest(manifest_path)
    base = manifest_path.resolve().parent
    baseline = _resolve(base, args.baseline or manifest.artifacts.baseline_directory)
    output = _resolve(base, args.output or manifest.artifacts.failure_review)
    summary_output = _resolve(base, args.summary_output or "pilot_failure_summary.md")
    identity, report = load_baseline_review_identity(baseline)
    if identity.pilot_id != manifest.pilot_id:
        raise ValueError("baseline pilot_id differs from pilot manifest")
    required = derive_required_failures(identity, report)
    _print_required(required)
    if args.reviews is None:
        print("No review input supplied; no official artifact was written.")
        print("Use --reviews after every listed FP/FN has a structured human review.")
        return 0 if not required else 2

    reviews = _load_reviews(args.reviews)
    document = build_failure_review_document(identity, reviews)
    coverage = assess_failure_review(identity, required, document)
    if not coverage.complete:
        print(json.dumps(coverage.model_dump(mode="json"), indent=2, sort_keys=True))
        raise ValueError("official failure review is not complete and exact")
    save_failure_review(document, output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        render_failure_review_summary(identity, coverage, document), encoding="utf-8"
    )
    print(f"Failure review: {output}")
    print(f"Failure review SHA-256: {document.content_sha256}")
    print(f"Coverage: {coverage.reviewed_count}/{coverage.required_count}")
    return 0


def _load_reviews(path: str) -> list[FailureReviewEntry]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value: Any = json.load(stream)
    raw = value.get("reviews") if isinstance(value, dict) else value
    if not isinstance(raw, list):
        raise TypeError("review input must be a JSON list or object with reviews list")
    return [FailureReviewEntry.model_validate(item) for item in raw]


def _print_required(required: list[Any]) -> None:
    print(f"Required frozen-baseline FP/FN reviews: {len(required)}")
    for item in required:
        print(
            f"{item.failure_id} {item.failure_type} video={item.video_id} "
            f"prediction={item.prediction_event_id or '-'} "
            f"ground_truth={item.ground_truth_event_id or '-'} "
            f"time={item.timestamp_start:.3f}-{item.timestamp_end:.3f} "
            f"artifact={item.artifact_directory or '-'}"
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

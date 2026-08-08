"""Review the deterministic first-N current official agreement reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.dataset.io import load_agreement
from app.dataset.pilot import (
    current_required_agreement_reports,
    load_pilot_manifest,
)
from app.dataset.pilot_review import (
    FirstAgreementReviewSummary,
    assess_first_agreement_review,
    build_first_agreement_review_document,
    save_first_agreement_review,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-manifest",
        default="data/benchmark/pilot/mini_pilot_manifest.json",
    )
    parser.add_argument("--summary", required=True, help="structured summary JSON")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    manifest_path = Path(args.pilot_manifest)
    manifest = load_pilot_manifest(manifest_path)
    required = current_required_agreement_reports(manifest, manifest_path)
    if len(required) != manifest.first_agreement_review_count:
        raise ValueError(
            "first agreement review is not yet triggered: exact current report set unavailable"
        )
    for item in required:
        print(
            f"required video={item.video_id} agreement_id={item.agreement_id} "
            f"content_sha256={item.agreement_content_sha256}"
        )
    _print_disagreement_summary(
        _resolve(
            manifest_path.resolve().parent, manifest.artifacts.agreements_directory
        ),
        {item.video_id for item in required},
    )
    summary = _load_summary(args.summary)
    document = build_first_agreement_review_document(
        manifest.pilot_id, required, summary
    )
    status = assess_first_agreement_review(manifest.pilot_id, required, document)
    if not status.complete:
        raise ValueError("initial agreement review is not complete and current")
    output = _resolve(
        manifest_path.resolve().parent,
        args.output or manifest.artifacts.first_agreement_review,
    )
    save_first_agreement_review(document, output)
    print(f"First agreement review: {output}")
    print(f"First agreement review SHA-256: {document.content_sha256}")
    return 0


def _load_summary(path: str) -> FirstAgreementReviewSummary:
    with Path(path).open("r", encoding="utf-8") as stream:
        value: Any = json.load(stream)
    return FirstAgreementReviewSummary.model_validate(value)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _print_disagreement_summary(directory: Path, required_video_ids: set[str]) -> None:
    categories: Counter[str] = Counter()
    report_count = 0
    disagreement_count = 0
    for path in sorted(directory.rglob("*.json")):
        report = load_agreement(path)
        if report.video_id not in required_video_ids:
            continue
        report_count += 1
        disagreement_count += report.disagreement_count
        categories.update(
            item.value
            for disagreement in report.disagreements
            for item in disagreement.disagreement_types
        )
    print(f"Agreement reports summarized: {report_count}")
    print(f"Disagreements summarized: {disagreement_count}")
    for category, count in sorted(categories.items()):
        print(f"disagreement_category {category}={count}")


if __name__ == "__main__":
    raise SystemExit(main())

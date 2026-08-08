"""Record a reasoned scale-up decision bound to current pilot evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.benchmark.fingerprints import streaming_file_sha256
from app.dataset.pilot import (
    current_required_agreement_reports,
    load_pilot_manifest,
)
from app.dataset.pilot_review import (
    ScaleUpDecision,
    assess_failure_review,
    assess_first_agreement_review,
    assess_scale_up_decision,
    build_scale_up_decision_document,
    derive_required_failures,
    load_baseline_review_identity,
    load_optional_failure_review,
    load_optional_first_agreement_review,
    save_scale_up_decision,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-manifest",
        default="data/benchmark/pilot/mini_pilot_manifest.json",
    )
    parser.add_argument("--baseline")
    parser.add_argument("--failure-review")
    parser.add_argument("--agreement-review")
    parser.add_argument(
        "--decision", choices=[item.value for item in ScaleUpDecision], required=True
    )
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--known-blocker", action="append", default=[])
    parser.add_argument("--known-limitation", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    manifest_path = Path(args.pilot_manifest)
    manifest = load_pilot_manifest(manifest_path)
    base = manifest_path.resolve().parent
    baseline = _resolve(base, args.baseline or manifest.artifacts.baseline_directory)
    identity, report = load_baseline_review_identity(baseline)
    if identity.pilot_id != manifest.pilot_id:
        raise ValueError("baseline pilot_id differs from pilot manifest")
    failure_path = _resolve(
        base, args.failure_review or manifest.artifacts.failure_review
    )
    agreement_path = _resolve(
        base, args.agreement_review or manifest.artifacts.first_agreement_review
    )
    failure_review = load_optional_failure_review(failure_path)
    agreement_review = load_optional_first_agreement_review(agreement_path)
    if agreement_review is None:
        raise ValueError("current first-agreement review artifact is required")
    failure_coverage = assess_failure_review(
        identity, derive_required_failures(identity, report), failure_review
    )
    required_agreements = current_required_agreement_reports(manifest, manifest_path)
    agreement_status = assess_first_agreement_review(
        manifest.pilot_id, required_agreements, agreement_review
    )
    if not failure_coverage.complete or not agreement_status.complete:
        raise ValueError("review prerequisites are incomplete or stale")
    release_path = _resolve(base, manifest.artifacts.dataset_release)
    if not release_path.is_file():
        raise ValueError("current dataset release is missing")
    current_release_hash = streaming_file_sha256(release_path)
    if current_release_hash != identity.dataset_release_sha256:
        raise ValueError("current dataset release differs from frozen baseline release")
    document = build_scale_up_decision_document(
        identity,
        failure_review,
        agreement_review,
        failure_coverage,
        agreement_status,
        decision=ScaleUpDecision(args.decision),
        rationale=args.rationale,
        conditions=args.condition,
        known_blockers=args.known_blocker,
        known_limitations=args.known_limitation,
    )
    validation = assess_scale_up_decision(
        identity,
        failure_coverage,
        agreement_status,
        failure_review,
        agreement_review,
        document,
        current_dataset_release_sha256=current_release_hash,
    )
    if not validation.valid:
        raise ValueError("scale-up decision does not bind to current evidence")
    output = _resolve(base, args.output or manifest.artifacts.scale_up_decision)
    save_scale_up_decision(document, output)
    print(f"Decision: {document.decision.value}")
    print(f"Scale-up decision: {output}")
    print(f"Scale-up decision SHA-256: {document.content_sha256}")
    print(
        f"Failure review coverage: {failure_coverage.reviewed_count}/"
        f"{failure_coverage.required_count}"
    )
    print(f"Agreement review complete: {agreement_status.complete}")
    print(f"Known limitations: {len(document.known_limitations)}")
    print(f"Conditions: {len(document.conditions)}")
    return 0


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a versioned release manifest and evaluate annotation quality gates."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.io import (
    load_adjudication,
    load_annotation,
    load_registry,
    read_json_model,
    write_json_model,
)
from app.dataset.models import AnnotationQualityConfig, SplitAssignmentDocument
from app.dataset.release import build_dataset_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="data/benchmark/intake_registry.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--adjudications-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-label-agreement", type=float)
    parser.add_argument("--minimum-event-match-rate", type=float)
    args = parser.parse_args(argv)
    annotations: dict[str, list] = {}
    for path in sorted(Path(args.annotations_dir).rglob("*.json")):
        document = load_annotation(path)
        annotations.setdefault(document.video_id, []).append(document)
    adjudications = {}
    for path in sorted(Path(args.adjudications_dir).rglob("*.json")):
        artifact = load_adjudication(path)
        adjudications[artifact.video_id] = artifact
    release = build_dataset_release(
        load_registry(args.registry),
        read_json_model(args.splits, SplitAssignmentDocument),
        annotations,
        adjudications,
        quality_config=AnnotationQualityConfig(
            minimum_label_agreement=args.minimum_label_agreement,
            minimum_event_match_rate=args.minimum_event_match_rate,
        ),
    )
    write_json_model(release, args.output)
    for gate in release.quality_gates:
        print(f"{'PASS' if gate.passed else 'FAIL'} {gate.gate}: {gate.details}")
    return 0 if release.quality_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

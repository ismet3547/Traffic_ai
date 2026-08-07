"""Export approved adjudication into the existing benchmark annotation schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.io import (
    document_sha256,
    load_adjudication,
    load_annotation,
    load_registry,
    read_json_model,
    write_json_model,
)
from app.dataset.models import DatasetRelease, SplitAssignmentDocument
from app.dataset.release import export_adjudicated_annotation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--registry", default="data/benchmark/intake_registry.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Current locked source annotations used to detect stale adjudication.",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Validated release manifest containing the expected ground-truth hash.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    artifact = load_adjudication(args.adjudication)
    registry = load_registry(args.registry)
    splits = read_json_model(args.splits, SplitAssignmentDocument)
    release = read_json_model(args.release, DatasetRelease)
    record = next(
        (item for item in registry.videos if item.video_id == artifact.video_id), None
    )
    assignments = [
        item for item in splits.assignments if item.video_id == artifact.video_id
    ]
    release_entry = next(
        (item for item in release.videos if item.video_id == artifact.video_id), None
    )
    if record is None or len(assignments) != 1 or release_entry is None:
        raise ValueError(
            "adjudicated video must have one registry, split, and release entry"
        )
    assignment = assignments[0]
    if (
        not release.integrity_report.passed
        or assignment.source_group_id != record.source_group_id
        or release_entry.source_group_id != record.source_group_id
        or release_entry.split != assignment.split
        or release_entry.source_video_sha256 != record.source_video_sha256
    ):
        raise ValueError("release, split, and registry provenance do not agree")
    if release_entry.benchmark_ground_truth_sha256 is None:
        raise ValueError("release entry has no adjudicated ground-truth hash")
    source_annotations = []
    for path in sorted(Path(args.annotations_dir).rglob("*.json")):
        annotation = load_annotation(path)
        if annotation.video_id == artifact.video_id:
            source_annotations.append(annotation)
    document = export_adjudicated_annotation(
        artifact,
        record,
        split=assignment.split,
        source_annotations=source_annotations,
        expected_ground_truth_sha256=release_entry.benchmark_ground_truth_sha256,
    )
    current_annotation_hashes = {
        item.annotator_id: document_sha256(item) for item in source_annotations
    }
    if (
        current_annotation_hashes != release_entry.annotation_hashes
        or document_sha256(artifact) != release_entry.adjudicated_annotation_hash
    ):
        raise ValueError(
            "current annotation/adjudication revisions differ from release manifest"
        )
    write_json_model(document, args.output)
    print(f"Exported {len(document.events)} adjudicated benchmark event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

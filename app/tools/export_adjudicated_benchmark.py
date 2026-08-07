"""Export approved adjudication into the existing benchmark annotation schema."""

from __future__ import annotations

import argparse

from app.dataset.io import (
    load_adjudication,
    load_registry,
    read_json_model,
    write_json_model,
)
from app.dataset.models import SplitAssignmentDocument
from app.dataset.release import export_adjudicated_annotation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--registry", default="data/benchmark/intake_registry.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    artifact = load_adjudication(args.adjudication)
    registry = load_registry(args.registry)
    splits = read_json_model(args.splits, SplitAssignmentDocument)
    record = next(
        (item for item in registry.videos if item.video_id == artifact.video_id), None
    )
    assignment = next(
        (item for item in splits.assignments if item.video_id == artifact.video_id),
        None,
    )
    if record is None or assignment is None:
        raise ValueError(
            "adjudicated video is absent from registry or split assignments"
        )
    document = export_adjudicated_annotation(artifact, record, split=assignment.split)
    write_json_model(document, args.output)
    print(f"Exported {len(document.events)} adjudicated benchmark event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

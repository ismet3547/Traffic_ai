"""Compare two independent human annotation passes."""

from __future__ import annotations

import argparse

from app.dataset.agreement import compare_independent_annotations
from app.dataset.io import load_annotation, write_json_model
from app.dataset.models import AgreementConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_a")
    parser.add_argument("annotation_b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-temporal-iou", type=float, default=0.30)
    parser.add_argument("--boundary-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--ignore-vehicle-reference", action="store_true")
    args = parser.parse_args(argv)
    report = compare_independent_annotations(
        load_annotation(args.annotation_a),
        load_annotation(args.annotation_b),
        AgreementConfig(
            minimum_temporal_iou=args.minimum_temporal_iou,
            boundary_tolerance_seconds=args.boundary_tolerance_seconds,
            require_vehicle_reference_match=not args.ignore_vehicle_reference,
        ),
    )
    write_json_model(report, args.output)
    print(
        f"matched={report.matched_event_count} event_agreement={report.event_detection_agreement:.3f} "
        f"label_agreement={report.label_agreement:.3f} disagreements={report.disagreement_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

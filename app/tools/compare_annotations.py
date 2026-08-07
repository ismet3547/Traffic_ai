"""Compare two independent human annotation passes."""

from __future__ import annotations

import argparse

from app.dataset.agreement import compare_independent_annotations
from app.dataset.io import load_annotation, write_json_model
from app.dataset.models import (
    CANONICAL_AGREEMENT_CONFIG,
    AgreementConfig,
    AgreementMode,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_a")
    parser.add_argument("annotation_b")
    parser.add_argument("--output", required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--official",
        action="store_true",
        help="use the immutable canonical agreement protocol (default)",
    )
    mode_group.add_argument(
        "--exploratory",
        action="store_true",
        help="mark the report as research-only and ineligible for official release",
    )
    parser.add_argument("--minimum-temporal-iou", type=float)
    parser.add_argument("--boundary-tolerance-seconds", type=float)
    parser.add_argument("--ignore-vehicle-reference", action="store_true")
    args = parser.parse_args(argv)
    custom_requested = any(
        (
            args.minimum_temporal_iou is not None,
            args.boundary_tolerance_seconds is not None,
            args.ignore_vehicle_reference,
        )
    )
    if args.official and custom_requested:
        parser.error("--official does not permit agreement config overrides")
    mode = (
        AgreementMode.EXPLORATORY
        if args.exploratory or custom_requested
        else AgreementMode.OFFICIAL
    )
    config = (
        AgreementConfig(
            minimum_temporal_iou=(
                args.minimum_temporal_iou
                if args.minimum_temporal_iou is not None
                else CANONICAL_AGREEMENT_CONFIG.minimum_temporal_iou
            ),
            boundary_tolerance_seconds=(
                args.boundary_tolerance_seconds
                if args.boundary_tolerance_seconds is not None
                else CANONICAL_AGREEMENT_CONFIG.boundary_tolerance_seconds
            ),
            require_vehicle_reference_match=not args.ignore_vehicle_reference,
        )
        if mode == AgreementMode.EXPLORATORY
        else CANONICAL_AGREEMENT_CONFIG
    )
    report = compare_independent_annotations(
        load_annotation(args.annotation_a),
        load_annotation(args.annotation_b),
        config,
        mode=mode,
    )
    write_json_model(report, args.output)
    print(
        f"video_id={report.video_id} source_sha={report.source_video_sha256[:12]} "
        f"annotation_a_sha={report.annotation_a_content_sha256[:12]} "
        f"annotation_b_sha={report.annotation_b_content_sha256[:12]} "
        f"protocol={report.agreement_protocol_version} "
        f"mode={report.agreement_mode.value} "
        f"config_version={report.agreement_config_version} "
        f"config_sha={report.agreement_config_fingerprint[:12]} "
        f"agreement_id={report.agreement_id[:12]} "
        f"matched={report.matched_event_count} "
        f"event_agreement={report.event_detection_agreement:.3f} "
        f"label_agreement={report.label_agreement:.3f} "
        f"disagreements={report.disagreement_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

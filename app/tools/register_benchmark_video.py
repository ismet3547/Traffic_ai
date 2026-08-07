"""Register a permitted source video without copying it into the repository."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.dataset.intake import register_video
from app.dataset.io import load_registry, write_json_model
from app.dataset.models import (
    IntakeRegistry,
    PermissionStatus,
    SourceType,
    VehicleClass,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--registry", default="data/benchmark/intake_registry.json")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--source-group-id", required=True)
    parser.add_argument(
        "--source-type", required=True, choices=[item.value for item in SourceType]
    )
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--acquisition-date", required=True)
    parser.add_argument(
        "--permission-status",
        required=True,
        choices=[item.value for item in PermissionStatus],
    )
    parser.add_argument("--redistribution-allowed", action="store_true")
    parser.add_argument("--benchmark-use-allowed", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--scenario-tag", action="append", default=[])
    parser.add_argument(
        "--vehicle-class",
        action="append",
        default=[],
        choices=[item.value for item in VehicleClass],
    )
    parser.add_argument("--allow-duplicate-content", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry_path = Path(args.registry)
    registry = (
        load_registry(registry_path) if registry_path.is_file() else IntakeRegistry()
    )
    updated, record, warnings = register_video(
        registry,
        args.video,
        video_id=args.video_id,
        source_group_id=args.source_group_id,
        source_type=SourceType(args.source_type),
        source_reference=args.source_reference,
        acquisition_date=date.fromisoformat(args.acquisition_date),
        permission_status=PermissionStatus(args.permission_status),
        redistribution_allowed=args.redistribution_allowed,
        benchmark_use_allowed=args.benchmark_use_allowed,
        notes=args.notes,
        scenario_tags=args.scenario_tag,
        vehicle_classes=[VehicleClass(item) for item in args.vehicle_class],
        allow_duplicate_content=args.allow_duplicate_content,
        replace_existing=args.replace_existing,
    )
    write_json_model(updated, registry_path)
    print(f"Registered {record.video_id}: sha256={record.source_video_sha256}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

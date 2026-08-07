"""List the most actionable FP/FN records from a benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect benchmark failure records.")
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--kind", choices=["false_positive", "false_negative", "all"], default="all"
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    with Path(args.report).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    failures = report.get("failures", [])
    if args.kind != "all":
        failures = [item for item in failures if item.get("kind") == args.kind]
    if args.category:
        failures = [
            item
            for item in failures
            if item.get("suspected_failure_category") == args.category
        ]
    failures.sort(
        key=lambda item: (
            -float((item.get("prediction") or {}).get("confidence", 0.0)),
            item.get("video_id", ""),
            item.get("failure_id", ""),
        )
    )
    for item in failures[: args.limit]:
        print(
            f"{item['failure_id']} {item['video_id']} {item['kind']} "
            f"{item['suspected_failure_category']} @ {item['timestamp_seconds']:.3f}s"
        )
        for reason in item.get("diagnostic_rationale", []):
            print(f"  - {reason}")
        if item.get("artifact_directory"):
            print(f"  artifacts: {item['artifact_directory']}")
    print(f"Displayed {min(len(failures), args.limit)} of {len(failures)} failure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

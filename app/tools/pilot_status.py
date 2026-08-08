"""Report truthful progress and blockers for the real mini pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.io import write_json_model
from app.dataset.pilot import (
    build_pilot_status,
    load_pilot_manifest,
    render_pilot_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/benchmark/pilot/mini_pilot_manifest.json",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit code 2 while blockers remain",
    )
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    status = build_pilot_status(load_pilot_manifest(manifest_path), manifest_path)
    if args.output_json:
        write_json_model(status, args.output_json)
    markdown = render_pilot_status(status)
    if args.output_markdown:
        destination = Path(args.output_markdown)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    print(status.real_pilot_status)
    print(f"pilot_state={status.pilot_state.value}")
    for field, value in status.counts.model_dump(mode="json").items():
        print(f"{field}={value}")
    print(
        "first_agreement_review="
        f"required:{status.first_agreement_review.required},"
        f"complete:{status.first_agreement_review.complete},"
        f"stale:{status.first_agreement_review.stale},"
        f"required_reports:{status.first_agreement_review.required_report_count}"
    )
    print(
        "failure_review="
        f"required:{status.failure_review.required_count},"
        f"reviewed:{status.failure_review.reviewed_count},"
        f"missing:{status.failure_review.missing_count},"
        f"duplicate:{status.failure_review.duplicate_count},"
        f"unknown:{status.failure_review.unknown_count},"
        f"stale:{status.failure_review.stale_count},"
        f"complete:{status.failure_review.complete}"
    )
    print(
        "scale_up_decision="
        f"present:{status.scale_up_decision.present},"
        f"valid:{status.scale_up_decision.valid},"
        f"stale:{status.scale_up_decision.stale},"
        f"decision:{status.scale_up_recommendation}"
    )
    for blocker in status.blockers:
        scope = f" video_id={blocker.video_id}" if blocker.video_id else ""
        print(f"BLOCKER {blocker.code}{scope}: {blocker.details}")
    for warning in status.warnings:
        scope = f" video_id={warning.video_id}" if warning.video_id else ""
        print(f"WARNING {warning.code}{scope}: {warning.details}")
    for notice in status.information:
        scope = f" video_id={notice.video_id}" if notice.video_id else ""
        print(f"INFO {notice.code}{scope}: {notice.details}")
    return 2 if args.require_ready and status.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

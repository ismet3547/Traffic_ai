"""Generate honest JSON and Markdown coverage reports from supplied artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.dataset.io import load_agreement, load_annotation, load_registry
from app.dataset.reporting import build_coverage_report, write_coverage_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="data/benchmark/intake_registry.json")
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--agreements-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    annotations: dict[str, list] = {}
    for path in sorted(Path(args.annotations_dir).rglob("*.json")):
        document = load_annotation(path)
        annotations.setdefault(document.video_id, []).append(document)
    agreements = []
    if args.agreements_dir:
        agreements = [
            load_agreement(path)
            for path in sorted(Path(args.agreements_dir).rglob("*.json"))
        ]
    report = build_coverage_report(
        load_registry(args.registry), annotations, agreements
    )
    json_path, markdown_path = write_coverage_report(report, args.output_dir)
    print(f"Wrote {json_path} and {markdown_path}; clips={report.total_clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

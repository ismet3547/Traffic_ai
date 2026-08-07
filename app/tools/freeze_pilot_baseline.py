"""Freeze a completed clean real-pilot benchmark run as pilot_baseline_0."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.dataset.pilot import freeze_pilot_baseline, load_pilot_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/benchmark/pilot/mini_pilot_manifest.json",
    )
    parser.add_argument(
        "--frozen-at",
        help="optional reproducible ISO-8601 timestamp; defaults to current UTC time",
    )
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    timestamp = datetime.fromisoformat(args.frozen_at) if args.frozen_at else None
    destination, metadata = freeze_pilot_baseline(
        load_pilot_manifest(manifest_path),
        manifest_path,
        frozen_at=timestamp,
    )
    print(f"Frozen {metadata.baseline_id}: {destination}")
    print(f"pilot_id={metadata.pilot_id}")
    print(f"git_commit={metadata.system_git_commit}")
    print(f"production_config_sha256={metadata.production_config_hash_sha256}")
    print(f"dataset_fingerprint={metadata.dataset_fingerprint}")
    print(metadata.accuracy_warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

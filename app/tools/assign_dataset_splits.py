"""Assign source-group-isolated development, validation, and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from app.benchmark.models import DatasetSplit
from app.dataset.io import write_json_model
from app.dataset.models import SplitCandidate
from app.dataset.splitting import assign_group_aware_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", required=True, help="JSON list of split candidates"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development", type=float, default=0.50)
    parser.add_argument("--validation", type=float, default=0.25)
    parser.add_argument("--test", type=float, default=0.25)
    args = parser.parse_args(argv)
    candidates = TypeAdapter(list[SplitCandidate]).validate_python(
        json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    )
    result = assign_group_aware_splits(
        candidates,
        target_ratios={
            DatasetSplit.DEVELOPMENT: args.development,
            DatasetSplit.VALIDATION: args.validation,
            DatasetSplit.TEST: args.test,
        },
        seed=args.seed,
    )
    write_json_model(result, args.output)
    print(f"Assigned {len(result.assignments)} clip(s) with source-group isolation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

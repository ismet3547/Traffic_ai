"""Create an auditable adjudication artifact from two locked human passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from app.dataset.adjudication import create_adjudication, lock_adjudication
from app.dataset.io import load_annotation, write_json_model
from app.dataset.models import AdjudicationDecision, AgreementConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_a")
    parser.add_argument("annotation_b")
    parser.add_argument("--adjudicator-id", required=True)
    parser.add_argument(
        "--decisions", required=True, help="JSON list of explicit decisions"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-temporal-iou", type=float, default=0.30)
    parser.add_argument("--boundary-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    decisions = TypeAdapter(list[AdjudicationDecision]).validate_python(payload)
    annotation_a = load_annotation(args.annotation_a)
    annotation_b = load_annotation(args.annotation_b)
    artifact = create_adjudication(
        annotation_a,
        annotation_b,
        adjudicator_id=args.adjudicator_id,
        decisions=decisions,
        agreement_config=AgreementConfig(
            minimum_temporal_iou=args.minimum_temporal_iou,
            boundary_tolerance_seconds=args.boundary_tolerance_seconds,
        ),
    )
    if args.lock:
        artifact = lock_adjudication(artifact)
    for disagreement in artifact.agreement_report.disagreements:
        event_a = next(
            (
                item
                for item in annotation_a.events
                if item.event_id == disagreement.event_id_a
            ),
            None,
        )
        event_b = next(
            (
                item
                for item in annotation_b.events
                if item.event_id == disagreement.event_id_b
            ),
            None,
        )
        print(
            f"{disagreement.disagreement_id}: "
            f"types={[item.value for item in disagreement.disagreement_types]} "
            f"A={event_a.model_dump(mode='json') if event_a else None} "
            f"B={event_b.model_dump(mode='json') if event_b else None}"
        )
    write_json_model(artifact, args.output)
    print(
        f"Adjudicated {len(artifact.final_events)} event(s); locked={artifact.locked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

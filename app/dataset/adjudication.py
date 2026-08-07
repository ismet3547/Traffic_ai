"""Explicit adjudication that preserves both blinded annotation originals."""

from __future__ import annotations

from datetime import datetime, timezone

from app.dataset.agreement import compare_independent_annotations
from app.dataset.io import (
    adjudication_content_hash,
    annotation_content_hash,
    document_sha256,
)
from app.dataset.models import (
    AdjudicationArtifact,
    AdjudicationDecision,
    AdjudicationOutcome,
    AgreementConfig,
    AgreementMode,
    DatasetAnnotation,
)


def create_adjudication(
    annotation_a: DatasetAnnotation,
    annotation_b: DatasetAnnotation,
    *,
    adjudicator_id: str,
    decisions: list[AdjudicationDecision],
    agreement_config: AgreementConfig | None = None,
    agreement_mode: AgreementMode | None = None,
    created_at: datetime | None = None,
    approved: bool = True,
) -> AdjudicationArtifact:
    if not annotation_a.locked or not annotation_b.locked:
        raise ValueError("independent annotations must be locked before adjudication")
    for annotation in (annotation_a, annotation_b):
        if annotation.annotation_hash != annotation_content_hash(annotation):
            raise ValueError("locked annotation content hash is invalid")
    report = compare_independent_annotations(
        annotation_a,
        annotation_b,
        agreement_config,
        mode=agreement_mode,
    )
    event_a = {event.event_id: event for event in annotation_a.events}
    auto_decisions: list[AdjudicationDecision] = []
    final_events = []
    disagreement_pairs = {
        (item.event_id_a, item.event_id_b)
        for item in report.disagreements
        if item.event_id_a is not None and item.event_id_b is not None
    }
    for match in report.matches:
        if (match.event_id_a, match.event_id_b) in disagreement_pairs:
            continue
        selected = event_a[match.event_id_a]
        auto_decisions.append(
            AdjudicationDecision(
                decision_id=f"agree_{match.event_id_a}_{match.event_id_b}",
                event_ids_a=[match.event_id_a],
                event_ids_b=[match.event_id_b],
                outcome=AdjudicationOutcome.AGREE,
                adjudicated_event=selected,
                rationale="independent annotations agree within configured tolerances",
                adjudication_confidence=selected.confidence,
            )
        )
        final_events.append(selected)

    disagreement_ids = {item.disagreement_id for item in report.disagreements}
    covered = [
        identifier for decision in decisions for identifier in decision.disagreement_ids
    ]
    if set(covered) != disagreement_ids or len(covered) != len(set(covered)):
        missing = sorted(disagreement_ids - set(covered))
        unknown = sorted(set(covered) - disagreement_ids)
        raise ValueError(
            "adjudication decisions must cover every disagreement exactly once; "
            f"missing={missing}, unknown={unknown}"
        )
    final_events.extend(
        decision.adjudicated_event
        for decision in decisions
        if decision.adjudicated_event is not None
    )
    return AdjudicationArtifact(
        video_id=annotation_a.video_id,
        source_video_sha256=annotation_a.source_video_sha256,
        source_video_size_bytes=(
            annotation_a.source_video_size_bytes
            if annotation_a.source_video_size_bytes is not None
            else annotation_b.source_video_size_bytes
        ),
        annotation_a=annotation_a,
        annotation_b=annotation_b,
        annotation_a_hash=document_sha256(annotation_a),
        annotation_b_hash=document_sha256(annotation_b),
        agreement_report=report,
        adjudicator_id=adjudicator_id,
        created_at=created_at or datetime.now(timezone.utc),
        decisions=[*auto_decisions, *decisions],
        final_events=sorted(
            final_events, key=lambda item: (item.start_seconds, item.event_id)
        ),
        approved=approved,
    )


def lock_adjudication(
    artifact: AdjudicationArtifact,
    *,
    locked_at: datetime | None = None,
) -> AdjudicationArtifact:
    if not artifact.approved:
        raise ValueError("only approved adjudication can be locked")
    timestamp = locked_at or datetime.now(timezone.utc)
    candidate = artifact.model_copy(
        update={"locked": True, "locked_at": timestamp, "adjudication_hash": None}
    )
    digest = adjudication_content_hash(candidate)
    return candidate.model_copy(update={"adjudication_hash": digest})

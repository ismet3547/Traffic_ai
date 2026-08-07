from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.benchmark.models import AnnotationConfidence
from app.dataset.adjudication import create_adjudication
from app.dataset.agreement import compare_independent_annotations
from app.dataset.io import (
    load_annotation,
    lock_annotation,
    save_annotation,
    validate_annotation_protocol,
)
from app.dataset.models import (
    AdjudicationDecision,
    AdjudicationOutcome,
    AgreementConfig,
    DatasetAnnotation,
    DatasetEvent,
    DatasetLabel,
    DisagreementType,
    EventEvidence,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SHA = "a" * 64


def event(
    event_id: str,
    label: DatasetLabel = DatasetLabel.UNNECESSARY_LEFT_LANE_OCCUPATION,
    start: float = 10,
    end: float = 20,
    *,
    confidence: AnnotationConfidence = AnnotationConfidence.HIGH,
    vehicle_ref: str = "vehicle_01",
) -> DatasetEvent:
    return DatasetEvent(
        event_id=event_id,
        vehicle_ref=vehicle_ref,
        start_seconds=start,
        end_seconds=end,
        label=label,
        confidence=confidence,
        evidence=EventEvidence(right_lane_available=True, congestion_present=False),
    )


def annotation(annotator: str, events: list[DatasetEvent]) -> DatasetAnnotation:
    return DatasetAnnotation(
        video_id="clip_01",
        source_video_sha256=SHA,
        source_file="clip.mp4",
        fps=30,
        video_duration_seconds=60,
        annotator_id=annotator,
        created_at=NOW,
        events=events,
    )


def disagreement_types(report) -> set[DisagreementType]:
    return {kind for item in report.disagreements for kind in item.disagreement_types}


def test_annotation_schema_accepts_evidence_and_preserves_annotator() -> None:
    document = annotation("anonymous_a", [event("e1")])
    assert document.annotator_id == "anonymous_a"
    assert document.events[0].evidence.right_lane_available is True


def test_annotation_schema_rejects_invalid_vehicle_and_time() -> None:
    with pytest.raises(ValidationError):
        event("bad", vehicle_ref="ABC-PLATE")
    with pytest.raises(ValidationError):
        event("bad", start=10, end=10)


def test_raw_schema_rejects_prediction_leakage_fields() -> None:
    payload = annotation("a", []).model_dump(mode="json")
    payload["system_prediction"] = "candidate"
    with pytest.raises(ValidationError):
        DatasetAnnotation.model_validate(payload)


def test_two_distinct_independent_annotators_agree() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("a1")]), annotation("b", [event("b1")])
    )
    assert report.event_detection_agreement == 1
    assert report.label_agreement == 1
    assert report.disagreement_count == 0


def test_same_annotator_is_not_independent() -> None:
    with pytest.raises(ValueError, match="distinct"):
        compare_independent_annotations(annotation("a", []), annotation("a", []))


def test_label_and_confidence_disagreement_are_classified() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("a1")]),
        annotation(
            "b",
            [
                event(
                    "b1",
                    DatasetLabel.LEGITIMATE_OVERTAKING,
                    confidence=AnnotationConfidence.MEDIUM,
                )
            ],
        ),
    )
    assert {
        DisagreementType.LABEL_DISAGREEMENT,
        DisagreementType.CONFIDENCE_DISAGREEMENT,
    } <= disagreement_types(report)
    assert report.label_agreement == 0


def test_event_missing_from_b_is_classified() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("a1")]), annotation("b", [])
    )
    assert DisagreementType.EVENT_MISSING_B in disagreement_types(report)
    assert report.event_detection_agreement == 0


def test_boundary_disagreement_uses_tolerance() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("a1", start=10, end=20)]),
        annotation("b", [event("b1", start=12, end=22)]),
        AgreementConfig(boundary_tolerance_seconds=1, minimum_temporal_iou=0.3),
    )
    assert DisagreementType.BOUNDARY_DISAGREEMENT in disagreement_types(report)
    assert report.temporal_boundary_agreement == 0


def test_one_long_against_two_short_is_not_perfect() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("long", start=0, end=20)]),
        annotation(
            "b", [event("short1", start=0, end=9), event("short2", start=11, end=20)]
        ),
    )
    assert report.matched_event_count == 1
    assert report.event_detection_agreement < 1


def test_tiny_overlap_is_not_a_match_at_default_threshold() -> None:
    report = compare_independent_annotations(
        annotation("a", [event("a1", start=0, end=10)]),
        annotation("b", [event("b1", start=9.9, end=20)]),
    )
    assert report.matched_event_count == 0


def test_adjudication_preserves_locked_originals() -> None:
    first = lock_annotation(annotation("a", [event("a1")]), locked_at=NOW)
    second = lock_annotation(annotation("b", [event("b1")]), locked_at=NOW)
    artifact = create_adjudication(
        first, second, adjudicator_id="reviewer", decisions=[], created_at=NOW
    )
    assert artifact.annotation_a == first
    assert artifact.annotation_b == second
    assert artifact.final_events[0].event_id == "a1"


def test_unresolved_ambiguity_can_remain_insufficient_evidence() -> None:
    first = lock_annotation(annotation("a", [event("a1")]), locked_at=NOW)
    second = lock_annotation(annotation("b", []), locked_at=NOW)
    report = compare_independent_annotations(first, second)
    ambiguous = event("final1", DatasetLabel.INSUFFICIENT_EVIDENCE)
    decision = AdjudicationDecision(
        decision_id="d1",
        disagreement_ids=[report.disagreements[0].disagreement_id],
        event_ids_a=["a1"],
        outcome=AdjudicationOutcome.REMAINS_AMBIGUOUS,
        adjudicated_event=ambiguous,
        rationale="Visual context remains unresolved.",
        adjudication_confidence=AnnotationConfidence.LOW,
    )
    artifact = create_adjudication(
        first, second, adjudicator_id="reviewer", decisions=[decision], created_at=NOW
    )
    assert artifact.final_events[0].label == DatasetLabel.INSUFFICIENT_EVIDENCE


def test_locked_annotation_rejects_normal_edit(tmp_path) -> None:
    path = tmp_path / "annotation.json"
    locked = lock_annotation(annotation("a", [event("a1")]), locked_at=NOW)
    save_annotation(locked, path)
    edited = locked.model_copy(update={"events": [event("changed")]})
    with pytest.raises(PermissionError, match="locked"):
        save_annotation(edited, path)


def test_locked_override_is_auditable(tmp_path) -> None:
    path = tmp_path / "annotation.json"
    locked = lock_annotation(annotation("a", [event("a1")]), locked_at=NOW)
    save_annotation(locked, path)
    edited = locked.model_copy(update={"events": [event("changed")]})
    saved = save_annotation(
        edited,
        path,
        override_lock=True,
        override_reason="approved correction",
        timestamp=NOW,
    )
    loaded = load_annotation(path)
    assert saved.lock_override_history[-1].reason == "approved correction"
    assert loaded.annotation_hash is not None


def test_handbook_and_ontology_drift_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="HANDBOOK_VERSION_MISMATCH"):
        validate_annotation_protocol(
            annotation("a", []).model_copy(update={"handbook_version": "0.9"})
        )
    with pytest.raises(ValueError, match="ONTOLOGY_VERSION_MISMATCH"):
        validate_annotation_protocol(
            annotation("a", []).model_copy(update={"ontology_version": "old"})
        )

"""Transparent heuristics for suspected, not asserted, failure causes."""

from __future__ import annotations

from app.benchmark.matcher import temporal_iou
from app.benchmark.models import (
    AnnotationLabel,
    FailureRecord,
    GroundTruthEvent,
    PredictedEvent,
)

FAILURE_CATEGORIES = {
    "DETECTION_FAILURE",
    "TRACKING_ID_SWITCH",
    "LANE_ASSIGNMENT_ERROR",
    "GEOMETRY_INTEGRITY_ERROR",
    "OVERTAKING_LOGIC_ERROR",
    "CONGESTION_LOGIC_ERROR",
    "RIGHT_LANE_OPPORTUNITY_ERROR",
    "CANDIDATE_LIFECYCLE_ERROR",
    "EVENT_MATCHING_ERROR",
    "ANNOTATION_AMBIGUOUS",
    "UNKNOWN",
}


def diagnose_false_positive(
    prediction: PredictedEvent,
    annotations: list[GroundTruthEvent],
    sequence: int,
) -> FailureRecord:
    related = _best_overlapping_annotation(prediction, annotations)
    category = "UNKNOWN"
    rationale: list[str] = []

    hints = set(prediction.diagnostic_hints)
    if "TRACK_ID_SWITCH_SUSPECTED" in hints:
        category = "TRACKING_ID_SWITCH"
        rationale.append(
            "prediction diagnostic hints report a suspected track-ID switch"
        )
    elif prediction.geometry_status in {"degraded", "invalid", "unverified"}:
        category = "GEOMETRY_INTEGRITY_ERROR"
        rationale.append(
            f"candidate was emitted with geometry status {prediction.geometry_status}"
        )
    elif related is not None:
        category, reason = _category_for_annotation(related, prediction)
        rationale.append(reason)
    elif prediction.lifecycle_state not in {None, "finalized"}:
        category = "CANDIDATE_LIFECYCLE_ERROR"
        rationale.append(
            f"pending-review prediction has lifecycle state {prediction.lifecycle_state}"
        )
    else:
        rationale.append("available event metadata does not isolate a likely subsystem")

    return FailureRecord(
        failure_id=f"fp_{sequence:04d}",
        video_id=prediction.video_id,
        kind="false_positive",
        suspected_failure_category=category,
        diagnostic_rationale=rationale,
        timestamp_seconds=prediction.start_seconds,
        ground_truth=(related.model_dump(mode="json") if related else None),
        prediction=prediction.model_dump(mode="json"),
    )


def diagnose_false_negative(
    truth: GroundTruthEvent,
    annotations: list[GroundTruthEvent],
    video_id: str,
    sequence: int,
) -> FailureRecord:
    overlapping_context = [
        event
        for event in annotations
        if event.event_id != truth.event_id and temporal_iou(event, truth) > 0
    ]
    category = "UNKNOWN"
    rationale = ["no review candidate met the configured temporal matching criteria"]
    if truth.confidence.value != "high":
        category = "ANNOTATION_AMBIGUOUS"
        rationale.append(f"ground-truth confidence is {truth.confidence.value}")
    elif any(
        event.label in {AnnotationLabel.GEOMETRY_INVALID, AnnotationLabel.CAMERA_MOTION}
        for event in overlapping_context
    ):
        category = "GEOMETRY_INTEGRITY_ERROR"
        rationale.append("ground truth overlaps a geometry/camera-motion annotation")
    elif any(
        event.label == AnnotationLabel.LANE_ASSIGNMENT_UNCERTAIN
        for event in overlapping_context
    ):
        category = "LANE_ASSIGNMENT_ERROR"
        rationale.append("ground truth overlaps lane-assignment uncertainty")
    return FailureRecord(
        failure_id=f"fn_{sequence:04d}",
        video_id=video_id,
        kind="false_negative",
        suspected_failure_category=category,
        diagnostic_rationale=rationale,
        timestamp_seconds=truth.start_seconds,
        ground_truth=truth.model_dump(mode="json"),
    )


def _best_overlapping_annotation(
    prediction: PredictedEvent, annotations: list[GroundTruthEvent]
) -> GroundTruthEvent | None:
    ranked = sorted(
        (
            (-temporal_iou(prediction, annotation), annotation.event_id, annotation)
            for annotation in annotations
            if temporal_iou(prediction, annotation) > 0
        ),
        key=lambda item: (item[0], item[1]),
    )
    return ranked[0][2] if ranked else None


def _category_for_annotation(
    annotation: GroundTruthEvent, prediction: PredictedEvent
) -> tuple[str, str]:
    label = annotation.label
    if (
        annotation.confidence.value == "low"
        or label == AnnotationLabel.INSUFFICIENT_EVIDENCE
    ):
        return (
            "ANNOTATION_AMBIGUOUS",
            f"prediction overlaps {label.value} annotation ({annotation.confidence.value})",
        )
    if label == AnnotationLabel.LEGITIMATE_OVERTAKING:
        return (
            "OVERTAKING_LOGIC_ERROR",
            (
                "prediction overlaps a legitimate-overtaking annotation; "
                f"runtime overtake status was {prediction.overtaking_status}"
            ),
        )
    if label == AnnotationLabel.CONGESTION_LEFT_LANE_USE:
        return (
            "CONGESTION_LOGIC_ERROR",
            "prediction overlaps a congestion-left-lane-use annotation",
        )
    if label == AnnotationLabel.RIGHT_LANE_UNAVAILABLE:
        return (
            "RIGHT_LANE_OPPORTUNITY_ERROR",
            "prediction overlaps a right-lane-unavailable annotation",
        )
    if label in {AnnotationLabel.GEOMETRY_INVALID, AnnotationLabel.CAMERA_MOTION}:
        return (
            "GEOMETRY_INTEGRITY_ERROR",
            f"prediction overlaps a {label.value} annotation",
        )
    if label == AnnotationLabel.LANE_ASSIGNMENT_UNCERTAIN:
        return (
            "LANE_ASSIGNMENT_ERROR",
            "prediction overlaps a lane-assignment-uncertain annotation",
        )
    if label == AnnotationLabel.TEMPORARY_LEFT_LANE_USE:
        return (
            "CANDIDATE_LIFECYCLE_ERROR",
            "prediction overlaps a temporary-left-lane-use annotation",
        )
    return "UNKNOWN", f"prediction overlaps annotation label {label.value}"

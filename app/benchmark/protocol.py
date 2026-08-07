"""Explicit semantic identity for the benchmark evaluation protocol."""

from __future__ import annotations

from typing import Final

from app.benchmark.models import EvaluationProtocolIdentity

EVALUATION_PROTOCOL_VERSION: Final = "4.1.1"
MATCHER_SEMANTICS_VERSION: Final = (
    "maximum_cardinality_then_maximum_total_temporal_iou_deterministic_v2"
)
METRIC_SEMANTICS_VERSION: Final = (
    "explicit_accounting_validated_duration_zero_denominators_are_zero_v2"
)
ANNOTATION_ONTOLOGY_VERSION: Final = "canonical_annotation_roles_v1"
IGNORE_POLICY_VERSION: Final = "label_aware_prediction_coverage_and_iou_v1"
CONTROL_EVENT_POLICY_VERSION: Final = (
    "separate_prediction_coverage_and_iou_one_to_one_v1"
)


def current_evaluation_protocol() -> EvaluationProtocolIdentity:
    """Return versions for behavior that changes the meaning of benchmark metrics."""

    return EvaluationProtocolIdentity(
        protocol_version=EVALUATION_PROTOCOL_VERSION,
        matcher_semantics_version=MATCHER_SEMANTICS_VERSION,
        metric_semantics_version=METRIC_SEMANTICS_VERSION,
        annotation_ontology_version=ANNOTATION_ONTOLOGY_VERSION,
        ignore_policy_version=IGNORE_POLICY_VERSION,
        control_event_policy_version=CONTROL_EVENT_POLICY_VERSION,
    )

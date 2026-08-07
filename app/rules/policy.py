"""Explainable policy for contextual left-lane review eligibility."""

from __future__ import annotations

from app.config import (
    CalibrationConfig,
    LeftLaneRuleConfig,
    RightLaneOpportunityConfig,
    TrafficContextConfig,
)
from app.models import (
    BehaviorClassification,
    CandidateDecision,
    CongestionLevel,
    GlobalTrafficContext,
    OvertakingAssessment,
    OvertakingStatus,
    ReviewReasonCode,
    SpeedEstimate,
    SuppressionReason,
    VehicleTrafficContext,
)


class ContextualLeftLaneDecisionPolicy:
    """Conservative evidence gate for human-review candidates."""

    def __init__(
        self,
        rule_config: LeftLaneRuleConfig,
        context_config: TrafficContextConfig,
        opportunity_config: RightLaneOpportunityConfig,
        calibration_config: CalibrationConfig | None = None,
    ) -> None:
        self._rule = rule_config
        self._context = context_config
        self._opportunity = opportunity_config
        self._calibration = calibration_config or CalibrationConfig()

    def decide(
        self,
        left_lane_duration_seconds: float,
        mean_detector_confidence: float,
        history_duration_seconds: float,
        traffic: GlobalTrafficContext | None,
        vehicle_context: VehicleTrafficContext | None,
        overtaking: OvertakingAssessment | None,
        speed: SpeedEstimate | None = None,
    ) -> CandidateDecision:
        calibration = traffic.calibration_status if traffic is not None else None
        if (
            self._calibration.suppress_candidates_when_unreliable
            and calibration is not None
            and calibration.mode in {"homography", "homography_fallback"}
            and (
                not calibration.valid
                or calibration.confidence
                < self._calibration.minimum_confidence_for_physical_measurements
            )
        ):
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.CALIBRATION_UNRELIABLE,
                calibration.confidence,
            )
        if (
            traffic is not None
            and traffic.camera_motion is not None
            and traffic.camera_motion.valid
            and traffic.camera_motion.level == "high"
        ):
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.CAMERA_MOTION_HIGH,
                traffic.camera_motion.confidence,
            )
        if speed is not None and speed.speed_mode in {
            "rejected_position_jump",
            "rejected_unreasonable_speed",
        }:
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.UNSTABLE_TRACK,
            )
        if traffic is not None and traffic.congestion_level in {
            CongestionLevel.DENSE,
            CongestionLevel.STOP_AND_GO,
            CongestionLevel.MODERATE,
        }:
            return _suppressed(
                BehaviorClassification.CONGESTION,
                SuppressionReason.CONGESTION,
                traffic.confidence,
            )
        if (
            overtaking is not None
            and overtaking.status == OvertakingStatus.OVERTAKING_CONFIRMED
        ):
            return _suppressed(
                BehaviorClassification.OVERTAKING,
                SuppressionReason.OVERTAKING_CONFIRMED,
                overtaking.confidence,
            )
        if (
            overtaking is not None
            and overtaking.status == OvertakingStatus.LIKELY_OVERTAKING
        ):
            return _suppressed(
                BehaviorClassification.LIKELY_OVERTAKING,
                SuppressionReason.ACTIVE_OVERTAKE,
                overtaking.confidence,
            )
        if left_lane_duration_seconds < self._rule.occupancy_threshold_seconds:
            return _suppressed(
                BehaviorClassification.TEMPORARY_LEFT_LANE_USE,
                SuppressionReason.DURATION_BELOW_THRESHOLD,
            )
        if (
            history_duration_seconds < self._context.minimum_history_seconds
            or traffic is None
            or vehicle_context is None
            or overtaking is None
        ):
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.INSUFFICIENT_CONTEXT,
            )
        if traffic.congestion_level == CongestionLevel.UNKNOWN:
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.INSUFFICIENT_CONTEXT,
            )
        if overtaking.status in {
            OvertakingStatus.NOT_ASSESSED,
            OvertakingStatus.INSUFFICIENT_EVIDENCE,
        }:
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.INSUFFICIENT_CONTEXT,
                overtaking.confidence,
            )
        if (
            vehicle_context.right_lane_available is not True
            or vehicle_context.right_lane_available_seconds
            < self._opportunity.minimum_available_seconds
            or vehicle_context.right_lane_confidence
            < self._opportunity.minimum_confidence
        ):
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.RIGHT_LANE_UNAVAILABLE,
                vehicle_context.right_lane_confidence,
            )

        evidence_confidence = min(
            1.0,
            0.30 * mean_detector_confidence
            + 0.20 * traffic.confidence
            + 0.25 * vehicle_context.right_lane_confidence
            + 0.25 * overtaking.confidence,
        )
        if evidence_confidence < self._rule.minimum_evidence_confidence:
            return _suppressed(
                BehaviorClassification.INSUFFICIENT_EVIDENCE,
                SuppressionReason.LOW_EVIDENCE_CONFIDENCE,
                evidence_confidence,
            )
        return CandidateDecision(
            eligible=True,
            classification=BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION,
            evidence_confidence=evidence_confidence,
            reason_codes=(
                ReviewReasonCode.LEFT_LANE_DURATION_EXCEEDED.value,
                ReviewReasonCode.NO_ACTIVE_OVERTAKE.value,
                ReviewReasonCode.RIGHT_LANE_AVAILABLE.value,
                ReviewReasonCode.FREE_FLOW_TRAFFIC.value,
            ),
        )


def _suppressed(
    classification: BehaviorClassification,
    reason: SuppressionReason,
    confidence: float = 0.0,
) -> CandidateDecision:
    return CandidateDecision(
        eligible=False,
        classification=classification,
        evidence_confidence=max(0.0, min(1.0, confidence)),
        suppression_reason=reason.value,
    )

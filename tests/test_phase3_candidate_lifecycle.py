from __future__ import annotations

from app.candidates import CandidateLifecycleManager
from app.config import (
    CalibrationConfig,
    CandidateLifecycleConfig,
    LeftLaneRuleConfig,
    RightLaneOpportunityConfig,
    TrafficContextConfig,
)
from app.models import (
    BehaviorClassification,
    CalibrationStatus,
    CandidateDecision,
    CandidateLifecycleState,
    CongestionLevel,
    GlobalTrafficContext,
    NeighborVehicles,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    VehicleTrafficContext,
)
from app.rules import ContextualLeftLaneDecisionPolicy


def _eligible() -> CandidateDecision:
    return CandidateDecision(
        eligible=True,
        classification=BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION,
        evidence_confidence=0.9,
        reason_codes=(
            "LEFT_LANE_DURATION_EXCEEDED",
            "RIGHT_LANE_AVAILABLE",
        ),
    )


def _suppressed(reason: str) -> CandidateDecision:
    return CandidateDecision(
        eligible=False,
        classification=BehaviorClassification.INSUFFICIENT_EVIDENCE,
        evidence_confidence=0.7,
        suppression_reason=reason,
    )


def _manager(**updates: float) -> CandidateLifecycleManager:
    values = {
        "invalidation_grace_seconds": 2.0,
        "suspension_grace_seconds": 3.0,
        "finalize_after_seconds": 5.0,
        "restart_cooldown_seconds": 1.0,
    }
    values.update(updates)
    return CandidateLifecycleManager(CandidateLifecycleConfig(**values))


def test_candidate_starts_when_policy_becomes_eligible() -> None:
    manager = _manager()
    accumulating = manager.update(1, 0.0, _suppressed("DURATION_BELOW_THRESHOLD"))
    started = manager.update(1, 2.0, _eligible())
    assert accumulating.state == CandidateLifecycleState.ACCUMULATING
    assert started.transition == "started"
    assert started.state == CandidateLifecycleState.CANDIDATE_ACTIVE
    assert started.candidate_started_at == 2.0


def test_candidate_suspends_on_temporary_invalid_context() -> None:
    manager = _manager()
    manager.update(1, 0.0, _eligible())
    suspended = manager.update(1, 1.0, _suppressed("RIGHT_LANE_UNAVAILABLE"))
    assert suspended.transition == "suspended"
    assert suspended.state == CandidateLifecycleState.SUSPENDED
    assert suspended.suspended_at == 1.0


def test_candidate_resumes_within_configured_grace() -> None:
    manager = _manager(suspension_grace_seconds=3.0)
    manager.update(1, 0.0, _eligible())
    manager.update(1, 1.0, _suppressed("RIGHT_LANE_UNAVAILABLE"))
    resumed = manager.update(1, 2.0, _eligible())
    assert resumed.transition == "resumed"
    assert resumed.state == CandidateLifecycleState.CANDIDATE_ACTIVE
    assert resumed.suspended_at == 1.0


def test_candidate_cancels_after_persistent_congestion() -> None:
    manager = _manager(invalidation_grace_seconds=2.0)
    manager.update(1, 0.0, _eligible())
    manager.update(1, 1.0, _suppressed("CONGESTION"))
    cancelled = manager.update(1, 3.0, _suppressed("CONGESTION"))
    assert cancelled.transition == "cancelled"
    assert cancelled.state == CandidateLifecycleState.CANCELLED
    assert cancelled.cancellation_reason == "CONGESTION"


def test_candidate_cancels_when_overtaking_becomes_confirmed() -> None:
    manager = _manager(invalidation_grace_seconds=0.5)
    manager.update(1, 0.0, _eligible())
    manager.update(1, 1.0, _suppressed("OVERTAKING_CONFIRMED"))
    cancelled = manager.update(1, 1.5, _suppressed("OVERTAKING_CONFIRMED"))
    assert cancelled.transition == "cancelled"
    assert cancelled.cancellation_reason == "OVERTAKING_CONFIRMED"


def test_finalized_candidate_is_terminal() -> None:
    manager = _manager(finalize_after_seconds=0.5)
    manager.update(1, 0.0, _eligible())
    finalized = manager.update(1, 0.5, _eligible())
    later = manager.update(1, 1.0, _suppressed("CONGESTION"))
    assert finalized.transition == "finalized"
    assert later.transition is None
    assert later.state == CandidateLifecycleState.FINALIZED
    assert later.cancelled_at is None


def test_cancelled_episode_can_restart_after_cooldown() -> None:
    manager = _manager(
        invalidation_grace_seconds=0.5,
        restart_cooldown_seconds=0.5,
    )
    manager.update(1, 0.0, _eligible())
    manager.update(1, 0.5, _suppressed("OVERTAKING_CONFIRMED"))
    manager.update(1, 1.0, _suppressed("OVERTAKING_CONFIRMED"))
    restarted = manager.update(1, 1.6, _eligible())
    assert restarted.transition == "started"
    assert restarted.state == CandidateLifecycleState.CANDIDATE_ACTIVE
    assert restarted.candidate_started_at == 1.6


def test_low_calibration_confidence_degrades_when_configured() -> None:
    policy = ContextualLeftLaneDecisionPolicy(
        LeftLaneRuleConfig(
            occupancy_threshold_seconds=1.0,
            minimum_evidence_confidence=0.1,
        ),
        TrafficContextConfig(minimum_history_seconds=0.0),
        RightLaneOpportunityConfig(minimum_available_seconds=0.0),
        CalibrationConfig(
            suppress_candidates_when_unreliable=True,
            minimum_confidence_for_physical_measurements=0.8,
        ),
    )
    traffic = GlobalTrafficContext(
        congestion_level=CongestionLevel.FREE_FLOW,
        traffic_density=0.1,
        active_vehicle_count=1,
        lane_vehicle_counts={"left": 1, "right": 0},
        average_normalized_motion_per_second=0.1,
        confidence=0.9,
        coordinate_system="normalized_image",
        calibration_status=CalibrationStatus(
            mode="homography_fallback",
            valid=False,
            reprojection_error_pixels=None,
            confidence=0.0,
            reason="invalid homography",
        ),
    )
    vehicle = VehicleTrafficContext(
        track_id=1,
        neighbors=NeighborVehicles(),
        nearby_vehicle_count=0,
        adjacent_right_lane_id="right",
        right_lane_available=True,
        right_lane_available_seconds=5.0,
        right_lane_confidence=0.9,
    )
    overtake = OvertakingAssessment(
        track_id=1,
        status=OvertakingStatus.NOT_OVERTAKING,
        state=OvertakeState.NONE,
        confidence=0.9,
    )
    decision = policy.decide(5.0, 0.9, 5.0, traffic, vehicle, overtake)
    assert not decision.eligible
    assert decision.suppression_reason == "CALIBRATION_UNRELIABLE"

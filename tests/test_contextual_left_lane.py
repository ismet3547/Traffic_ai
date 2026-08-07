from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    CongestionConfig,
    GeometryIntegrityConfig,
    LaneChangeConfig,
    LaneConfig,
    LanesConfig,
    LeftLaneRuleConfig,
    OvertakingConfig,
    RightLaneOpportunityConfig,
    RoadPositionConfig,
    TrafficContextConfig,
)
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.geometry import GeometryIntegrityPolicy
from app.models import (
    BoundingBox,
    LaneObservation,
    OvertakeState,
    OvertakingStatus,
    PhysicalMeasurementPermission,
    TrackedVehicle,
)
from app.motion import LaneTransitionDetector, MotionHistoryStore
from app.overtaking import ContextualOvertakingPolicy
from app.positioning import NormalizedImageRoadPositionEstimator
from app.rules import ContextualLeftLaneDecisionPolicy, LeftLaneRuleEngine


@dataclass(frozen=True)
class _Spec:
    track_id: int
    lane: str
    longitudinal: float
    confidence: float = 0.9


class _Harness:
    def __init__(self, dense_count_per_lane: int = 2) -> None:
        self.context_config = TrafficContextConfig(
            history_seconds=5.0,
            minimum_history_seconds=0.4,
            maximum_samples_per_track=100,
        )
        self.opportunity_config = RightLaneOpportunityConfig(
            minimum_available_seconds=0.4,
            front_gap_normalized=0.08,
            rear_gap_normalized=0.06,
            minimum_confidence=0.5,
        )
        self.rule_config = LeftLaneRuleConfig(
            left_lane_id="left",
            occupancy_threshold_seconds=0.5,
            track_lost_grace_seconds=0.6,
            minimum_mean_confidence=0.25,
            minimum_evidence_confidence=0.6,
            overtaking_clearance_mode="contextual",
        )
        self.history = MotionHistoryStore(self.context_config)
        self.lane_changes = LaneTransitionDetector(
            LaneChangeConfig(
                confirmation_seconds=0.2,
                minimum_frames=2,
                state_ttl_seconds=1.0,
            )
        )
        self.positioning = NormalizedImageRoadPositionEstimator(
            RoadPositionConfig(travel_direction="toward_bottom")
        )
        self.context_analyzer = TrafficContextAnalyzer(
            ["left", "center", "right"],
            self.context_config,
            CongestionConfig(
                minimum_observed_vehicles=1,
                dense_vehicle_count_per_lane=dense_count_per_lane,
                moderate_density_ratio=0.70,
                dense_density_ratio=0.95,
            ),
        )
        self.opportunities = RightLaneOpportunityTracker(self.opportunity_config)
        self.overtaking = ContextualOvertakingPolicy(
            OvertakingConfig(
                observation_window_seconds=3.0,
                completion_timeout_seconds=3.0,
                minimum_confidence=0.65,
                entry_target_max_gap_normalized=0.2,
                pass_order_margin_normalized=0.01,
                post_overtake_grace_seconds=0.8,
            ),
            self.context_config,
            "left",
        )
        self.rule = LeftLaneRuleEngine(
            self.rule_config,
            ContextualLeftLaneDecisionPolicy(
                self.rule_config,
                self.context_config,
                self.opportunity_config,
            ),
        )
        self.geometry = GeometryIntegrityPolicy(
            GeometryIntegrityConfig(
                external_fixed_camera_guarantee=True,
                external_guarantee_id="synthetic-test-fixture",
            ),
            LanesConfig(
                reference_width=100,
                reference_height=100,
                lanes=[
                    LaneConfig(
                        id="left",
                        label="Left",
                        leftmost=True,
                        polygon=[(0, 0), (1, 0), (1, 1)],
                    )
                ],
            ),
        ).evaluate(
            100,
            100,
            None,
            PhysicalMeasurementPermission(False, 0.0, "unavailable"),
        )
        self.frame_index = 0
        self.transitions = []
        self.last_lane_transitions = []
        self.last_assessments = {}
        self.last_evaluation = None

    def step(self, timestamp: float, specs: list[_Spec]):
        raw = [_observation(spec) for spec in specs]
        lane_frame = self.lane_changes.update(raw, timestamp)
        positions = self.positioning.estimate(lane_frame.observations, 100, 100)
        context = self.context_analyzer.analyze(
            timestamp,
            lane_frame.observations,
            positions,
            self.history,
            geometry_integrity=self.geometry,
        )
        context = self.opportunities.update(context, timestamp)
        self.history.update(
            self.frame_index,
            timestamp,
            lane_frame.observations,
            positions,
            context.vehicles,
            lane_frame.transitions,
        )
        assessments = self.overtaking.update(
            timestamp,
            lane_frame.observations,
            lane_frame.transitions,
            context,
            self.history,
        )
        evaluation = self.rule.evaluate(
            lane_frame.observations,
            timestamp,
            traffic_context=context,
            overtaking_assessments=assessments,
            history_durations={
                item.vehicle.track_id: self.history.duration_seconds(
                    item.vehicle.track_id
                )
                for item in lane_frame.observations
            },
        )
        self.frame_index += 1
        self.transitions.extend(evaluation.transitions)
        self.last_lane_transitions = lane_frame.transitions
        self.last_assessments = assessments
        self.last_evaluation = evaluation
        return evaluation


def _observation(spec: _Spec) -> LaneObservation:
    lane_x = {"left": 20.0, "center": 50.0, "right": 80.0}[spec.lane]
    bottom_y = spec.longitudinal * 100
    return LaneObservation(
        vehicle=TrackedVehicle(
            track_id=spec.track_id,
            bbox=BoundingBox(lane_x - 4, bottom_y - 8, lane_x + 4, bottom_y),
            confidence=spec.confidence,
            class_id=2,
            class_name="car",
        ),
        lane_id=spec.lane,
    )


def _complete_pass(harness: _Harness) -> None:
    harness.step(0.00, [_Spec(1, "center", 0.20), _Spec(2, "center", 0.34)])
    harness.step(0.10, [_Spec(1, "left", 0.22), _Spec(2, "center", 0.35)])
    harness.step(0.31, [_Spec(1, "left", 0.25), _Spec(2, "center", 0.36)])
    harness.step(0.85, [_Spec(1, "left", 0.38), _Spec(2, "center", 0.40)])
    harness.step(1.00, [_Spec(1, "left", 0.44), _Spec(2, "center", 0.40)])


def test_legitimate_overtake_does_not_create_review_candidate() -> None:
    harness = _Harness()
    _complete_pass(harness)

    assert harness.last_assessments[1].status == OvertakingStatus.OVERTAKING_CONFIRMED
    assert harness.last_assessments[1].state == OvertakeState.PASSED_TARGET
    assert harness.last_evaluation.statuses[1].behavior_classification == "overtaking"
    assert not any(item.transition == "started" for item in harness.transitions)


def test_left_lane_without_pass_and_with_free_right_lane_becomes_candidate() -> None:
    harness = _Harness()
    harness.step(0.0, [_Spec(1, "left", 0.20)])
    harness.step(0.4, [_Spec(1, "left", 0.25)])
    evaluation = harness.step(0.6, [_Spec(1, "left", 0.28)])

    starts = [item for item in harness.transitions if item.transition == "started"]
    assert len(starts) == 1
    assert starts[0].review_reason_codes == (
        "LEFT_LANE_DURATION_EXCEEDED",
        "NO_ACTIVE_OVERTAKE",
        "RIGHT_LANE_AVAILABLE",
        "FREE_FLOW_TRAFFIC",
    )
    assert evaluation.statuses[1].is_review_candidate


def test_dense_traffic_suppresses_candidate() -> None:
    harness = _Harness(dense_count_per_lane=1)
    frames = [
        _Spec(1, "left", 0.30),
        _Spec(2, "center", 0.31),
        _Spec(3, "right", 0.32),
    ]
    harness.step(0.0, frames)
    harness.step(0.4, frames)
    evaluation = harness.step(0.7, frames)

    assert not any(item.transition == "started" for item in harness.transitions)
    assert evaluation.statuses[1].suppression_reason == "CONGESTION"
    assert evaluation.statuses[1].behavior_classification == "congestion"


def test_active_candidate_suspends_then_resumes_after_temporary_congestion() -> None:
    harness = _Harness(dense_count_per_lane=1)
    harness.step(0.0, [_Spec(1, "left", 0.20)])
    harness.step(0.4, [_Spec(1, "left", 0.25)])
    harness.step(0.6, [_Spec(1, "left", 0.28)])
    dense = [
        _Spec(1, "left", 0.30),
        _Spec(2, "center", 0.31),
        _Spec(3, "right", 0.32),
    ]
    suspended = harness.step(0.8, dense)
    harness.step(1.0, [_Spec(1, "left", 0.32)])
    resumed = harness.step(1.5, [_Spec(1, "left", 0.36)])

    assert suspended.transitions[0].transition == "suspended"
    assert suspended.statuses[1].candidate_lifecycle_state == "suspended"
    assert resumed.transitions[0].transition == "resumed"
    assert resumed.statuses[1].candidate_lifecycle_state == "candidate_active"


def test_active_candidate_cancels_after_persistent_congestion() -> None:
    harness = _Harness(dense_count_per_lane=1)
    harness.step(0.0, [_Spec(1, "left", 0.20)])
    harness.step(0.4, [_Spec(1, "left", 0.25)])
    harness.step(0.6, [_Spec(1, "left", 0.28)])
    dense = [
        _Spec(1, "left", 0.30),
        _Spec(2, "center", 0.31),
        _Spec(3, "right", 0.32),
    ]
    harness.step(0.8, dense)
    cancelled = harness.step(2.8, dense)

    assert cancelled.transitions[0].transition == "cancelled"
    assert cancelled.transitions[0].cancellation_reason == "CONGESTION"
    assert cancelled.statuses[1].candidate_lifecycle_state == "cancelled"
    assert not cancelled.statuses[1].is_review_candidate


def test_blocked_adjacent_right_lane_suppresses_candidate() -> None:
    harness = _Harness()
    frames = [
        _Spec(1, "left", 0.30),
        _Spec(2, "center", 0.34),
        _Spec(3, "center", 0.26),
    ]
    harness.step(0.0, frames)
    harness.step(0.4, frames)
    evaluation = harness.step(0.7, frames)

    assert not any(item.transition == "started" for item in harness.transitions)
    assert evaluation.statuses[1].suppression_reason == "RIGHT_LANE_UNAVAILABLE"


def test_temporary_tracker_disappearance_preserves_occupancy_state() -> None:
    harness = _Harness()
    harness.step(0.0, [_Spec(1, "left", 0.20)])
    harness.step(0.4, [_Spec(1, "left", 0.24)])
    harness.step(0.7, [])
    evaluation = harness.step(0.9, [_Spec(1, "left", 0.28)])

    assert evaluation.statuses[1].left_lane_duration_seconds == 0.9
    assert evaluation.statuses[1].is_review_candidate


def test_confirmed_overtake_followed_by_return_right_is_completed() -> None:
    harness = _Harness()
    _complete_pass(harness)
    harness.step(1.10, [_Spec(1, "center", 0.47), _Spec(2, "center", 0.41)])
    harness.step(1.31, [_Spec(1, "center", 0.50), _Spec(2, "center", 0.42)])

    assert harness.last_lane_transitions[0].from_lane == "left"
    assert harness.last_lane_transitions[0].to_lane == "center"
    assert harness.last_assessments[1].status == OvertakingStatus.OVERTAKING_CONFIRMED
    assert harness.last_assessments[1].state == OvertakeState.COMPLETED
    assert not any(item.transition == "started" for item in harness.transitions)


def test_stale_overtaking_attempt_expires() -> None:
    harness = _Harness()
    harness.step(0.00, [_Spec(1, "center", 0.20), _Spec(2, "center", 0.34)])
    harness.step(0.10, [_Spec(1, "left", 0.22), _Spec(2, "center", 0.35)])
    harness.step(0.31, [_Spec(1, "left", 0.25), _Spec(2, "center", 0.36)])
    harness.step(3.40, [_Spec(1, "left", 0.30), _Spec(2, "center", 0.40)])

    assert harness.last_assessments[1].state == OvertakeState.ABORTED
    assert harness.last_assessments[1].status == OvertakingStatus.NOT_OVERTAKING


def test_completed_overtake_does_not_grant_permanent_left_lane_immunity() -> None:
    harness = _Harness()
    _complete_pass(harness)
    harness.step(1.40, [_Spec(1, "left", 0.50), _Spec(2, "center", 0.42)])
    evaluation = harness.step(1.81, [_Spec(1, "left", 0.56), _Spec(2, "center", 0.44)])

    assert harness.last_assessments[1].status == OvertakingStatus.NOT_OVERTAKING
    assert evaluation.statuses[1].left_lane_duration_seconds == 0.81
    assert evaluation.statuses[1].is_review_candidate
    starts = [item for item in harness.transitions if item.transition == "started"]
    assert len(starts) == 1
    assert starts[0].start_timestamp_seconds == 1.0

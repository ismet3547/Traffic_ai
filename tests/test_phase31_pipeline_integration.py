from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.camera_motion import CameraPoseValidator
from app.config import (
    CalibrationConfig,
    CameraPoseValidationConfig,
    CandidateLifecycleConfig,
    CongestionConfig,
    GeometryIntegrityConfig,
    LaneChangeConfig,
    LaneConfig,
    LanesConfig,
    LeftLaneRuleConfig,
    PhysicalMeasurementsConfig,
    RightLaneOpportunityConfig,
    RoadPositionConfig,
    SpeedEstimationConfig,
    TrafficContextConfig,
)
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.geometry import GeometryIntegrityPolicy
from app.lanes import LaneAssigner
from app.models import (
    BoundingBox,
    CameraMotionEstimate,
    FramePacket,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    RuleEvaluation,
    TrackedVehicle,
    VideoInfo,
)
from app.motion import LaneTransitionDetector, MotionHistoryStore
from app.physical_measurements import PhysicalMeasurementPolicy
from app.pipeline import TrafficAnalysisPipeline
from app.positioning import build_road_coordinate_transformer
from app.rules import ContextualLeftLaneDecisionPolicy, LeftLaneRuleEngine
from app.speed import RollingSpeedEstimator


class _Source:
    def __init__(self, count: int, step: float = 0.2) -> None:
        self.info = VideoInfo(100, 100, 5.0, count)
        self._count = count
        self._step = step
        self.closed = False

    def __iter__(self):
        for index in range(self._count):
            yield FramePacket(
                index, index * self._step, np.zeros((100, 100, 3), np.uint8)
            )

    def close(self) -> None:
        self.closed = True


class _Detector:
    def detect(self, frame: np.ndarray):
        del frame
        return []


class _Tracker:
    def __init__(self, xs: list[float | None]) -> None:
        self._xs = iter(xs)

    def update(self, detections):
        del detections
        x = next(self._xs)
        if x is None:
            return []
        return [TrackedVehicle(1, BoundingBox(x - 4, 35, x + 4, 50), 0.9, 2, "car")]


class _Motion:
    def __init__(self, values: list[tuple[float, float]] | None = None) -> None:
        self._values = iter(values or [])

    def update(self, frame, excluded_boxes=None):
        del frame, excluded_boxes
        try:
            dx, rotation = next(self._values)
        except StopIteration:
            dx, rotation = 0.0, 0.0
        return CameraMotionEstimate(dx, 0.0, rotation, 0.95, True, "low", "fake", False)


class _Overtaking:
    def __init__(self, statuses: list[OvertakingStatus] | None = None) -> None:
        self._statuses = iter(statuses or [])

    def update(
        self,
        timestamp_seconds,
        observations,
        transitions,
        context,
        history,
        speeds=None,
    ):
        del transitions, context, history, speeds
        try:
            status = next(self._statuses)
        except StopIteration:
            status = OvertakingStatus.NOT_OVERTAKING
        return {
            item.vehicle.track_id: OvertakingAssessment(
                item.vehicle.track_id,
                status,
                OvertakeState.PASSED_TARGET
                if status == OvertakingStatus.OVERTAKING_CONFIRMED
                else OvertakeState.NONE,
                0.95,
                ("TARGET_PASSED_VEHICLE",)
                if status == OvertakingStatus.OVERTAKING_CONFIRMED
                else (),
                completed_at=timestamp_seconds
                if status == OvertakingStatus.OVERTAKING_CONFIRMED
                else None,
            )
            for item in observations
        }


class _Annotator:
    def annotate(self, frame, observations, statuses, traffic_context=None):
        del observations, statuses, traffic_context
        return frame


class _Writer:
    def __init__(self) -> None:
        self.transitions = []
        self.completed_count = 0
        self.cancelled_count = 0
        self._terminal: set[tuple[int, str]] = set()

    def process_frame(self, frame, transitions) -> None:
        del frame
        self._record(transitions)

    def finalize(self, transitions) -> None:
        self._record(transitions)

    def _record(self, transitions) -> None:
        self.transitions.extend(transitions)
        for transition in transitions:
            key = (transition.track_id, transition.transition)
            if key in self._terminal:
                continue
            if transition.transition in {"finalized", "ended"}:
                self.completed_count += 1
                self._terminal.add(key)
            elif transition.transition == "cancelled":
                self.cancelled_count += 1
                self._terminal.add(key)


@dataclass
class _CapturingRule:
    contexts: list

    def evaluate(self, observations, timestamp_seconds, traffic_context=None, **kwargs):
        del timestamp_seconds, kwargs
        self.contexts.append(traffic_context)
        return RuleEvaluation(
            statuses={},
            transitions=[],
            traffic_context=traffic_context.global_context if traffic_context else None,
        )

    def finalize(self):
        return []


def _lanes() -> LanesConfig:
    return LanesConfig(
        reference_width=100,
        reference_height=100,
        reference_pose_id="synthetic_fixed_pose",
        lanes=[
            LaneConfig(
                id="left",
                label="Left",
                leftmost=True,
                polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
            ),
            LaneConfig(
                id="right", label="Right", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)]
            ),
        ],
    )


def _calibration(calibrated: bool) -> CalibrationConfig:
    if not calibrated:
        return CalibrationConfig()
    return CalibrationConfig(
        mode="homography",
        image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        world_points=[(0, 0), (10, 0), (10, 20), (0, 20)],
        validation_image_points=[(25, 75), (75, 25)],
        validation_world_points=[(2.5, 15), (7.5, 5)],
        reference_width=100,
        reference_height=100,
        minimum_validation_coverage=0.2,
        fallback_to_normalized=False,
    )


def _run(
    xs: list[float | None],
    *,
    calibrated: bool = False,
    motion: list[tuple[float, float]] | None = None,
    overtake: list[OvertakingStatus] | None = None,
    capture_only: bool = False,
    motion_estimator=None,
):
    source = _Source(len(xs))
    lanes = _lanes()
    calibration = _calibration(calibrated)
    context_config = TrafficContextConfig(minimum_history_seconds=0.0)
    opportunity_config = RightLaneOpportunityConfig(
        minimum_available_seconds=0.0, minimum_confidence=0.5
    )
    rule_config = LeftLaneRuleConfig(
        occupancy_threshold_seconds=0.1,
        track_lost_grace_seconds=0.2,
        minimum_evidence_confidence=0.5,
    )
    capturing = _CapturingRule([])
    rule = (
        capturing
        if capture_only
        else LeftLaneRuleEngine(
            rule_config,
            ContextualLeftLaneDecisionPolicy(
                rule_config, context_config, opportunity_config, calibration
            ),
            CandidateLifecycleConfig(
                invalidation_grace_seconds=0.0,
                evidence_settle_seconds=0.2,
                track_loss_close_seconds=0.2,
                max_event_duration_seconds=10.0,
            ),
        )
    )
    writer = _Writer()
    pipeline = TrafficAnalysisPipeline(
        source=source,
        detector=_Detector(),
        tracker=_Tracker(xs),
        lane_assigner=LaneAssigner(lanes),
        lane_transition_detector=LaneTransitionDetector(
            LaneChangeConfig(confirmation_seconds=0.0, minimum_frames=2)
        ),
        road_position_estimator=build_road_coordinate_transformer(
            calibration, RoadPositionConfig(travel_direction="toward_bottom")
        ),
        speed_estimator=RollingSpeedEstimator(
            SpeedEstimationConfig(
                minimum_window_seconds=0.1,
                minimum_samples=2,
                max_position_jump_meters=2.0,
            )
        ),
        camera_motion_estimator=motion_estimator or _Motion(motion),
        camera_pose_validator=CameraPoseValidator(
            CameraPoseValidationConfig(
                minimum_samples=1,
                persistence_seconds=0.0,
                scale_persistence_seconds=0.0,
            )
        ),
        physical_measurement_policy=PhysicalMeasurementPolicy(
            PhysicalMeasurementsConfig(), calibration
        ),
        motion_history=MotionHistoryStore(context_config),
        traffic_context_analyzer=TrafficContextAnalyzer(
            ["left", "right"], context_config, CongestionConfig()
        ),
        right_lane_opportunities=RightLaneOpportunityTracker(opportunity_config),
        overtaking_policy=_Overtaking(overtake),
        rule_engine=rule,  # type: ignore[arg-type]
        event_writer=writer,  # type: ignore[arg-type]
        annotator=_Annotator(),  # type: ignore[arg-type]
        debug_sink=None,
        geometry_integrity_policy=GeometryIntegrityPolicy(
            GeometryIntegrityConfig(), lanes
        ),
    )
    summary = pipeline.run()
    return summary, writer, capturing


def test_pipeline_calibrated_fixed_camera_allows_world_measurements() -> None:
    _, _, capture = _run([25, 25], calibrated=True, capture_only=True)
    last = capture.contexts[-1]
    assert last.global_context.physical_measurements.allowed
    assert last.positions[1].world_position_m is not None
    assert last.speeds[1].speed_kph is not None


def test_pipeline_uncalibrated_flow_remains_normalized() -> None:
    _, _, capture = _run([25, 25], capture_only=True)
    last = capture.contexts[-1]
    assert not last.global_context.physical_measurements.allowed
    assert last.positions[1].world_position_m is None
    assert last.speeds[1].speed_kph is None


def test_pipeline_camera_move_disables_previously_valid_measurements() -> None:
    _, _, capture = _run(
        [25, 25], calibrated=True, motion=[(0.0, 0.0), (4.0, 0.0)], capture_only=True
    )
    assert capture.contexts[0].global_context.physical_measurements.allowed
    assert not capture.contexts[1].global_context.physical_measurements.allowed
    assert capture.contexts[1].speeds[1].speed_kph is None


def test_pipeline_tracker_jump_scrubs_world_position_before_context() -> None:
    _, _, capture = _run([25, 75], calibrated=True, capture_only=True)
    last = capture.contexts[-1]
    assert last.speeds[1].speed_mode == "rejected_position_jump"
    assert last.positions[1].world_position_m is None
    assert last.positions[1].physical_measurement_reason_codes == ("UNSTABLE_TRACK",)


def test_pipeline_delayed_overtake_cancels_open_candidate() -> None:
    statuses = [
        OvertakingStatus.NOT_OVERTAKING,
        OvertakingStatus.NOT_OVERTAKING,
        OvertakingStatus.OVERTAKING_CONFIRMED,
        OvertakingStatus.OVERTAKING_CONFIRMED,
    ]
    summary, writer, _ = _run([25, 25, 25, 25], overtake=statuses)
    assert any(item.transition == "started" for item in writer.transitions)
    assert any(
        item.transition == "cancelled"
        and item.cancellation_reason == "OVERTAKING_CONFIRMED"
        for item in writer.transitions
    )
    assert summary.review_candidates == 0
    assert summary.cancelled_candidates == 1


def test_pipeline_leave_left_settles_then_finalizes_once() -> None:
    summary, writer, _ = _run([25, 25, 75, 75, 75])
    kinds = [item.transition for item in writer.transitions]
    assert "pending_close" in kinds
    assert kinds.count("finalized") == 1
    assert summary.review_candidates == 1


def test_pipeline_track_disappearance_and_video_end_flush_are_deterministic() -> None:
    summary, writer, _ = _run([25, 25, None, None])
    assert any(item.close_reason == "track_lost" for item in writer.transitions)
    assert summary.review_candidates == 1

    end_summary, end_writer, _ = _run([25, 25])
    assert end_summary.review_candidates == 1
    assert any(item.end_reason == "video_ended" for item in end_writer.transitions)

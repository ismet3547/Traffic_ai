"""Core source-agnostic traffic analysis loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from app.camera_motion import CameraMotionEstimator, CameraPoseValidator
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.detection import Detector
from app.events import EventArtifactWriter
from app.geometry import GeometryIntegrityPolicy
from app.lanes import LaneAssigner
from app.models import PhysicalMeasurementPermission, RoadPosition, SpeedEstimate
from app.motion import LaneTransitionDetector, MotionHistoryStore
from app.overtaking import OvertakingClearancePolicy
from app.physical_measurements import PhysicalMeasurementPolicy
from app.positioning import RoadCoordinateTransformer
from app.rules import LeftLaneRuleEngine
from app.speed import SpeedEstimator
from app.tracking import VehicleTracker
from app.video.annotation import DebugAnnotator
from app.video.protocols import FrameSource, VideoSink

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    frames_processed: int
    duration_seconds: float
    review_candidates: int
    cancelled_candidates: int = 0


class TrafficAnalysisPipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: VehicleTracker,
        lane_assigner: LaneAssigner,
        lane_transition_detector: LaneTransitionDetector,
        road_position_estimator: RoadCoordinateTransformer,
        speed_estimator: SpeedEstimator,
        camera_motion_estimator: CameraMotionEstimator,
        camera_pose_validator: CameraPoseValidator,
        physical_measurement_policy: PhysicalMeasurementPolicy,
        motion_history: MotionHistoryStore,
        traffic_context_analyzer: TrafficContextAnalyzer,
        right_lane_opportunities: RightLaneOpportunityTracker,
        overtaking_policy: OvertakingClearancePolicy,
        rule_engine: LeftLaneRuleEngine,
        event_writer: EventArtifactWriter,
        annotator: DebugAnnotator,
        debug_sink: VideoSink | None,
        geometry_integrity_policy: GeometryIntegrityPolicy,
    ) -> None:
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._lane_assigner = lane_assigner
        self._lane_transition_detector = lane_transition_detector
        self._road_position_estimator = road_position_estimator
        self._speed_estimator = speed_estimator
        self._camera_motion_estimator = camera_motion_estimator
        self._camera_pose_validator = camera_pose_validator
        self._physical_measurement_policy = physical_measurement_policy
        self._motion_history = motion_history
        self._traffic_context_analyzer = traffic_context_analyzer
        self._right_lane_opportunities = right_lane_opportunities
        self._overtaking_policy = overtaking_policy
        self._rule_engine = rule_engine
        self._event_writer = event_writer
        self._annotator = annotator
        self._debug_sink = debug_sink
        self._geometry_integrity_policy = geometry_integrity_policy

    def run(self) -> PipelineSummary:
        frame_count = 0
        last_timestamp = 0.0
        try:
            for packet in self._source:
                detections = self._detector.detect(packet.image)
                vehicles = self._tracker.update(detections)
                camera_motion = self._camera_motion_estimator.update(
                    packet.image, [vehicle.bbox for vehicle in vehicles]
                )
                camera_pose = self._camera_pose_validator.update(
                    packet.timestamp_seconds, camera_motion
                )
                physical_permission = self._physical_measurement_policy.evaluate(
                    self._road_position_estimator.calibration_quality,
                    camera_pose,
                )
                geometry_integrity = self._geometry_integrity_policy.evaluate(
                    self._source.info.width,
                    self._source.info.height,
                    camera_pose,
                    physical_permission,
                )
                if not geometry_integrity.physical_measurements_allowed:
                    physical_permission = PhysicalMeasurementPermission(
                        allowed=False,
                        confidence=0.0,
                        status="unavailable_geometry_integrity",
                        reason_codes=tuple(
                            dict.fromkeys(
                                (
                                    *physical_permission.reason_codes,
                                    "GEOMETRY_INTEGRITY_LOST",
                                    *geometry_integrity.reason_codes,
                                )
                            )
                        ),
                    )
                raw_observations = self._lane_assigner.assign(
                    vehicles,
                    frame_width=self._source.info.width,
                    frame_height=self._source.info.height,
                    geometry_integrity=geometry_integrity,
                )
                lane_frame = self._lane_transition_detector.update(
                    raw_observations,
                    packet.timestamp_seconds,
                    geometry_allowed=geometry_integrity.lane_assignment_allowed,
                )
                observations = lane_frame.observations
                positions = self._road_position_estimator.estimate(
                    observations,
                    frame_width=self._source.info.width,
                    frame_height=self._source.info.height,
                    physical_permission=physical_permission,
                )
                speeds = self._speed_estimator.update(
                    packet.timestamp_seconds, positions, physical_permission
                )
                positions = self._scrub_unstable_physical_positions(
                    positions, speeds, physical_permission
                )
                traffic_context = self._traffic_context_analyzer.analyze(
                    packet.timestamp_seconds,
                    observations,
                    positions,
                    self._motion_history,
                    calibration_status=self._road_position_estimator.calibration_status,
                    camera_motion=camera_motion,
                    camera_pose=camera_pose,
                    physical_measurements=physical_permission,
                    speeds=speeds,
                    geometry_integrity=geometry_integrity,
                )
                traffic_context = self._right_lane_opportunities.update(
                    traffic_context, packet.timestamp_seconds
                )
                self._motion_history.update(
                    frame_index=packet.index,
                    timestamp_seconds=packet.timestamp_seconds,
                    observations=observations,
                    positions=positions,
                    vehicle_contexts=traffic_context.vehicles,
                    transitions=lane_frame.transitions,
                    speed_estimates=speeds,
                )
                overtaking_assessments = self._overtaking_policy.update(
                    timestamp_seconds=packet.timestamp_seconds,
                    observations=observations,
                    transitions=lane_frame.transitions,
                    context=traffic_context,
                    history=self._motion_history,
                    speeds=speeds,
                )
                evaluation = self._rule_engine.evaluate(
                    observations,
                    packet.timestamp_seconds,
                    traffic_context=traffic_context,
                    overtaking_assessments=overtaking_assessments,
                    history_durations={
                        observation.vehicle.track_id: self._motion_history.duration_seconds(
                            observation.vehicle.track_id
                        )
                        for observation in observations
                    },
                )
                annotated = self._annotator.annotate(
                    packet.image,
                    observations,
                    evaluation.statuses,
                    traffic_context=traffic_context,
                )
                self._event_writer.process_frame(annotated, evaluation.transitions)
                if self._debug_sink is not None:
                    self._debug_sink.write(annotated)
                frame_count += 1
                last_timestamp = packet.timestamp_seconds
                if frame_count % max(1, round(self._source.info.fps * 10)) == 0:
                    LOGGER.info(
                        "Processed %d frames (video time %.1fs)",
                        frame_count,
                        last_timestamp,
                    )

            self._event_writer.finalize(self._rule_engine.finalize())
        finally:
            # Also closes any partial candidate clip if inference is interrupted.
            self._event_writer.finalize([])
            self._source.close()
            if self._debug_sink is not None:
                self._debug_sink.close()

        return PipelineSummary(
            frames_processed=frame_count,
            duration_seconds=last_timestamp,
            review_candidates=self._event_writer.completed_count,
            cancelled_candidates=self._event_writer.cancelled_count,
        )

    def _scrub_unstable_physical_positions(
        self,
        positions: dict[int, RoadPosition],
        speeds: dict[int, SpeedEstimate],
        physical_permission: PhysicalMeasurementPermission,
    ) -> dict[int, RoadPosition]:
        """Prevent tracker jumps from reaching meter-gap/history consumers."""

        scrubbed = dict(positions)
        unstable_modes = {"rejected_position_jump", "rejected_unreasonable_speed"}
        for track_id, speed in speeds.items():
            position = scrubbed.get(track_id)
            if position is None or speed.speed_mode not in unstable_modes:
                continue
            track_permission = self._physical_measurement_policy.apply_track_stability(
                physical_permission, track_stable=False
            )
            normalized = position.normalized_position
            scrubbed[track_id] = replace(
                position,
                lateral=normalized[0] if normalized is not None else position.lateral,
                longitudinal=(
                    normalized[1] if normalized is not None else position.longitudinal
                ),
                coordinate_system="normalized_image",
                calibrated=False,
                world_position_m=None,
                world_position_confidence=0.0,
                physical_measurement_status=track_permission.status,
                physical_measurement_reason_codes=track_permission.reason_codes,
            )
        return scrubbed

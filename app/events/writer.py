"""Write representative images, bounded clips, and JSON review records."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.config import OutputConfig
from app.models import (
    CalibrationQualityMetadata,
    CameraMotionMetadata,
    CameraPoseMetadata,
    CandidateDecisionMetadata,
    CandidateLifecycleMetadata,
    CandidateTransition,
    EventMetadata,
    FrameGeometryMetadata,
    GapEstimate,
    GapEstimateMetadata,
    GeometryIntegrityMetadata,
    LaneGeometryMetadata,
    OvertakingAssessmentMetadata,
    PhysicalMeasurementMetadata,
    SpeedEstimateMetadata,
    TrafficContextMetadata,
    VideoInfo,
)

LOGGER = logging.getLogger(__name__)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return cv2


@dataclass(slots=True)
class _Recording:
    metadata: EventMetadata
    directory: Path
    writer: object
    written_frames: int = 0


class EventArtifactWriter:
    def __init__(
        self,
        run_directory: Path,
        output_config: OutputConfig,
        video_info: VideoInfo,
        source_video_name: str,
    ) -> None:
        self._run_directory = run_directory
        self._events_directory = run_directory / "events"
        self._events_directory.mkdir(parents=True, exist_ok=True)
        self._index_path = run_directory / "events.jsonl"
        self._cancelled_index_path = run_directory / "cancelled_events.jsonl"
        self._config = output_config
        self._video_info = video_info
        self._source_video_name = source_video_name
        pre_event_frames = max(
            0, round(output_config.clip_pre_event_seconds * video_info.fps)
        )
        self._prebuffer: deque[np.ndarray] = deque(maxlen=pre_event_frames)
        self._active: dict[int, _Recording] = {}
        self._completed_count = 0
        self._cancelled_count = 0

    @property
    def completed_count(self) -> int:
        return self._completed_count

    @property
    def cancelled_count(self) -> int:
        return self._cancelled_count

    def process_frame(
        self,
        frame: np.ndarray,
        transitions: list[CandidateTransition],
    ) -> None:
        for transition in transitions:
            if transition.transition == "started":
                self._start(transition, frame)
            elif transition.transition in {"suspended", "resumed", "pending_close"}:
                self._update(transition)

        for recording in self._active.values():
            self._write_clip_frame(recording, frame)

        for transition in transitions:
            if transition.transition in {"ended", "finalized"}:
                self._finish(transition, cancelled=False)
            elif transition.transition == "cancelled":
                self._finish(transition, cancelled=True)

        if self._prebuffer.maxlen:
            self._prebuffer.append(frame.copy())

    def finalize(self, transitions: list[CandidateTransition]) -> None:
        for transition in transitions:
            if transition.transition in {"ended", "finalized"}:
                self._finish(transition, cancelled=False)
            elif transition.transition == "cancelled":
                self._finish(transition, cancelled=True)
        for track_id in list(self._active):
            recording = self._active[track_id]
            fallback = CandidateTransition(
                transition="cancelled",
                track_id=track_id,
                lane_id=recording.metadata.lane_id,
                start_timestamp_seconds=recording.metadata.event_start_timestamp_seconds,
                timestamp_seconds=recording.metadata.candidate_created_timestamp_seconds,
                duration_seconds=recording.metadata.duration_seconds,
                confidence_score=recording.metadata.confidence_score,
                end_reason="pipeline_stopped",
                lifecycle_state="cancelled",
                cancelled_at=recording.metadata.candidate_created_timestamp_seconds,
                cancellation_reason="pipeline_stopped",
            )
            self._finish(fallback, cancelled=True)

    def _start(self, transition: CandidateTransition, frame: np.ndarray) -> None:
        if transition.track_id in self._active:
            return
        cv2 = _cv2()
        start_ms = round(transition.start_timestamp_seconds * 1000)
        event_id = f"left_lane_track_{transition.track_id}_{start_ms:010d}"
        event_directory = self._events_directory / event_id
        event_directory.mkdir(parents=True, exist_ok=False)
        image_path = event_directory / "representative.jpg"
        clip_path = event_directory / "event.mp4"
        image_params = [
            cv2.IMWRITE_JPEG_QUALITY,
            self._config.representative_image_quality,
        ]
        if not cv2.imwrite(str(image_path), frame, image_params):
            raise RuntimeError(f"could not write representative image: {image_path}")

        fourcc = cv2.VideoWriter_fourcc(*self._config.codec)
        writer = cv2.VideoWriter(
            str(clip_path),
            fourcc,
            self._video_info.fps,
            (self._video_info.width, self._video_info.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create event clip: {clip_path}")

        metadata = EventMetadata(
            event_id=event_id,
            track_id=transition.track_id,
            event_start_timestamp_seconds=transition.start_timestamp_seconds,
            candidate_created_timestamp_seconds=transition.timestamp_seconds,
            duration_seconds=transition.duration_seconds,
            lane_id=transition.lane_id,
            confidence_score=transition.confidence_score,
            source_video_name=self._source_video_name,
            representative_frame=image_path.name,
            event_video_clip=clip_path.name,
            behavior_classification=transition.behavior_classification,
            evidence_confidence_score=transition.evidence_confidence_score,
            review_reason_codes=list(transition.review_reason_codes),
            policy_version=transition.policy_version,
            traffic_context=_traffic_context_metadata(transition),
            overtaking_assessment=_overtaking_metadata(transition),
            candidate_lifecycle=_lifecycle_metadata(transition),
            decision_history=_decision_history(transition),
            calibration=_calibration_metadata(transition),
            camera_motion=_camera_motion_metadata(transition),
            camera_pose=_camera_pose_metadata(transition),
            physical_measurements=_physical_measurement_metadata(transition),
            geometry_integrity=_geometry_integrity_metadata(transition),
            speed_estimate=_speed_metadata(transition),
            image_position=(
                transition.position.image_position if transition.position else None
            ),
            normalized_position=(
                transition.position.normalized_position if transition.position else None
            ),
            world_position_m=(
                transition.position.world_position_m if transition.position else None
            ),
            coordinate_mode=(
                transition.position.coordinate_mode
                if transition.position
                else "normalized_image"
            ),
            inside_calibrated_region=(
                transition.position.inside_calibrated_region
                if transition.position
                else None
            ),
            calibrated_region_status=(
                transition.position.calibrated_region_status
                if transition.position
                else "unavailable"
            ),
        )
        recording = _Recording(
            metadata=metadata, directory=event_directory, writer=writer
        )
        self._active[transition.track_id] = recording
        for buffered_frame in self._prebuffer:
            self._write_clip_frame(recording, buffered_frame)
        self._write_metadata(recording)
        LOGGER.info("Review candidate started: %s", event_id)

    def _update(self, transition: CandidateTransition) -> None:
        recording = self._active.get(transition.track_id)
        if recording is None:
            return
        _refresh_metadata(recording.metadata, transition)
        self._write_metadata(recording)
        LOGGER.info(
            "Review candidate %s: %s",
            transition.transition,
            recording.metadata.event_id,
        )

    def _write_clip_frame(self, recording: _Recording, frame: np.ndarray) -> None:
        max_frames = max(
            1, round(self._config.clip_max_duration_seconds * self._video_info.fps)
        )
        if recording.written_frames >= max_frames:
            return
        recording.writer.write(frame)
        recording.written_frames += 1

    def _finish(self, transition: CandidateTransition, cancelled: bool) -> None:
        recording = self._active.pop(transition.track_id, None)
        if recording is None:
            return
        recording.writer.release()
        _refresh_metadata(recording.metadata, transition)
        recording.metadata.review_status = (
            "cancelled" if cancelled else "pending_human_review"
        )
        self._write_metadata(recording)
        index_path = self._cancelled_index_path if cancelled else self._index_path
        with index_path.open("a", encoding="utf-8") as index:
            index.write(recording.metadata.model_dump_json() + "\n")
        if cancelled:
            self._cancelled_count += 1
            LOGGER.info("Review candidate cancelled: %s", recording.metadata.event_id)
        else:
            self._completed_count += 1
            LOGGER.info("Review candidate finalized: %s", recording.metadata.event_id)

    @staticmethod
    def _write_metadata(recording: _Recording) -> None:
        metadata_path = recording.directory / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(
                recording.metadata.model_dump(mode="json"),
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")


def _traffic_context_metadata(
    transition: CandidateTransition,
) -> TrafficContextMetadata | None:
    traffic = transition.traffic_context
    vehicle = transition.vehicle_traffic_context
    if traffic is None:
        return None
    return TrafficContextMetadata(
        congestion_level=traffic.congestion_level.value,
        traffic_density=traffic.traffic_density,
        nearby_vehicle_count=(vehicle.nearby_vehicle_count if vehicle else 0),
        active_vehicle_count=traffic.active_vehicle_count,
        lane_vehicle_counts=traffic.lane_vehicle_counts,
        average_normalized_motion_per_second=(
            traffic.average_normalized_motion_per_second
        ),
        right_lane_available=(vehicle.right_lane_available if vehicle else None),
        right_lane_available_seconds=(
            vehicle.right_lane_available_seconds if vehicle else 0.0
        ),
        right_lane_confidence=(vehicle.right_lane_confidence if vehicle else 0.0),
        coordinate_system=traffic.coordinate_system,
        calibrated=(traffic.coordinate_system == "calibrated_world"),
        confidence=traffic.confidence,
        right_lane_opportunity_mode=(
            vehicle.right_lane_opportunity_mode if vehicle else "unavailable"
        ),
        right_lane_front_gap=_gap_metadata(
            vehicle.right_lane_front_gap if vehicle else None
        ),
        right_lane_rear_gap=_gap_metadata(
            vehicle.right_lane_rear_gap if vehicle else None
        ),
        right_lane_front_gap_m=_gap_value(
            vehicle.right_lane_front_gap if vehicle else None, "meters"
        ),
        right_lane_rear_gap_m=_gap_value(
            vehicle.right_lane_rear_gap if vehicle else None, "meters"
        ),
        right_lane_front_gap_normalized=_gap_value(
            vehicle.right_lane_front_gap if vehicle else None, "normalized"
        ),
        right_lane_rear_gap_normalized=_gap_value(
            vehicle.right_lane_rear_gap if vehicle else None, "normalized"
        ),
    )


def _overtaking_metadata(
    transition: CandidateTransition,
) -> OvertakingAssessmentMetadata | Literal["not_implemented"]:
    assessment = transition.overtaking_assessment
    if assessment is None:
        return "not_implemented"
    return OvertakingAssessmentMetadata(
        status=assessment.status.value,
        state=assessment.state.value,
        confidence=assessment.confidence,
        evidence=list(assessment.evidence),
        related_track_ids=list(assessment.related_track_ids),
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
    )


def _refresh_metadata(metadata: EventMetadata, transition: CandidateTransition) -> None:
    if transition.transition in {"ended", "finalized", "cancelled"}:
        metadata.event_end_timestamp_seconds = transition.timestamp_seconds
    metadata.duration_seconds = transition.duration_seconds
    metadata.confidence_score = transition.confidence_score
    metadata.end_reason = transition.end_reason
    metadata.candidate_lifecycle = _lifecycle_metadata(transition)
    metadata.decision_history = _decision_history(transition)
    if transition.traffic_context is not None:
        metadata.traffic_context = _traffic_context_metadata(transition)
    if transition.overtaking_assessment is not None:
        metadata.overtaking_assessment = _overtaking_metadata(transition)
    if transition.position is not None:
        metadata.image_position = transition.position.image_position
        metadata.normalized_position = transition.position.normalized_position
        metadata.world_position_m = transition.position.world_position_m
        metadata.coordinate_mode = transition.position.coordinate_mode
        metadata.inside_calibrated_region = transition.position.inside_calibrated_region
        metadata.calibrated_region_status = transition.position.calibrated_region_status
    if transition.calibration_quality is not None:
        metadata.calibration = _calibration_metadata(transition)
    if transition.camera_motion is not None:
        metadata.camera_motion = _camera_motion_metadata(transition)
    if transition.camera_pose is not None:
        metadata.camera_pose = _camera_pose_metadata(transition)
    if transition.physical_measurements is not None:
        metadata.physical_measurements = _physical_measurement_metadata(transition)
    if transition.geometry_integrity is not None:
        metadata.geometry_integrity = _geometry_integrity_metadata(transition)
    if transition.speed_estimate is not None:
        metadata.speed_estimate = _speed_metadata(transition)


def _gap_metadata(gap: GapEstimate | None) -> GapEstimateMetadata | None:
    if gap is None:
        return None
    return GapEstimateMetadata(
        value=gap.value,
        unit=gap.unit,
        confidence=gap.confidence,
        coordinate_mode=gap.coordinate_mode,
    )


def _gap_value(gap: GapEstimate | None, unit: str) -> float | None:
    return gap.value if gap is not None and gap.unit == unit else None


def _lifecycle_metadata(
    transition: CandidateTransition,
) -> CandidateLifecycleMetadata:
    legacy_finalized = transition.transition == "ended"
    return CandidateLifecycleMetadata(
        state="finalized" if legacy_finalized else transition.lifecycle_state,
        candidate_started_at=transition.candidate_started_at,
        suspended_at=transition.suspended_at,
        finalized_at=(
            transition.timestamp_seconds
            if legacy_finalized
            else transition.finalized_at
        ),
        cancelled_at=transition.cancelled_at,
        cancellation_reason=transition.cancellation_reason,
        close_requested_at=transition.close_requested_at,
        close_reason=transition.close_reason,
    )


def _decision_history(
    transition: CandidateTransition,
) -> list[CandidateDecisionMetadata]:
    return [
        CandidateDecisionMetadata(
            timestamp_seconds=item.timestamp_seconds,
            decision=item.decision,
            reason_codes=list(item.reason_codes),
        )
        for item in transition.decision_history
    ]


def _calibration_metadata(
    transition: CandidateTransition,
) -> CalibrationQualityMetadata | None:
    status = transition.calibration_quality
    if status is None:
        return None
    return CalibrationQualityMetadata(
        mode=status.mode,
        matrix_valid=status.matrix_valid,
        numerically_stable=status.numerically_stable,
        validation_mode=status.validation_mode,
        fit_reprojection_error_pixels=status.fit_reprojection_error_pixels,
        validation_reprojection_error_pixels=status.validation_reprojection_error_pixels,
        condition_metric=status.condition_metric,
        confidence=status.confidence,
        confidence_basis=status.confidence_basis,
        reason_codes=list(status.reason_codes),
        world_units=status.world_units,
        validation_world_rmse=status.validation_world_rmse,
        validation_world_mae=status.validation_world_mae,
        validation_world_max_error=status.validation_world_max_error,
        validation_world_p95_error=status.validation_world_p95_error,
        validation_coverage=status.validation_coverage,
        support_region_defined=status.support_region_defined,
        validity_region_status=(
            transition.position.calibrated_region_status
            if transition.position is not None
            else "unavailable"
        ),
    )


def _camera_motion_metadata(
    transition: CandidateTransition,
) -> CameraMotionMetadata | None:
    motion = transition.camera_motion
    if motion is None:
        return None
    return CameraMotionMetadata(
        dx=motion.dx,
        dy=motion.dy,
        rotation_degrees=motion.rotation_degrees,
        confidence=motion.confidence,
        valid=motion.valid,
        level=motion.level,
        method=motion.method,
        stabilization_applied=motion.stabilization_applied,
        scale_ratio=motion.scale_ratio,
        scale_delta=motion.scale_delta,
        scale_confidence=motion.scale_confidence,
        projective_drift_score=(
            motion.projective.drift_score if motion.projective else None
        ),
        projective_reprojection_error_pixels=(
            motion.projective.reprojection_error_pixels if motion.projective else None
        ),
        projective_inlier_ratio=(
            motion.projective.inlier_ratio if motion.projective else None
        ),
    )


def _camera_pose_metadata(
    transition: CandidateTransition,
) -> CameraPoseMetadata | None:
    pose = transition.camera_pose
    if pose is None:
        return None
    return CameraPoseMetadata(
        status=pose.status,
        translation_px=pose.translation_px,
        rotation_deg=pose.rotation_deg,
        confidence=pose.confidence,
        sample_count=pose.sample_count,
        stabilization_applied=pose.stabilization_applied,
        reason_codes=list(pose.reason_codes),
        cumulative_scale_ratio=pose.cumulative_scale_ratio,
        projective_drift_score=pose.projective_drift_score,
        verification_mode=(
            "runtime_background_diagnostic"
            if transition.geometry_integrity is not None
            and transition.geometry_integrity.trust_source == "measured_camera_pose"
            else transition.geometry_integrity.trust_source
            if transition.geometry_integrity is not None
            else "unavailable"
        ),
        trust_source=(
            transition.geometry_integrity.trust_source
            if transition.geometry_integrity is not None
            else "unavailable"
        ),
        cumulative_translation_px=pose.translation_px,
        cumulative_rotation_deg=pose.rotation_deg,
    )


def _geometry_integrity_metadata(
    transition: CandidateTransition,
) -> GeometryIntegrityMetadata | None:
    geometry = transition.geometry_integrity
    if geometry is None:
        return None
    frame = geometry.frame_geometry
    lane = geometry.lane_geometry
    return GeometryIntegrityMetadata(
        status=geometry.status.value,
        confidence=geometry.confidence,
        trust_source=geometry.trust_source,
        lane_assignment_allowed=geometry.lane_assignment_allowed,
        normalized_relationships_allowed=geometry.normalized_relationships_allowed,
        world_relationships_allowed=geometry.world_relationships_allowed,
        physical_measurements_allowed=geometry.physical_measurements_allowed,
        physical_speed_allowed=geometry.physical_speed_allowed,
        physical_gaps_allowed=geometry.physical_gaps_allowed,
        right_lane_opportunity_allowed=geometry.right_lane_opportunity_allowed,
        overtaking_inference_allowed=geometry.overtaking_inference_allowed,
        candidate_generation_allowed=geometry.candidate_generation_allowed,
        reason_codes=list(geometry.reason_codes),
        frame_geometry=FrameGeometryMetadata(
            width=frame.width,
            height=frame.height,
            aspect_ratio=frame.aspect_ratio,
            reference_width=frame.reference_width,
            reference_height=frame.reference_height,
            reference_aspect_ratio=frame.reference_aspect_ratio,
            scale_x=frame.scale_x,
            scale_y=frame.scale_y,
            compatible=frame.compatible,
            mapping_mode=frame.mapping_mode,
            scaling_mode=frame.scaling_mode,
            reason_codes=list(frame.reason_codes),
        ),
        lane_geometry=LaneGeometryMetadata(
            status=lane.status,
            trusted=lane.trusted,
            confidence=lane.confidence,
            reference_pose_id=lane.reference_pose_id,
            trust_source=lane.trust_source,
            reason_codes=list(lane.reason_codes),
        ),
    )


def _physical_measurement_metadata(
    transition: CandidateTransition,
) -> PhysicalMeasurementMetadata | None:
    permission = transition.physical_measurements
    if permission is None:
        return None
    return PhysicalMeasurementMetadata(
        allowed=permission.allowed,
        confidence=permission.confidence,
        status=permission.status,
        reason_codes=list(permission.reason_codes),
    )


def _speed_metadata(
    transition: CandidateTransition,
) -> SpeedEstimateMetadata | None:
    speed = transition.speed_estimate
    if speed is None:
        return None
    return SpeedEstimateMetadata(
        speed_mps=speed.speed_mps,
        speed_kph=speed.speed_kph,
        speed_confidence=speed.speed_confidence,
        speed_mode=speed.speed_mode,
        normalized_motion_rate=speed.normalized_motion_rate,
        sample_count=speed.sample_count,
        physical_measurement_status=speed.physical_measurement_status,
        reason_codes=list(speed.reason_codes),
    )

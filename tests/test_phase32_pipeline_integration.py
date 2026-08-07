from __future__ import annotations

from collections.abc import Iterator

from app.camera_motion import NoCameraMotionEstimator
from app.models import CameraMotionEstimate, ProjectivePoseEstimate
from tests.test_phase31_pipeline_integration import _run


class _MotionSequence:
    def __init__(self, values: list[CameraMotionEstimate]) -> None:
        self._values: Iterator[CameraMotionEstimate] = iter(values)

    def update(self, frame, excluded_boxes=None) -> CameraMotionEstimate:
        del frame, excluded_boxes
        try:
            return next(self._values)
        except StopIteration:
            return _motion()


def _motion(
    *, scale: float = 1.0, projective: ProjectivePoseEstimate | None = None
) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        dx=0.0,
        dy=0.0,
        rotation_degrees=0.0,
        confidence=0.95,
        valid=True,
        level="low",
        method="synthetic",
        scale_ratio=scale,
        scale_delta=scale - 1.0,
        scale_confidence=0.95,
        projective=projective,
    )


def test_trusted_fixed_camera_assigns_lanes_and_can_form_candidate() -> None:
    summary, writer, _ = _run([25, 25, 75, 75, 75])
    assert any(item.transition == "started" for item in writer.transitions)
    assert summary.review_candidates == 1


def test_disabled_pose_estimator_without_external_guarantee_blocks_candidates() -> None:
    summary, writer, _ = _run(
        [25, 25, 25, 25], motion_estimator=NoCameraMotionEstimator()
    )
    assert not any(item.transition == "started" for item in writer.transitions)
    assert summary.review_candidates == 0


def test_normalized_motion_remains_diagnostic_while_candidates_are_disabled() -> None:
    _, _, capture = _run(
        [25, 27],
        motion_estimator=NoCameraMotionEstimator(),
        capture_only=True,
    )
    context = capture.contexts[-1]
    assert context.positions[1].normalized_position is not None
    assert context.global_context.average_normalized_motion_per_second is not None
    assert not context.global_context.geometry_integrity.candidate_generation_allowed


def test_camera_zoom_mid_event_prevents_candidate_finalization() -> None:
    estimator = _MotionSequence(
        [_motion(), _motion(), _motion(scale=1.02), _motion(scale=1.02)]
    )
    summary, writer, _ = _run([25, 25, 25, 25], motion_estimator=estimator)
    assert any(item.transition == "started" for item in writer.transitions)
    assert any(item.transition == "suspended" for item in writer.transitions)
    assert not any(item.transition == "finalized" for item in writer.transitions)
    assert summary.review_candidates == 0


def test_projective_drift_invalidates_static_lane_geometry() -> None:
    diagnostic = ProjectivePoseEstimate(
        valid=True,
        drift_score=0.02,
        reprojection_error_pixels=0.2,
        inlier_ratio=0.9,
        confidence=0.9,
        reference_frame_index=0,
        sample_frame_index=2,
        method="synthetic_reference_homography",
    )
    estimator = _MotionSequence(
        [
            _motion(),
            _motion(),
            _motion(projective=diagnostic),
            _motion(projective=diagnostic),
        ]
    )
    _, writer, capture = _run(
        [25, 25, 25, 25],
        motion_estimator=estimator,
        capture_only=True,
    )
    assert not writer.transitions
    last = capture.contexts[-1].global_context.geometry_integrity
    assert last is not None and not last.lane_assignment_allowed
    assert "PROJECTIVE_DRIFT_DETECTED" in last.reason_codes


def test_calibrated_track_outside_support_region_has_no_speed_or_meter_gap() -> None:
    _, _, capture = _run([150, 150], calibrated=True, capture_only=True)
    context = capture.contexts[-1]
    position = context.positions[1]
    speed = context.speeds[1]
    vehicle = context.vehicles[1]
    assert position.world_position_m is None
    assert "OUTSIDE_CALIBRATION_REGION" in position.physical_measurement_reason_codes
    assert speed.speed_kph is None
    assert vehicle.right_lane_front_gap is None or (
        vehicle.right_lane_front_gap.unit == "normalized"
    )

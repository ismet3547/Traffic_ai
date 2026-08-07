from __future__ import annotations

import numpy as np
import pytest

from app.camera_motion import CameraPoseValidator
from app.camera_motion.feature_based import (
    _projective_corner_drift,
    _similarity_scale,
)
from app.candidates import CandidateLifecycleManager
from app.config import (
    CalibrationConfig,
    CameraPoseValidationConfig,
    CandidateLifecycleConfig,
    GeometryIntegrityConfig,
    LaneConfig,
    LanesConfig,
    LeftLaneRuleConfig,
    RightLaneOpportunityConfig,
    TrafficContextConfig,
)
from app.geometry import GeometryIntegrityPolicy, resolve_frame_geometry
from app.lanes import LaneAssigner
from app.models import (
    BehaviorClassification,
    BoundingBox,
    CameraMotionEstimate,
    CameraPoseStatus,
    CandidateDecision,
    CongestionLevel,
    GeometryIntegrityStatus,
    GlobalTrafficContext,
    NeighborVehicles,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    PhysicalMeasurementPermission,
    ProjectivePoseEstimate,
    TrackedVehicle,
    VehicleTrafficContext,
)
from app.rules import ContextualLeftLaneDecisionPolicy


def _lanes(*, scaling_mode: str = "uniform") -> LanesConfig:
    return LanesConfig(
        reference_width=100,
        reference_height=50,
        reference_pose_id="pose-a",
        scaling_mode=scaling_mode,
        lanes=[
            LaneConfig(
                id="left",
                label="Left",
                leftmost=True,
                polygon=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
            ),
            LaneConfig(
                id="right",
                label="Right",
                polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
            ),
        ],
    )


def _permission(allowed: bool = False) -> PhysicalMeasurementPermission:
    return PhysicalMeasurementPermission(
        allowed,
        0.9 if allowed else 0.0,
        "available_approximate" if allowed else "unavailable",
        () if allowed else ("CALIBRATION_NOT_CONFIGURED",),
    )


def _pose(status: str = "stable", confidence: float = 0.9) -> CameraPoseStatus:
    return CameraPoseStatus(status, 0.0, 0.0, confidence, 10)


def _motion(
    *,
    dx: float = 0.0,
    rotation: float = 0.0,
    scale: float = 1.0,
    projective: ProjectivePoseEstimate | None = None,
) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        dx=dx,
        dy=0.0,
        rotation_degrees=rotation,
        confidence=0.95,
        valid=True,
        level="low",
        method="synthetic",
        scale_ratio=scale,
        scale_delta=scale - 1.0,
        scale_confidence=0.95,
        projective=projective,
    )


def _pose_validator(**updates: object) -> CameraPoseValidator:
    values: dict[str, object] = {
        "minimum_samples": 1,
        "persistence_seconds": 0.0,
        "scale_persistence_seconds": 0.0,
    }
    values.update(updates)
    return CameraPoseValidator(CameraPoseValidationConfig(**values))


def test_identity_transform_stays_stable() -> None:
    assert _pose_validator().update(0.0, _motion()).status == "stable"


def test_partial_affine_scale_uses_both_column_norms() -> None:
    angle = 0.4
    scale = 1.01
    affine = np.asarray(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), 4.0],
            [scale * np.sin(angle), scale * np.cos(angle), -2.0],
        ]
    )
    assert _similarity_scale(affine) == pytest.approx(scale)


def test_projective_corner_metric_detects_perspective_not_similarity() -> None:
    import cv2

    affine = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    projective = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.002, 0.0, 1.0]])
    identity = np.eye(3)
    assert _projective_corner_drift(identity, affine, 100, 50, cv2) == pytest.approx(
        0.0
    )
    assert _projective_corner_drift(projective, affine, 100, 50, cv2) > 0.01


def test_one_percent_scale_drift_is_detected() -> None:
    status = _pose_validator().update(0.0, _motion(scale=1.011))
    assert status.status == "moved"
    assert "CAMERA_SCALE_CHANGED" in status.reason_codes


def test_cumulative_small_scale_drift_is_detected() -> None:
    validator = _pose_validator(scale_invalid_ratio=0.01)
    status = None
    for index in range(8):
        status = validator.update(index * 0.1, _motion(scale=1.002))
    assert status is not None and status.status == "moved"
    assert status.cumulative_scale_ratio is not None
    assert status.cumulative_scale_ratio > 1.01


def test_pure_zoom_invalidates_static_geometry() -> None:
    validator = _pose_validator()
    status = validator.update(0.0, _motion(scale=1.02))
    assert status.status == "moved"
    assert status.translation_px == pytest.approx(0.0)
    assert status.rotation_deg == pytest.approx(0.0)


def test_projective_reference_drift_is_detected() -> None:
    projective = ProjectivePoseEstimate(
        valid=True,
        drift_score=0.02,
        reprojection_error_pixels=0.2,
        inlier_ratio=0.9,
        confidence=0.9,
        reference_frame_index=0,
        sample_frame_index=15,
        method="synthetic_reference_homography",
    )
    status = _pose_validator().update(1.0, _motion(projective=projective))
    assert status.status == "moved"
    assert "PROJECTIVE_DRIFT_DETECTED" in status.reason_codes


def test_tiny_frame_motion_cannot_hide_reference_translation_drift() -> None:
    validator = _pose_validator(translation_invalid_px=3.0)
    status = None
    for index in range(20):
        status = validator.update(index * 0.1, _motion(dx=0.4))
    assert status is not None and status.status == "moved"
    assert status.translation_px is not None and status.translation_px > 3.0


def test_pose_unavailable_disables_candidates_by_default() -> None:
    assessment = GeometryIntegrityPolicy(GeometryIntegrityConfig(), _lanes()).evaluate(
        100, 50, _pose("unavailable", 0.0), _permission()
    )
    assert assessment.status == GeometryIntegrityStatus.UNVERIFIED
    assert not assessment.candidate_generation_allowed
    assert assessment.trust_source == "none"
    assert "CAMERA_POSE_UNVERIFIED" in assessment.reason_codes


def test_external_fixed_camera_guarantee_is_explicit_escape_hatch() -> None:
    assessment = GeometryIntegrityPolicy(
        GeometryIntegrityConfig(
            external_fixed_camera_guarantee=True,
            external_guarantee_id="controlled-tripod-7",
        ),
        _lanes(),
    ).evaluate(100, 50, _pose("unavailable", 0.0), _permission())
    assert assessment.status == GeometryIntegrityStatus.TRUSTED
    assert assessment.lane_assignment_allowed
    assert assessment.trust_source == "external_deployment_guarantee"
    assert "EXTERNAL_FIXED_CAMERA_GUARANTEE_USED" in assessment.reason_codes


def test_stable_runtime_pose_outranks_external_guarantee() -> None:
    assessment = _external_policy().evaluate(100, 50, _pose(), _permission(True))
    assert assessment.status == GeometryIntegrityStatus.TRUSTED
    assert assessment.trust_source == "runtime_pose_validation"
    assert "EXTERNAL_FIXED_CAMERA_GUARANTEE_USED" not in assessment.reason_codes


def test_uncertain_runtime_pose_cannot_be_overridden_by_external_guarantee() -> None:
    assessment = _external_policy().evaluate(
        100,
        50,
        _pose("uncertain"),
        _permission(True),
    )
    assert assessment.status == GeometryIntegrityStatus.DEGRADED
    assert assessment.trust_source == "none"
    assert not assessment.candidate_generation_allowed
    assert not assessment.physical_measurements_allowed
    assert not assessment.world_relationships_allowed
    assert not assessment.physical_speed_allowed
    assert not assessment.physical_gaps_allowed
    assert "CAMERA_POSE_UNCERTAIN" in assessment.reason_codes


def test_moved_runtime_pose_cannot_be_overridden_by_external_guarantee() -> None:
    assessment = _external_policy().evaluate(
        100,
        50,
        _pose("moved"),
        _permission(True),
    )
    assert assessment.status == GeometryIntegrityStatus.INVALID
    assert assessment.trust_source == "none"
    assert not assessment.candidate_generation_allowed
    assert not assessment.physical_measurements_allowed
    assert not assessment.world_relationships_allowed
    assert not assessment.physical_speed_allowed
    assert not assessment.physical_gaps_allowed
    assert "CAMERA_POSE_CHANGED" in assessment.reason_codes


def test_scale_change_cannot_be_overridden_by_external_guarantee() -> None:
    changed_pose = _pose_validator().update(0.0, _motion(scale=1.02))
    assessment = _external_policy().evaluate(100, 50, changed_pose, _permission(True))
    assert assessment.status == GeometryIntegrityStatus.INVALID
    assert assessment.trust_source == "none"
    assert "CAMERA_SCALE_CHANGED" in assessment.reason_codes
    assert not assessment.physical_measurements_allowed


def test_projective_drift_cannot_be_overridden_by_external_guarantee() -> None:
    diagnostic = ProjectivePoseEstimate(
        valid=True,
        drift_score=0.02,
        reprojection_error_pixels=0.2,
        inlier_ratio=0.9,
        confidence=0.9,
        method="synthetic_reference_homography",
    )
    changed_pose = _pose_validator().update(0.0, _motion(projective=diagnostic))
    assessment = _external_policy().evaluate(100, 50, changed_pose, _permission(True))
    assert assessment.status == GeometryIntegrityStatus.INVALID
    assert assessment.trust_source == "none"
    assert "PROJECTIVE_DRIFT_DETECTED" in assessment.reason_codes
    assert not assessment.physical_measurements_allowed


def test_camera_move_invalidates_previously_trusted_lane_geometry() -> None:
    policy = GeometryIntegrityPolicy(GeometryIntegrityConfig(), _lanes())
    assert policy.evaluate(100, 50, _pose(), _permission()).lane_geometry.trusted
    moved = policy.evaluate(100, 50, _pose("moved"), _permission())
    assert moved.status == GeometryIntegrityStatus.INVALID
    assert not moved.lane_geometry.trusted
    assert moved.trust_source == "none"


def test_invalid_lane_geometry_returns_unavailable_lane() -> None:
    policy = GeometryIntegrityPolicy(GeometryIntegrityConfig(), _lanes())
    invalid = policy.evaluate(100, 50, _pose("moved"), _permission())
    vehicle = TrackedVehicle(1, BoundingBox(10, 10, 20, 30), 0.9, 2, "car")
    observation = LaneAssigner(_lanes()).assign(
        [vehicle], 100, 50, geometry_integrity=invalid
    )[0]
    assert observation.lane_id is None


def test_missing_integrity_assessment_fails_lane_assignment_closed() -> None:
    vehicle = TrackedVehicle(1, BoundingBox(10, 10, 20, 30), 0.9, 2, "car")
    observation = LaneAssigner(_lanes()).assign([vehicle], 100, 50)[0]
    assert observation.lane_id is None


def test_active_candidate_suspends_when_geometry_is_lost() -> None:
    manager = CandidateLifecycleManager(CandidateLifecycleConfig())
    assert manager.update(4, 0.0, _eligible()).transition == "started"
    update = manager.update(4, 0.2, _geometry_lost())
    assert update.transition == "suspended"
    assert update.state.value == "suspended"


def test_persistent_geometry_failure_cancels_with_explicit_reason() -> None:
    manager = CandidateLifecycleManager(
        CandidateLifecycleConfig(invalidation_grace_seconds=1.0)
    )
    manager.update(4, 0.0, _eligible())
    manager.update(4, 0.2, _geometry_lost())
    update = manager.update(4, 1.2, _geometry_lost())
    assert update.transition == "cancelled"
    assert update.cancellation_reason == "GEOMETRY_INTEGRITY_LOST"


def test_geometry_failure_cannot_finalize_pending_candidate() -> None:
    manager = CandidateLifecycleManager(
        CandidateLifecycleConfig(
            invalidation_grace_seconds=0.5, evidence_settle_seconds=0.1
        )
    )
    manager.update(4, 0.0, _eligible())
    manager.request_close(4, 0.1, "left_lane_exit")
    suspended = manager.update(4, 0.2, _geometry_lost())
    cancelled = manager.update(4, 0.7, _geometry_lost())
    assert suspended.transition == "suspended"
    assert cancelled.transition == "cancelled"
    assert cancelled.finalized_at is None


def test_matching_resolution_is_accepted() -> None:
    frame = resolve_frame_geometry(100, 50, _lanes(), GeometryIntegrityConfig())
    assert frame.compatible and frame.mapping_mode == "exact"


def test_uniform_resize_is_accepted_when_configured() -> None:
    frame = resolve_frame_geometry(200, 100, _lanes(), GeometryIntegrityConfig())
    assert frame.compatible and frame.mapping_mode == "uniform_scale"


def test_incompatible_aspect_ratio_is_rejected() -> None:
    frame = resolve_frame_geometry(100, 60, _lanes(), GeometryIntegrityConfig())
    assert not frame.compatible
    assert "FRAME_ASPECT_RATIO_MISMATCH" in frame.reason_codes


def test_resize_is_rejected_in_exact_mode() -> None:
    frame = resolve_frame_geometry(
        200, 100, _lanes(scaling_mode="exact"), GeometryIntegrityConfig()
    )
    assert not frame.compatible
    assert "FRAME_SCALING_NOT_ALLOWED" in frame.reason_codes


def test_candidate_policy_requires_central_geometry_permission() -> None:
    trusted = GeometryIntegrityPolicy(
        GeometryIntegrityConfig(
            external_fixed_camera_guarantee=True,
            external_guarantee_id="test-fixture",
        ),
        _lanes(),
    ).evaluate(100, 50, None, _permission())
    unverified = GeometryIntegrityPolicy(GeometryIntegrityConfig(), _lanes()).evaluate(
        100, 50, None, _permission()
    )
    policy = ContextualLeftLaneDecisionPolicy(
        LeftLaneRuleConfig(
            occupancy_threshold_seconds=1.0, minimum_evidence_confidence=0.5
        ),
        TrafficContextConfig(minimum_history_seconds=0.0),
        RightLaneOpportunityConfig(
            minimum_available_seconds=0.0, minimum_confidence=0.5
        ),
        CalibrationConfig(),
    )
    vehicle = VehicleTrafficContext(
        track_id=1,
        neighbors=NeighborVehicles(),
        nearby_vehicle_count=0,
        adjacent_right_lane_id="right",
        right_lane_available=True,
        right_lane_available_seconds=2.0,
        right_lane_confidence=0.9,
    )
    overtake = OvertakingAssessment(
        1,
        OvertakingStatus.NOT_OVERTAKING,
        OvertakeState.NONE,
        0.9,
    )
    accepted = policy.decide(2.0, 0.9, 2.0, _traffic(trusted), vehicle, overtake)
    rejected = policy.decide(2.0, 0.9, 2.0, _traffic(unverified), vehicle, overtake)
    assert accepted.eligible
    assert not rejected.eligible
    assert rejected.suppression_reason == "GEOMETRY_INTEGRITY_LOST"


def _eligible() -> CandidateDecision:
    return CandidateDecision(
        True,
        BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION,
        0.9,
        ("LEFT_LANE_DURATION_EXCEEDED",),
    )


def _external_policy() -> GeometryIntegrityPolicy:
    return GeometryIntegrityPolicy(
        GeometryIntegrityConfig(
            external_fixed_camera_guarantee=True,
            external_guarantee_id="controlled-tripod-7",
        ),
        _lanes(),
    )


def _geometry_lost() -> CandidateDecision:
    return CandidateDecision(
        False,
        BehaviorClassification.INSUFFICIENT_EVIDENCE,
        0.0,
        suppression_reason="GEOMETRY_INTEGRITY_LOST",
    )


def _traffic(geometry) -> GlobalTrafficContext:
    return GlobalTrafficContext(
        congestion_level=CongestionLevel.FREE_FLOW,
        traffic_density=0.2,
        active_vehicle_count=2,
        lane_vehicle_counts={"left": 1, "right": 1},
        average_normalized_motion_per_second=0.05,
        confidence=0.9,
        geometry_integrity=geometry,
    )

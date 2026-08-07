from __future__ import annotations

from app.models import (
    FrameGeometry,
    GeometryIntegrityAssessment,
    GeometryIntegrityStatus,
    LaneGeometryTrust,
)


def trusted_geometry(
    width: int = 100,
    height: int = 100,
    *,
    physical_allowed: bool = False,
) -> GeometryIntegrityAssessment:
    frame = FrameGeometry(
        width=width,
        height=height,
        aspect_ratio=width / height,
        reference_width=width,
        reference_height=height,
        reference_aspect_ratio=width / height,
        scale_x=1.0,
        scale_y=1.0,
        compatible=True,
        mapping_mode="exact",
        scaling_mode="uniform",
    )
    lane = LaneGeometryTrust(
        status="trusted",
        confidence=0.95,
        reference_pose_id="synthetic-test-pose",
        trust_source="synthetic_test_fixture",
    )
    return GeometryIntegrityAssessment(
        status=GeometryIntegrityStatus.TRUSTED,
        confidence=0.95,
        trust_source="synthetic_test_fixture",
        lane_assignment_allowed=True,
        normalized_relationships_allowed=True,
        world_relationships_allowed=physical_allowed,
        physical_measurements_allowed=physical_allowed,
        physical_speed_allowed=physical_allowed,
        physical_gaps_allowed=physical_allowed,
        right_lane_opportunity_allowed=True,
        overtaking_inference_allowed=True,
        candidate_generation_allowed=True,
        frame_geometry=frame,
        lane_geometry=lane,
    )

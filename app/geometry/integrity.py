"""Central fail-closed geometry capability policy."""

from __future__ import annotations

import logging

from app.config import GeometryIntegrityConfig, LanesConfig
from app.geometry.frame import resolve_frame_geometry
from app.models import (
    CameraPoseStatus,
    GeometryIntegrityAssessment,
    GeometryIntegrityStatus,
    LaneGeometryTrust,
    PhysicalMeasurementPermission,
)

LOGGER = logging.getLogger(__name__)


class GeometryIntegrityPolicy:
    """Authorizes every downstream use of camera-relative road geometry."""

    def __init__(self, config: GeometryIntegrityConfig, lanes: LanesConfig) -> None:
        self._config = config
        self._lanes = lanes
        self._last_signature: tuple[str, tuple[str, ...]] | None = None

    def evaluate(
        self,
        frame_width: int,
        frame_height: int,
        camera_pose: CameraPoseStatus | None,
        physical_permission: PhysicalMeasurementPermission,
    ) -> GeometryIntegrityAssessment:
        frame = resolve_frame_geometry(
            frame_width, frame_height, self._lanes, self._config
        )
        pose_status = camera_pose.status if camera_pose is not None else "unavailable"
        pose_confidence = camera_pose.confidence if camera_pose is not None else 0.0
        reasons = list(frame.reason_codes)

        external = self._config.external_fixed_camera_guarantee
        if not self._config.enabled:
            trust_source = "gate_disabled_by_configuration"
            pose_trusted = True
            reasons.append("GEOMETRY_GATE_DISABLED")
        elif external:
            trust_source = "external_deployment_guarantee"
            pose_trusted = True
            reasons.append("EXTERNAL_FIXED_CAMERA_GUARANTEE")
        else:
            trust_source = "measured_camera_pose"
            pose_trusted = (
                pose_status == "stable"
                and pose_confidence >= self._config.minimum_pose_confidence
            )
            if pose_status == "unavailable":
                reasons.append("CAMERA_POSE_UNAVAILABLE")
            elif pose_status == "uncertain":
                reasons.append("CAMERA_POSE_UNCERTAIN")
            elif pose_status == "moved":
                reasons.append("CAMERA_POSE_MOVED")
            elif pose_confidence < self._config.minimum_pose_confidence:
                reasons.append("CAMERA_POSE_CONFIDENCE_LOW")
            if camera_pose is not None:
                reasons.extend(camera_pose.reason_codes)

        geometry_allowed = frame.compatible and pose_trusted
        if geometry_allowed:
            status = GeometryIntegrityStatus.TRUSTED
            confidence = 0.85 if external else min(1.0, pose_confidence)
        elif not frame.compatible or pose_status == "moved":
            status = GeometryIntegrityStatus.INVALID
            confidence = 0.0
        elif pose_status == "uncertain":
            status = GeometryIntegrityStatus.DEGRADED
            confidence = 0.0
        else:
            status = GeometryIntegrityStatus.UNVERIFIED
            confidence = 0.0

        lane_geometry = LaneGeometryTrust(
            status=status.value,
            confidence=confidence,
            reference_pose_id=self._lanes.reference_pose_id,
            trust_source=trust_source,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
        assessment = GeometryIntegrityAssessment(
            status=status,
            confidence=confidence,
            trust_source=trust_source,
            lane_assignment_allowed=geometry_allowed,
            normalized_relationships_allowed=geometry_allowed,
            world_relationships_allowed=(
                geometry_allowed and physical_permission.allowed
            ),
            physical_measurements_allowed=(
                geometry_allowed and physical_permission.allowed
            ),
            physical_speed_allowed=(geometry_allowed and physical_permission.allowed),
            physical_gaps_allowed=(geometry_allowed and physical_permission.allowed),
            right_lane_opportunity_allowed=geometry_allowed,
            overtaking_inference_allowed=geometry_allowed,
            candidate_generation_allowed=geometry_allowed,
            frame_geometry=frame,
            lane_geometry=lane_geometry,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
        self._log_change(assessment)
        return assessment

    def _log_change(self, assessment: GeometryIntegrityAssessment) -> None:
        signature = (assessment.status.value, assessment.reason_codes)
        if signature == self._last_signature:
            return
        log = LOGGER.info if assessment.candidate_generation_allowed else LOGGER.warning
        log(
            "Geometry integrity %s; candidate generation %s: %s",
            assessment.status.value,
            "enabled" if assessment.candidate_generation_allowed else "disabled",
            ", ".join(assessment.reason_codes) or "verified",
        )
        self._last_signature = signature

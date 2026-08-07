"""One policy gate for every meter and physical-speed output."""

from __future__ import annotations

import logging

from app.config import CalibrationConfig, PhysicalMeasurementsConfig
from app.models import (
    CalibrationQuality,
    CameraPoseStatus,
    PhysicalMeasurementPermission,
)

LOGGER = logging.getLogger(__name__)


class PhysicalMeasurementPolicy:
    """Fail closed unless calibration and camera pose are trustworthy."""

    def __init__(
        self,
        config: PhysicalMeasurementsConfig,
        calibration_config: CalibrationConfig,
    ) -> None:
        self._config = config
        self._calibration_config = calibration_config
        self._last_signature: tuple[bool, tuple[str, ...]] | None = None

    def evaluate(
        self,
        calibration: CalibrationQuality,
        camera_pose: CameraPoseStatus | None,
        *,
        track_stable: bool = True,
        transform_valid: bool = True,
    ) -> PhysicalMeasurementPermission:
        reasons: list[str] = []
        if calibration.mode != "homography":
            reasons.append("CALIBRATION_NOT_CONFIGURED")
        if not calibration.matrix_valid or not transform_valid:
            reasons.append("CALIBRATION_MATRIX_INVALID")
        if not calibration.numerically_stable:
            reasons.append("HOMOGRAPHY_POORLY_CONDITIONED")
        independently_validated = (
            calibration.validation_mode == "INDEPENDENT_VALIDATION_POINTS"
            and calibration.validation_reprojection_error_pixels is not None
            and "VALIDATION_ERROR_HIGH" not in calibration.reason_codes
        )
        permit_unverified = (
            self._calibration_config.allow_unverified_physical_measurements
        )
        if (
            self._config.require_independent_validation
            and not independently_validated
            and not permit_unverified
        ):
            reasons.append("CALIBRATION_UNVERIFIED")
        if (
            calibration.confidence < self._config.minimum_calibration_confidence
            and not (
                permit_unverified and calibration.validation_mode == "FIT_POINTS_ONLY"
            )
        ):
            reasons.append("CALIBRATION_CONFIDENCE_LOW")
        pose_status = camera_pose.status if camera_pose is not None else "unavailable"
        if pose_status == "moved" and self._config.disable_on_camera_pose_moved:
            reasons.append("CAMERA_POSE_UNSTABLE")
        elif (
            pose_status == "uncertain" and self._config.disable_on_camera_pose_uncertain
        ):
            reasons.append("CAMERA_POSE_UNCERTAIN")
        elif (
            pose_status == "unavailable"
            and self._config.disable_on_camera_pose_unavailable
        ):
            reasons.append("CAMERA_POSE_UNAVAILABLE")
        if camera_pose is not None and camera_pose.stabilization_applied:
            # No current implementation applies compensation; keeping this
            # explicit prevents a diagnostic estimate from being mistaken for it.
            reasons.append("UNSUPPORTED_STABILIZATION_STATE")
        if not track_stable:
            reasons.append("UNSTABLE_TRACK")

        reasons = list(dict.fromkeys(reasons))
        allowed = not reasons
        permission = PhysicalMeasurementPermission(
            allowed=allowed,
            confidence=calibration.confidence if allowed else 0.0,
            status="available_approximate" if allowed else "unavailable",
            reason_codes=tuple(reasons),
        )
        signature = (permission.allowed, permission.reason_codes)
        if signature != self._last_signature:
            if permission.allowed:
                LOGGER.info(
                    "Physical measurements enabled (confidence %.2f)",
                    permission.confidence,
                )
            else:
                LOGGER.warning(
                    "Physical measurements disabled: %s",
                    ", ".join(permission.reason_codes),
                )
            self._last_signature = signature
        return permission

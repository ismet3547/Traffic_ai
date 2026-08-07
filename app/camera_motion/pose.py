"""Persistent fixed-camera pose validation for static homographies."""

from __future__ import annotations

import logging
import math
from collections import deque

from app.config import CameraPoseValidationConfig
from app.models import CameraMotionEstimate, CameraPoseStatus

LOGGER = logging.getLogger(__name__)


class CameraPoseValidator:
    """Accumulates diagnostic frame transforms relative to a rolling baseline.

    It validates camera stability; it never stabilizes, warps, or compensates a
    frame. Persistent pose movement is latched for the remainder of the run.
    """

    def __init__(self, config: CameraPoseValidationConfig) -> None:
        self._config = config
        self._samples: deque[tuple[float, float, float, float, float]] = deque(
            maxlen=120
        )
        self._cumulative_dx = 0.0
        self._cumulative_dy = 0.0
        self._cumulative_rotation = 0.0
        self._cumulative_scale = 1.0
        self._latest_projective_drift = 0.0
        self._unstable_since: float | None = None
        self._moved = False
        self._last_status: str | None = None

    def update(
        self, timestamp_seconds: float, estimate: CameraMotionEstimate
    ) -> CameraPoseStatus:
        if not self._config.enabled:
            return self._result(
                "unavailable",
                None,
                None,
                None,
                None,
                0.0,
                ("POSE_VALIDATION_DISABLED",),
            )
        if not estimate.valid:
            return self._result(
                "unavailable",
                None,
                None,
                None,
                None,
                0.0,
                ("CAMERA_MOTION_ESTIMATE_UNAVAILABLE",),
            )

        dx = (
            0.0
            if abs(estimate.dx) <= self._config.translation_noise_floor_px
            else estimate.dx
        )
        dy = (
            0.0
            if abs(estimate.dy) <= self._config.translation_noise_floor_px
            else estimate.dy
        )
        rotation = (
            0.0
            if abs(estimate.rotation_degrees) <= self._config.rotation_noise_floor_deg
            else estimate.rotation_degrees
        )
        self._cumulative_dx += dx
        self._cumulative_dy += dy
        self._cumulative_rotation += rotation
        measured_scale = (
            estimate.scale_ratio if estimate.scale_ratio is not None else 1.0
        )
        if abs(measured_scale - 1.0) <= self._config.scale_noise_floor_ratio:
            measured_scale = 1.0
        self._cumulative_scale *= measured_scale
        if estimate.projective is not None and estimate.projective.valid:
            self._latest_projective_drift = estimate.projective.drift_score or 0.0
        scale_drift = abs(self._cumulative_scale - 1.0)
        self._samples.append(
            (
                timestamp_seconds,
                math.hypot(self._cumulative_dx, self._cumulative_dy),
                abs(self._cumulative_rotation),
                scale_drift,
                self._latest_projective_drift,
            )
        )
        translation = self._samples[-1][1]
        rotation_abs = self._samples[-1][2]
        if len(self._samples) < self._config.minimum_samples:
            return self._result(
                "unavailable",
                translation,
                self._cumulative_rotation,
                self._cumulative_scale,
                self._latest_projective_drift,
                estimate.confidence * 0.5,
                ("CAMERA_POSE_INITIALIZING",),
            )

        invalid = (
            translation >= self._config.translation_invalid_px
            or rotation_abs >= self._config.rotation_invalid_deg
            or scale_drift >= self._config.scale_invalid_ratio
            or self._latest_projective_drift >= self._config.projective_invalid_score
        )
        warning = (
            translation >= self._config.translation_warning_px
            or rotation_abs >= self._config.rotation_warning_deg
            or scale_drift >= self._config.scale_warning_ratio
            or self._latest_projective_drift >= self._config.projective_warning_score
        )
        if invalid:
            if self._unstable_since is None:
                self._unstable_since = timestamp_seconds
            persistence = (
                self._config.scale_persistence_seconds
                if scale_drift >= self._config.scale_invalid_ratio
                else self._config.persistence_seconds
            )
            if timestamp_seconds - self._unstable_since >= persistence:
                self._moved = True
        elif not warning:
            self._unstable_since = None

        detailed_reasons: list[str] = []
        if scale_drift >= self._config.scale_warning_ratio:
            detailed_reasons.append("CAMERA_SCALE_CHANGED")
        if self._latest_projective_drift >= self._config.projective_warning_score:
            detailed_reasons.append("PROJECTIVE_DRIFT_DETECTED")
        if translation >= self._config.translation_warning_px:
            detailed_reasons.append("CAMERA_TRANSLATION_CHANGED")
        if rotation_abs >= self._config.rotation_warning_deg:
            detailed_reasons.append("CAMERA_ROTATION_CHANGED")
        if self._moved:
            status = "moved"
            reasons = ("CAMERA_POSE_UNSTABLE", *detailed_reasons)
        elif invalid or warning:
            status = "uncertain"
            reasons = ("CAMERA_POSE_CHANGE_DETECTED", *detailed_reasons)
        else:
            status, reasons = "stable", ()
        return self._result(
            status,
            translation,
            self._cumulative_rotation,
            self._cumulative_scale,
            self._latest_projective_drift,
            estimate.confidence,
            reasons,
        )

    def _result(
        self,
        status: str,
        translation: float | None,
        rotation: float | None,
        cumulative_scale: float | None,
        projective_drift: float | None,
        confidence: float,
        reasons: tuple[str, ...],
    ) -> CameraPoseStatus:
        if status != self._last_status and status in {"uncertain", "moved"}:
            LOGGER.warning(
                "Camera pose transitioned %s -> %s: %s",
                (self._last_status or "initializing").upper(),
                status.upper(),
                ", ".join(reasons),
            )
        self._last_status = status
        return CameraPoseStatus(
            status=status,
            translation_px=translation,
            rotation_deg=rotation,
            confidence=max(0.0, min(1.0, confidence)),
            sample_count=len(self._samples),
            stabilization_applied=False,
            reason_codes=reasons,
            cumulative_scale_ratio=cumulative_scale,
            projective_drift_score=projective_drift,
        )

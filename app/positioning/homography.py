"""Road-plane homography with explicit mathematical and physical quality."""

from __future__ import annotations

import math

import numpy as np

from app.config import CalibrationConfig, RoadPositionConfig
from app.models import (
    CalibrationQuality,
    LaneObservation,
    PhysicalMeasurementPermission,
    RoadPosition,
)


class CalibrationError(ValueError):
    """Raised when a configured homography is not mathematically usable."""


class HomographyRoadTransformer:
    coordinate_system = "calibrated_world"
    calibrated = True

    def __init__(
        self,
        calibration: CalibrationConfig,
        road_position: RoadPositionConfig,
    ) -> None:
        if calibration.mode != "homography":
            raise CalibrationError(
                "homography transformer requires calibration.mode=homography"
            )
        self._config = calibration
        self._road_position = road_position
        image_points = np.asarray(calibration.image_points, dtype=np.float64)
        world_points = np.asarray(calibration.world_points, dtype=np.float64)
        self._condition_metric = _normalized_dlt_condition(image_points, world_points)
        if (
            not math.isfinite(self._condition_metric)
            or self._condition_metric > calibration.maximum_condition_number
        ):
            raise CalibrationError(
                "HOMOGRAPHY_POORLY_CONDITIONED: normalized DLT condition metric "
                f"{self._condition_metric:.3g} exceeds "
                f"{calibration.maximum_condition_number:.3g}"
            )
        self._matrix = _calculate_homography(image_points, world_points)
        if (
            np.linalg.matrix_rank(self._matrix) < 3
            or abs(np.linalg.det(self._matrix)) < 1e-12
        ):
            raise CalibrationError(
                "HOMOGRAPHY_SINGULAR: matrix rank/determinant invalid"
            )
        try:
            self._inverse_matrix = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("HOMOGRAPHY_SINGULAR: inverse unavailable") from exc
        if not np.all(np.isfinite(self._inverse_matrix)):
            raise CalibrationError(
                "HOMOGRAPHY_SINGULAR: inverse contains non-finite values"
            )

        fit_error = _pixel_reprojection_error(
            image_points, world_points, self._inverse_matrix
        )
        if not math.isfinite(fit_error):
            raise CalibrationError("fit reprojection error is not finite")
        projected_world = _perspective_transform(image_points, self._matrix)
        if np.any(
            np.abs(projected_world) > calibration.maximum_absolute_world_coordinate
        ):
            raise CalibrationError("PROJECTED_COORDINATES_OUT_OF_RANGE")

        validation_error: float | None = None
        reasons: list[str] = []
        if calibration.validation_image_points:
            validation_mode = "INDEPENDENT_VALIDATION_POINTS"
            validation_image = np.asarray(
                calibration.validation_image_points, dtype=np.float64
            )
            validation_world = np.asarray(
                calibration.validation_world_points, dtype=np.float64
            )
            validation_error = _pixel_reprojection_error(
                validation_image, validation_world, self._inverse_matrix
            )
            confidence_basis = "independent_validation_reprojection_error"
            if (
                not math.isfinite(validation_error)
                or validation_error
                > calibration.maximum_validation_reprojection_error_pixels
            ):
                reasons.append("VALIDATION_ERROR_HIGH")
                confidence = 0.0
            else:
                error_score = max(
                    0.0,
                    1.0
                    - validation_error
                    / calibration.maximum_validation_reprojection_error_pixels,
                )
                condition_score = max(
                    0.5,
                    1.0 - self._condition_metric / calibration.maximum_condition_number,
                )
                confidence = min(0.95, 0.95 * error_score * condition_score)
        else:
            validation_mode = "FIT_POINTS_ONLY"
            validation_error = None
            confidence = 0.25
            confidence_basis = "unverified_control_points_only"
            reasons.append("CALIBRATION_UNVERIFIED")

        if fit_error > calibration.maximum_reprojection_error_pixels:
            reasons.append("FIT_REPROJECTION_ERROR_HIGH")
            confidence = min(confidence, 0.20)
        self._calibration_quality = CalibrationQuality(
            mode="homography",
            matrix_valid=True,
            numerically_stable=True,
            validation_mode=validation_mode,
            fit_reprojection_error_pixels=fit_error,
            validation_reprojection_error_pixels=validation_error,
            condition_metric=self._condition_metric,
            confidence=confidence,
            confidence_basis=confidence_basis,
            reason_codes=tuple(reasons),
            world_units=calibration.world_units,
        )

    @property
    def calibration_quality(self) -> CalibrationQuality:
        return self._calibration_quality

    @property
    def calibration_status(self) -> CalibrationQuality:
        """Phase 3 compatibility alias."""

        return self._calibration_quality

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def image_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        mapped = _perspective_transform(
            np.asarray([point], dtype=np.float64), self._matrix
        )[0]
        if np.any(np.abs(mapped) > self._config.maximum_absolute_world_coordinate):
            raise CalibrationError("PROJECTED_COORDINATES_OUT_OF_RANGE")
        return float(mapped[0]), float(mapped[1])

    def world_to_image(self, point: tuple[float, float]) -> tuple[float, float]:
        mapped = _perspective_transform(
            np.asarray([point], dtype=np.float64), self._inverse_matrix
        )[0]
        return float(mapped[0]), float(mapped[1])

    def estimate(
        self,
        observations: list[LaneObservation],
        frame_width: int,
        frame_height: int,
        physical_permission: PhysicalMeasurementPermission | None = None,
    ) -> dict[int, RoadPosition]:
        width = max(1, frame_width)
        height = max(1, frame_height)
        permission = physical_permission or PhysicalMeasurementPermission(
            allowed=False,
            confidence=0.0,
            status="unavailable",
            reason_codes=("PHYSICAL_PERMISSION_REQUIRED",),
        )
        positions: dict[int, RoadPosition] = {}
        for observation in observations:
            image_xy = observation.vehicle.bbox.bottom_center
            x, y = image_xy
            normalized_x = _clamp(x / width)
            normalized_y = _clamp(y / height)
            normalized_longitudinal = (
                normalized_y
                if self._road_position.travel_direction == "toward_bottom"
                else 1.0 - normalized_y
            )
            if permission.allowed:
                try:
                    projected_world = self.image_to_world(image_xy)
                except CalibrationError:
                    projected_world = None
                if projected_world is not None:
                    if self._config.world_longitudinal_axis == "y":
                        world_lateral, world_longitudinal = projected_world
                    else:
                        world_lateral, world_longitudinal = (
                            projected_world[1],
                            projected_world[0],
                        )
                    if self._config.world_longitudinal_direction == "negative":
                        world_longitudinal = -world_longitudinal
                    lateral, longitudinal = world_lateral, world_longitudinal
                    coordinate_system = "calibrated_world"
                    world_position_m = projected_world
                    position_status = permission.status
                    position_reasons = permission.reason_codes
                else:
                    lateral, longitudinal = normalized_x, normalized_longitudinal
                    coordinate_system = "normalized_image"
                    world_position_m = None
                    position_status = "unavailable"
                    position_reasons = ("COORDINATE_TRANSFORM_INVALID",)
            else:
                lateral, longitudinal = normalized_x, normalized_longitudinal
                coordinate_system = "normalized_image"
                world_position_m = None
                position_status = permission.status
                position_reasons = permission.reason_codes
            track_id = observation.vehicle.track_id
            positions[track_id] = RoadPosition(
                track_id=track_id,
                lateral=float(lateral),
                longitudinal=float(longitudinal),
                coordinate_system=coordinate_system,
                calibrated=(permission.allowed and world_position_m is not None),
                image_position=image_xy,
                normalized_position=(normalized_x, normalized_longitudinal),
                world_position_m=world_position_m,
                world_position_confidence=(
                    permission.confidence if world_position_m is not None else 0.0
                ),
                physical_measurement_status=position_status,
                physical_measurement_reason_codes=position_reasons,
            )
        return positions


def _calculate_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise CalibrationError("calibration point arrays must both have shape (N, 2)")
    design = _design_matrix(source, destination)
    if np.linalg.matrix_rank(design) < 8:
        raise CalibrationError("HOMOGRAPHY_SINGULAR: correspondences are degenerate")
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is a declared dependency
        cv2 = None
    if cv2 is not None:
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None:
            raise CalibrationError("OpenCV could not calculate a homography matrix")
    else:
        _, _, right_vectors = np.linalg.svd(design)
        matrix = right_vectors[-1].reshape(3, 3)
    scale = matrix[2, 2]
    if abs(scale) < 1e-12:
        scale = float(np.linalg.norm(matrix))
    if abs(scale) < 1e-12:
        raise CalibrationError("homography calculation produced a zero matrix")
    matrix = matrix / scale
    if not np.all(np.isfinite(matrix)):
        raise CalibrationError("homography calculation produced non-finite values")
    return matrix


def _normalized_dlt_condition(source: np.ndarray, destination: np.ndarray) -> float:
    normalized_source = _normalize_points(source)
    normalized_destination = _normalize_points(destination)
    singular_values = np.linalg.svd(
        _design_matrix(normalized_source, normalized_destination),
        compute_uv=False,
    )
    denominator_index = -2 if len(singular_values) >= 9 else -1
    denominator = float(singular_values[denominator_index])
    if denominator <= 1e-12:
        return math.inf
    return float(singular_values[0] / denominator)


def _normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    root_mean_square = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if root_mean_square <= 1e-12:
        raise CalibrationError("CONTROL_POINTS_NEAR_DEGENERATE")
    return centered * (math.sqrt(2.0) / root_mean_square)


def _design_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    return np.asarray(rows, dtype=np.float64)


def _pixel_reprojection_error(
    image_points: np.ndarray,
    world_points: np.ndarray,
    inverse_matrix: np.ndarray,
) -> float:
    projected_image = _perspective_transform(world_points, inverse_matrix)
    return float(
        np.sqrt(np.mean(np.sum((projected_image - image_points) ** 2, axis=1)))
    )


def _perspective_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = (matrix @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise CalibrationError("point projects to infinity under the homography")
    result = projected[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(result)):
        raise CalibrationError("projection produced non-finite values")
    return result


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

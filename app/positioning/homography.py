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

        self._support_region = _convex_hull(
            np.vstack(
                (
                    image_points,
                    np.asarray(calibration.validation_image_points, dtype=np.float64),
                )
            )
            if calibration.validation_image_points
            else image_points
        )
        validation_error: float | None = None
        world_rmse: float | None = None
        world_mae: float | None = None
        world_max: float | None = None
        world_p95: float | None = None
        validation_coverage: float | None = None
        reasons: list[str] = []
        if calibration.reference_width is None or calibration.reference_height is None:
            reasons.append("CALIBRATION_REFERENCE_SIZE_UNAVAILABLE")
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
            world_errors = np.linalg.norm(
                _perspective_transform(validation_image, self._matrix)
                - validation_world,
                axis=1,
            )
            world_rmse = float(np.sqrt(np.mean(world_errors**2)))
            world_mae = float(np.mean(world_errors))
            world_max = float(np.max(world_errors))
            world_p95 = float(np.percentile(world_errors, 95))
            validation_coverage = _validation_coverage(image_points, validation_image)
            confidence_basis = "independent_world_and_pixel_validation"
            if (
                not math.isfinite(validation_error)
                or validation_error
                > calibration.maximum_validation_reprojection_error_pixels
            ):
                reasons.append("VALIDATION_ERROR_HIGH")
            if (
                world_rmse > calibration.maximum_validation_rmse_world_units
                or world_p95 > calibration.maximum_validation_p95_world_units
            ):
                reasons.append("VALIDATION_WORLD_ERROR_HIGH")
            if validation_coverage < calibration.minimum_validation_coverage:
                reasons.append("VALIDATION_POINTS_CLUSTERED")
                reasons.append("ROAD_REGION_POORLY_VALIDATED")
            pixel_score = max(
                0.0,
                1.0
                - validation_error
                / calibration.maximum_validation_reprojection_error_pixels,
            )
            world_score = max(
                0.0,
                1.0 - world_rmse / calibration.maximum_validation_rmse_world_units,
            )
            condition_score = max(
                0.5,
                1.0 - self._condition_metric / calibration.maximum_condition_number,
            )
            confidence = min(
                0.95,
                0.95
                * pixel_score
                * world_score
                * condition_score
                * min(
                    1.0,
                    validation_coverage
                    / max(0.01, calibration.minimum_validation_coverage),
                ),
            )
            if any(
                reason in reasons
                for reason in (
                    "VALIDATION_ERROR_HIGH",
                    "VALIDATION_WORLD_ERROR_HIGH",
                    "ROAD_REGION_POORLY_VALIDATED",
                )
            ):
                confidence = min(confidence, 0.20)
        else:
            validation_mode = "FIT_POINTS_ONLY"
            validation_error = None
            confidence = 0.25
            confidence_basis = "unverified_control_points_only"
            reasons.append("CALIBRATION_UNVERIFIED")

        if fit_error > calibration.maximum_reprojection_error_pixels:
            reasons.append("FIT_REPROJECTION_ERROR_HIGH")
            confidence = min(confidence, 0.20)
        if "CALIBRATION_REFERENCE_SIZE_UNAVAILABLE" in reasons:
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
            validation_world_rmse=world_rmse,
            validation_world_mae=world_mae,
            validation_world_max_error=world_max,
            validation_world_p95_error=world_p95,
            validation_coverage=validation_coverage,
            support_region_defined=len(self._support_region) >= 3,
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

    @property
    def support_region_image_points(self) -> np.ndarray:
        return self._support_region.copy()

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

    def is_inside_support_region(self, point: tuple[float, float]) -> bool:
        """Return whether a reference-image point is within measured coverage."""

        if len(self._support_region) < 3:
            return False
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return False
        distance = cv2.pointPolygonTest(
            self._support_region.astype(np.float32), point, True
        )
        return bool(distance >= -self._config.support_region_margin_pixels)

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
            calibration_xy, frame_compatible = self._to_reference_point(
                image_xy, frame_width, frame_height
            )
            inside_region = bool(
                frame_compatible
                and calibration_xy is not None
                and self.is_inside_support_region(calibration_xy)
            )
            if permission.allowed and frame_compatible and inside_region:
                try:
                    projected_world = self.image_to_world(calibration_xy)
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
            elif permission.allowed and not frame_compatible:
                lateral, longitudinal = normalized_x, normalized_longitudinal
                coordinate_system = "normalized_image"
                world_position_m = None
                position_status = "unavailable"
                position_reasons = ("CALIBRATION_FRAME_GEOMETRY_INCOMPATIBLE",)
            elif permission.allowed and not inside_region:
                lateral, longitudinal = normalized_x, normalized_longitudinal
                coordinate_system = "normalized_image"
                world_position_m = None
                position_status = "unavailable"
                position_reasons = ("OUTSIDE_CALIBRATION_REGION",)
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
                inside_calibrated_region=inside_region,
                calibrated_region_status=(
                    "inside" if inside_region else "outside_or_unavailable"
                ),
            )
        return positions

    def _to_reference_point(
        self,
        point: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[float, float] | None, bool]:
        reference_width = self._config.reference_width
        reference_height = self._config.reference_height
        if reference_width is None or reference_height is None:
            return None, False
        scale_x = frame_width / reference_width
        scale_y = frame_height / reference_height
        if not math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-9):
            return None, False
        return (point[0] / scale_x, point[1] / scale_y), True


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


def _convex_hull(points: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return np.empty((0, 2), dtype=np.float64)
    return cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)


def _validation_coverage(control: np.ndarray, validation: np.ndarray) -> float:
    """Score how broadly independent points span the fitted image region."""

    control_span = np.ptp(control, axis=0)
    validation_span = np.ptp(validation, axis=0)
    ratios = np.divide(
        validation_span,
        control_span,
        out=np.zeros_like(validation_span),
        where=control_span > 1e-9,
    )
    span_score = float(np.clip(np.mean(ratios), 0.0, 1.0))
    count_score = min(1.0, len(validation) / 4.0)
    return span_score * count_score


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

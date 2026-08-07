"""Road-plane homography computed once at application startup."""

from __future__ import annotations

import math

import numpy as np

from app.config import CalibrationConfig, RoadPositionConfig
from app.models import CalibrationStatus, LaneObservation, RoadPosition


class CalibrationError(ValueError):
    """Raised when a configured homography cannot be trusted."""


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
        self._matrix = _calculate_homography(image_points, world_points)
        try:
            self._inverse_matrix = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("homography matrix is singular") from exc

        projected_image = _perspective_transform(world_points, self._inverse_matrix)
        error = float(
            np.sqrt(np.mean(np.sum((projected_image - image_points) ** 2, axis=1)))
        )
        if not math.isfinite(error):
            raise CalibrationError("homography reprojection error is not finite")
        confidence = max(
            0.0,
            min(1.0, 1.0 - error / calibration.maximum_reprojection_error_pixels),
        )
        valid = error <= calibration.maximum_reprojection_error_pixels
        self._calibration_status = CalibrationStatus(
            mode="homography",
            valid=valid,
            reprojection_error_pixels=error,
            confidence=confidence,
            reason=None if valid else "reprojection error exceeds configured maximum",
            world_units=calibration.world_units,
        )
        if not valid:
            raise CalibrationError(
                "homography reprojection error "
                f"{error:.3f}px exceeds {calibration.maximum_reprojection_error_pixels:.3f}px"
            )

    @property
    def calibration_status(self) -> CalibrationStatus:
        return self._calibration_status

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def image_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        mapped = _perspective_transform(
            np.asarray([point], dtype=np.float64), self._matrix
        )[0]
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
    ) -> dict[int, RoadPosition]:
        width = max(1, frame_width)
        height = max(1, frame_height)
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
            world_xy = self.image_to_world(image_xy)
            if self._config.world_longitudinal_axis == "y":
                world_lateral, world_longitudinal = world_xy[0], world_xy[1]
            else:
                world_lateral, world_longitudinal = world_xy[1], world_xy[0]
            if self._config.world_longitudinal_direction == "negative":
                world_longitudinal = -world_longitudinal
            physical_available = (
                self._calibration_status.confidence
                >= self._config.minimum_confidence_for_physical_measurements
            )
            lateral = world_lateral if physical_available else normalized_x
            longitudinal = (
                world_longitudinal if physical_available else normalized_longitudinal
            )
            track_id = observation.vehicle.track_id
            positions[track_id] = RoadPosition(
                track_id=track_id,
                lateral=float(lateral),
                longitudinal=float(longitudinal),
                coordinate_system=(
                    "calibrated_world" if physical_available else "normalized_image"
                ),
                calibrated=physical_available,
                image_position=image_xy,
                normalized_position=(normalized_x, normalized_longitudinal),
                world_position=world_xy,
                calibration_confidence=self._calibration_status.confidence,
            )
        return positions


def _calculate_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise CalibrationError("calibration point arrays must both have shape (N, 2)")
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        raise CalibrationError("calibration correspondences are degenerate")
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is a declared runtime dependency
        cv2 = None
    if cv2 is not None:
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None:
            raise CalibrationError("OpenCV could not calculate a homography matrix")
    else:
        # Keeps geometry unit-testable in minimal environments while the runtime
        # application uses OpenCV's calibrated implementation when installed.
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


def _perspective_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = (matrix @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise CalibrationError("point projects to infinity under the homography")
    return projected[:, :2] / denominator[:, None]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

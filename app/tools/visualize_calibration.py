"""Render a non-GUI calibration preview for a configured video."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from app.config import load_config
from app.lanes import LaneAssigner
from app.models import CameraPoseStatus
from app.physical_measurements import PhysicalMeasurementPolicy
from app.positioning import HomographyRoadTransformer, build_road_coordinate_transformer

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save a preview of lane geometry and road-plane calibration."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", help="Override video.input_path")
    parser.add_argument("--output", default="output/calibration_preview.jpg")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--show", action="store_true", help="Also open an OpenCV window"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("calibration visualization requires opencv-python") from exc

    config = load_config(args.config)
    video_path = Path(args.video or config.video.input_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        if args.frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {args.frame_index} from {video_path}")

    transformer = build_road_coordinate_transformer(
        config.calibration, config.road_position
    )
    preview = frame.copy()
    height, width = preview.shape[:2]
    lane_assigner = LaneAssigner(config.lanes)
    for lane_id, points in lane_assigner.polygons_for_frame(width, height).items():
        polygon = np.asarray(points, dtype=np.int32)
        cv2.polylines(preview, [polygon], True, (60, 230, 170), 2, cv2.LINE_AA)
        centroid = polygon.mean(axis=0).astype(int)
        cv2.putText(
            preview,
            lane_id,
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 230, 170),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(config.calibration.image_points):
        pixel = tuple(round(value) for value in point)
        cv2.circle(preview, pixel, 6, (20, 80, 255), -1, cv2.LINE_AA)
        world = config.calibration.world_points[index]
        cv2.putText(
            preview,
            f"P{index}: {world[0]:.1f},{world[1]:.1f}m",
            (pixel[0] + 8 if index % 2 == 0 else pixel[0] - 150, pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 80, 255),
            1,
            cv2.LINE_AA,
        )

    for index, point in enumerate(config.calibration.validation_image_points):
        pixel = tuple(round(value) for value in point)
        cv2.drawMarker(
            preview,
            pixel,
            (255, 80, 220),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"V{index}",
            (pixel[0] + 8, pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 80, 220),
            1,
            cv2.LINE_AA,
        )

    if isinstance(transformer, HomographyRoadTransformer):
        _draw_world_grid(preview, transformer, config.calibration.world_points, cv2)
    status = transformer.calibration_quality
    permission = PhysicalMeasurementPolicy(
        config.physical_measurements, config.calibration
    ).evaluate(
        status,
        CameraPoseStatus(
            status="unavailable",
            translation_px=None,
            rotation_deg=None,
            confidence=0.0,
            sample_count=0,
            reason_codes=("SINGLE_FRAME_TOOL_CANNOT_VALIDATE_CAMERA_POSE",),
        ),
    )
    fit = (
        f"{status.fit_reprojection_error_pixels:.3f}"
        if status.fit_reprojection_error_pixels is not None
        else "N/A"
    )
    validation = (
        f"{status.validation_reprojection_error_pixels:.3f}"
        if status.validation_reprojection_error_pixels is not None
        else "N/A"
    )
    label = (
        f"CALIBRATION {status.mode.upper()} matrix={status.matrix_valid} "
        f"validation={status.validation_mode} fit={fit}px validation_error={validation}px "
        f"confidence={status.confidence:.2f} physical={'ON' if permission.allowed else 'OFF'}"
    )
    cv2.rectangle(preview, (0, 0), (min(width, 900), 34), (30, 30, 30), -1)
    cv2.putText(
        preview,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"could not write preview: {output_path}")
    LOGGER.info("Saved calibration preview: %s", output_path)
    LOGGER.info(
        "Calibration diagnostics: matrix_valid=%s numerically_stable=%s "
        "validation_mode=%s fit_error_px=%s validation_error_px=%s "
        "condition_metric=%s confidence=%.2f basis=%s reasons=%s",
        status.matrix_valid,
        status.numerically_stable,
        status.validation_mode,
        status.fit_reprojection_error_pixels,
        status.validation_reprojection_error_pixels,
        status.condition_metric,
        status.confidence,
        status.confidence_basis,
        ",".join(status.reason_codes) or "none",
    )
    LOGGER.info(
        "Physical measurement permission: allowed=%s reasons=%s",
        permission.allowed,
        ",".join(permission.reason_codes) or "none",
    )
    if status.validation_mode == "FIT_POINTS_ONLY":
        LOGGER.warning(
            "Calibration matrix is mathematically solvable but physically unverified. "
            "Physical speed/gap measurements are disabled by default."
        )
    if args.show:
        cv2.imshow("Traffic AI calibration preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _draw_world_grid(
    image: np.ndarray,
    transformer: HomographyRoadTransformer,
    world_points: list[tuple[float, float]],
    cv2: object,
) -> None:
    if not world_points:
        return
    xs = np.asarray([point[0] for point in world_points])
    ys = np.asarray([point[1] for point in world_points])
    for x in np.linspace(float(xs.min()), float(xs.max()), 7):
        _world_line(image, transformer, (x, float(ys.min())), (x, float(ys.max())), cv2)
    for y in np.linspace(float(ys.min()), float(ys.max()), 11):
        _world_line(image, transformer, (float(xs.min()), y), (float(xs.max()), y), cv2)


def _world_line(
    image: np.ndarray,
    transformer: HomographyRoadTransformer,
    start: tuple[float, float],
    end: tuple[float, float],
    cv2: object,
) -> None:
    first = tuple(round(value) for value in transformer.world_to_image(start))
    second = tuple(round(value) for value in transformer.world_to_image(end))
    cv2.line(image, first, second, (255, 170, 40), 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())

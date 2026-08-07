"""Debug overlay rendering."""

from __future__ import annotations

import numpy as np

from app.config import OutputConfig
from app.lanes import LaneAssigner
from app.models import (
    GapEstimate,
    LaneObservation,
    TrafficFrameContext,
    VehicleRuleStatus,
)


class DebugAnnotator:
    def __init__(
        self, lane_assigner: LaneAssigner, output_config: OutputConfig | None = None
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenCV is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self._cv2 = cv2
        self._lane_assigner = lane_assigner
        self._config = output_config or OutputConfig()

    def annotate(
        self,
        frame: np.ndarray,
        observations: list[LaneObservation],
        statuses: dict[int, VehicleRuleStatus],
        traffic_context: TrafficFrameContext | None = None,
    ) -> np.ndarray:
        cv2 = self._cv2
        output = frame.copy()
        overlay = output.copy()
        height, width = frame.shape[:2]
        polygons = self._lane_assigner.polygons_for_frame(width, height)

        for lane_id, points in polygons.items():
            polygon = np.asarray(points, dtype=np.int32)
            color = (
                (80, 120, 255)
                if lane_id == self._lane_assigner.leftmost_lane_id
                else (80, 220, 160)
            )
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
            centroid = polygon.mean(axis=0).astype(int)
            cv2.putText(
                output,
                lane_id,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        output = cv2.addWeighted(overlay, 0.14, output, 0.86, 0)

        if traffic_context is not None:
            traffic = traffic_context.global_context
            calibration = traffic.calibration_quality
            motion = traffic.camera_motion
            pose = traffic.camera_pose
            permission = traffic.physical_measurements
            geometry = traffic.geometry_integrity
            calibration_label = (
                f"{calibration.mode.upper()}/{_quality(calibration.confidence)}"
                if calibration is not None
                else "UNKNOWN"
            )
            motion_label = (
                motion.level.upper()
                if motion is not None and motion.valid
                else "UNKNOWN"
            )
            traffic_label = (
                f"TRAFFIC: {traffic.congestion_level.value.upper()}  "
                f"DENSITY: {traffic.traffic_density:.2f}  "
                f"CALIBRATION: {calibration_label}  CAMERA MOTION: {motion_label}  "
                f"POSE: {pose.status.upper() if pose else 'UNKNOWN'}  "
                f"GEOMETRY: {geometry.status.value.upper() if geometry else 'UNKNOWN'}  "
                f"CANDIDATES: "
                f"{'ON' if geometry and geometry.candidate_generation_allowed else 'OFF'}  "
                f"PHYSICAL: {'ON' if permission and permission.allowed else 'OFF'}"
            )
            self._label(output, traffic_label, (10, 24), (55, 55, 55))
            if geometry is not None and not geometry.candidate_generation_allowed:
                reason = _geometry_overlay_reason(geometry.reason_codes)
                self._label(
                    output,
                    f"GEOMETRY JUDGMENTS DISABLED: {reason}",
                    (10, 46),
                    (20, 20, 180),
                )

        observation_by_track = {
            observation.vehicle.track_id: observation for observation in observations
        }
        for observation in observations:
            vehicle = observation.vehicle
            status = statuses.get(vehicle.track_id)
            candidate = status.is_review_candidate if status else False
            color = (20, 20, 230) if candidate else (30, 210, 70)
            x1, y1, x2, y2 = (
                int(vehicle.bbox.x1),
                int(vehicle.bbox.y1),
                int(vehicle.bbox.x2),
                int(vehicle.bbox.y2),
            )
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            duration = status.left_lane_duration_seconds if status else 0.0
            lane = observation.lane_id or "outside"
            decision = (
                "REVIEW"
                if candidate
                else (
                    f"SUP:{status.suppression_reason}"
                    if status and status.suppression_reason
                    else status.behavior_classification
                    if status
                    else ""
                )
            )
            overtake = status.overtake_state if status else "NONE"
            lifecycle = status.candidate_lifecycle_state.upper() if status else "IDLE"
            labels = [
                f"#{vehicle.track_id} {vehicle.class_name} {lane.upper()} {duration:.1f}s"
            ]
            if self._config.show_advanced_debug:
                measurement_parts = []
                if self._config.show_coordinates:
                    measurement_parts.append(
                        status.coordinate_mode if status else "normalized_image"
                    )
                if self._config.show_speed:
                    measurement_parts.append(
                        f"{status.speed_kph:.0f} km/h approx"
                        if status and status.speed_kph is not None
                        else "SPEED:N/A"
                    )
                if self._config.show_gaps:
                    gap_label = _gap_label(status.right_lane_gap if status else None)
                    measurement_parts.append(f"RIGHT GAP:{gap_label}")
                if measurement_parts:
                    labels.append(" ".join(measurement_parts))
                state_parts = [f"OVERTAKE:{overtake}"]
                if self._config.show_lifecycle:
                    state_parts.append(lifecycle)
                state_parts.append(decision)
                labels.append(" ".join(state_parts))
            else:
                labels.append(f"{lifecycle} {decision}")
            self._labels(output, labels, (x1, max(34, y1)), color)

            if status and status.related_track_ids:
                related = observation_by_track.get(status.related_track_ids[0])
                if related is not None:
                    start = tuple(int(value) for value in vehicle.bbox.bottom_center)
                    end = tuple(
                        int(value) for value in related.vehicle.bbox.bottom_center
                    )
                    cv2.line(output, start, end, (220, 180, 60), 1, cv2.LINE_AA)
        return output

    def _labels(
        self,
        image: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x, y = origin
        for offset, line in enumerate(reversed(lines)):
            self._label(image, line, (x, y - offset * 18), color)

    def _label(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        cv2.rectangle(
            image,
            (x, y - height - baseline - 4),
            (x + width + 5, y + 2),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x + 2, y - baseline - 1),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _quality(confidence: float) -> str:
    if confidence >= 0.8:
        return "HIGH"
    if confidence >= 0.5:
        return "MEDIUM"
    return "LOW"


def _gap_label(gap: GapEstimate | None) -> str:
    if gap is None:
        return "CLEAR/UNKNOWN"
    suffix = "m" if gap.unit == "meters" else " norm"
    return f"{gap.value:.2f}{suffix}"


def _geometry_overlay_reason(reason_codes: tuple[str, ...]) -> str:
    for code in (
        "CAMERA_SCALE_CHANGED",
        "PROJECTIVE_DRIFT_DETECTED",
        "FRAME_ASPECT_RATIO_MISMATCH",
        "CAMERA_POSE_UNAVAILABLE",
        "CAMERA_POSE_MOVED",
    ):
        if code in reason_codes:
            return code
    return reason_codes[0] if reason_codes else "UNVERIFIED"

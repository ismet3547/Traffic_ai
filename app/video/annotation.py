"""Debug overlay rendering."""

from __future__ import annotations

import numpy as np

from app.lanes import LaneAssigner
from app.models import LaneObservation, VehicleRuleStatus


class DebugAnnotator:
    def __init__(self, lane_assigner: LaneAssigner) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenCV is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self._cv2 = cv2
        self._lane_assigner = lane_assigner

    def annotate(
        self,
        frame: np.ndarray,
        observations: list[LaneObservation],
        statuses: dict[int, VehicleRuleStatus],
    ) -> np.ndarray:
        cv2 = self._cv2
        output = frame.copy()
        overlay = output.copy()
        height, width = frame.shape[:2]
        polygons = self._lane_assigner.polygons_for_frame(width, height)

        for lane_id, points in polygons.items():
            polygon = np.asarray(points, dtype=np.int32)
            color = (80, 120, 255) if lane_id == self._lane_assigner.leftmost_lane_id else (80, 220, 160)
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
            marker = " REVIEW" if candidate else ""
            label = (
                f"#{vehicle.track_id} {vehicle.class_name} "
                f"lane={lane} left={duration:.1f}s{marker}"
            )
            self._label(output, label, (x1, max(18, y1)), color)
        return output

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

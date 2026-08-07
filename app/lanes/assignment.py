"""Assign tracked vehicles to configured lane polygons."""

from __future__ import annotations

from collections.abc import Sequence

from app.config import LanesConfig
from app.models import LaneObservation, TrackedVehicle

Point = tuple[float, float]


class LaneAssigner:
    def __init__(self, config: LanesConfig) -> None:
        self._config = config

    @property
    def leftmost_lane_id(self) -> str:
        return self._config.leftmost_lane_id

    def polygons_for_frame(
        self, frame_width: int, frame_height: int
    ) -> dict[str, list[Point]]:
        if self._config.coordinate_space == "pixels":
            return {lane.id: list(lane.polygon) for lane in self._config.lanes}
        return {
            lane.id: [(x * frame_width, y * frame_height) for x, y in lane.polygon]
            for lane in self._config.lanes
        }

    def assign(
        self,
        vehicles: Sequence[TrackedVehicle],
        frame_width: int,
        frame_height: int,
    ) -> list[LaneObservation]:
        polygons = self.polygons_for_frame(frame_width, frame_height)
        observations: list[LaneObservation] = []
        for vehicle in vehicles:
            anchor = vehicle.bbox.bottom_center
            lane_id = next(
                (
                    lane.id
                    for lane in self._config.lanes
                    if _point_in_polygon(anchor, polygons[lane.id])
                ),
                None,
            )
            observations.append(LaneObservation(vehicle=vehicle, lane_id=lane_id))
        return observations


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test that treats polygon edges as inside."""

    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-7 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside

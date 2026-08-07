"""ByteTrack adapter using Supervision's stable 0.25-0.27 API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.config import TrackerConfig
from app.models import BoundingBox, Detection, TrackedVehicle


class ByteTrackVehicleTracker:
    def __init__(self, config: TrackerConfig, frame_rate: float) -> None:
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Supervision is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
        )

    def update(self, detections: Sequence[Detection]) -> list[TrackedVehicle]:
        if detections:
            sv_detections = self._sv.Detections(
                xyxy=np.asarray(
                    [
                        [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
                        for d in detections
                    ],
                    dtype=np.float32,
                ),
                confidence=np.asarray([d.confidence for d in detections], dtype=np.float32),
                class_id=np.asarray([d.class_id for d in detections], dtype=int),
                data={
                    "class_name": np.asarray([d.class_name for d in detections])
                },
            )
        else:
            sv_detections = self._sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )

        tracked = self._tracker.update_with_detections(sv_detections)
        if tracked.tracker_id is None:
            return []

        class_names = tracked.data.get("class_name")
        vehicles: list[TrackedVehicle] = []
        for index in range(len(tracked)):
            tracker_id = tracked.tracker_id[index]
            if tracker_id is None:
                continue
            coords = tracked.xyxy[index]
            class_id = int(tracked.class_id[index]) if tracked.class_id is not None else -1
            confidence = (
                float(tracked.confidence[index])
                if tracked.confidence is not None
                else 0.0
            )
            class_name = (
                str(class_names[index])
                if class_names is not None
                else str(class_id)
            )
            vehicles.append(
                TrackedVehicle(
                    track_id=int(tracker_id),
                    bbox=BoundingBox(*(float(value) for value in coords)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return vehicles

import pytest

from app.config import TrafficContextConfig
from app.models import (
    BoundingBox,
    LaneObservation,
    LaneTransition,
    NeighborReference,
    NeighborVehicles,
    RoadPosition,
    TrackedVehicle,
    VehicleTrafficContext,
)
from app.motion import MotionHistoryStore


def test_motion_history_is_bounded_and_retains_context() -> None:
    store = MotionHistoryStore(
        TrafficContextConfig(
            history_seconds=1.0,
            minimum_history_seconds=0.2,
            maximum_samples_per_track=3,
        )
    )
    observation = LaneObservation(
        vehicle=TrackedVehicle(
            track_id=7,
            bbox=BoundingBox(10, 10, 20, 20),
            confidence=0.9,
            class_id=2,
            class_name="car",
        ),
        lane_id="left",
    )
    context = VehicleTrafficContext(
        track_id=7,
        neighbors=NeighborVehicles(adjacent_right_ahead=NeighborReference(8, 0.1)),
        nearby_vehicle_count=1,
        adjacent_right_lane_id="right",
        right_lane_available=True,
        right_lane_available_seconds=1.0,
        right_lane_confidence=0.8,
    )

    for frame_index, timestamp, longitudinal in [
        (0, 0.0, 0.10),
        (1, 0.5, 0.15),
        (2, 1.1, 0.22),
        (3, 1.5, 0.30),
    ]:
        transition = (
            [LaneTransition(7, "right", "left", timestamp)] if frame_index == 1 else []
        )
        store.update(
            frame_index,
            timestamp,
            [observation],
            {
                7: RoadPosition(
                    track_id=7,
                    lateral=0.2,
                    longitudinal=longitudinal,
                    coordinate_system="normalized_image",
                    calibrated=False,
                )
            },
            {7: context},
            transition,
        )

    history = store.history(7)
    assert len(history) == 3
    assert history[0].timestamp_seconds == 0.5
    assert history[-1].estimated_longitudinal_progress == pytest.approx(0.08)
    assert history[-1].neighboring_vehicle_track_ids == (8,)
    assert history[0].lane_transition is not None
    assert store.duration_seconds(7) == 1.0

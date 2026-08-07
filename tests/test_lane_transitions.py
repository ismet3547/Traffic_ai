from app.config import LaneChangeConfig
from app.models import BoundingBox, LaneObservation, TrackedVehicle
from app.motion import LaneTransitionDetector


def _observation(lane_id: str, track_id: int = 17) -> LaneObservation:
    return LaneObservation(
        vehicle=TrackedVehicle(
            track_id=track_id,
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=0.9,
            class_id=2,
            class_name="car",
        ),
        lane_id=lane_id,
    )


def test_lane_transition_requires_time_and_frame_confirmation() -> None:
    detector = LaneTransitionDetector(
        LaneChangeConfig(confirmation_seconds=0.4, minimum_frames=3)
    )
    assert detector.update([_observation("right")], 0.0).transitions == []
    assert detector.update([_observation("left")], 0.1).transitions == []
    assert detector.update([_observation("left")], 0.3).transitions == []

    result = detector.update([_observation("left")], 0.5)

    assert result.observations[0].lane_id == "left"
    assert len(result.transitions) == 1
    assert result.transitions[0].from_lane == "right"
    assert result.transitions[0].to_lane == "left"
    assert result.transitions[0].timestamp_seconds == 0.5


def test_lane_boundary_jitter_does_not_create_repeated_transitions() -> None:
    detector = LaneTransitionDetector(
        LaneChangeConfig(confirmation_seconds=0.3, minimum_frames=3)
    )
    emitted = []
    for timestamp, lane in [
        (0.0, "right"),
        (0.1, "left"),
        (0.2, "right"),
        (0.3, "left"),
        (0.4, "right"),
        (0.5, "left"),
        (0.6, "right"),
    ]:
        emitted.extend(detector.update([_observation(lane)], timestamp).transitions)

    assert emitted == []

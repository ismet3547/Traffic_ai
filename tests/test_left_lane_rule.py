from app.config import LeftLaneRuleConfig
from app.models import BoundingBox, LaneObservation, TrackedVehicle
from app.rules import LeftLaneRuleEngine


def _observation(
    timestamp_track_id: int = 7,
    lane_id: str | None = "left",
    confidence: float = 0.9,
) -> LaneObservation:
    return LaneObservation(
        vehicle=TrackedVehicle(
            track_id=timestamp_track_id,
            bbox=BoundingBox(0, 0, 10, 10),
            confidence=confidence,
            class_id=2,
            class_name="car",
        ),
        lane_id=lane_id,
    )


def test_candidate_starts_at_threshold_and_ends_on_lane_exit() -> None:
    engine = LeftLaneRuleEngine(
        LeftLaneRuleConfig(
            occupancy_threshold_seconds=3.0,
            track_lost_grace_seconds=1.0,
        )
    )

    assert not engine.evaluate([_observation()], 10.0).transitions
    assert not engine.evaluate([_observation()], 12.9).transitions
    at_threshold = engine.evaluate([_observation()], 13.0)

    assert len(at_threshold.transitions) == 1
    start = at_threshold.transitions[0]
    assert start.transition == "started"
    assert start.start_timestamp_seconds == 10.0
    assert start.duration_seconds == 3.0
    assert at_threshold.statuses[7].is_review_candidate

    lane_exit = engine.evaluate([_observation(lane_id="center")], 14.5)
    end = lane_exit.transitions[0]
    assert end.transition == "ended"
    assert end.duration_seconds == 4.5
    assert end.end_reason == "left_lane_exit"


def test_short_occupancy_does_not_create_candidate() -> None:
    engine = LeftLaneRuleEngine(LeftLaneRuleConfig(occupancy_threshold_seconds=5.0))
    engine.evaluate([_observation()], 0.0)
    result = engine.evaluate([_observation(lane_id="center")], 4.9)

    assert result.transitions == []


def test_active_candidate_ends_after_track_lost_grace() -> None:
    engine = LeftLaneRuleEngine(
        LeftLaneRuleConfig(
            occupancy_threshold_seconds=1.0,
            track_lost_grace_seconds=0.5,
        )
    )
    engine.evaluate([_observation()], 0.0)
    engine.evaluate([_observation()], 1.0)

    assert not engine.evaluate([], 1.4).transitions
    result = engine.evaluate([], 1.5)

    assert result.transitions[0].transition == "ended"
    assert result.transitions[0].timestamp_seconds == 1.0
    assert result.transitions[0].end_reason == "track_lost"


def test_low_mean_confidence_is_not_promoted() -> None:
    engine = LeftLaneRuleEngine(
        LeftLaneRuleConfig(
            occupancy_threshold_seconds=1.0,
            minimum_mean_confidence=0.8,
        )
    )
    engine.evaluate([_observation(confidence=0.5)], 0.0)
    result = engine.evaluate([_observation(confidence=0.5)], 2.0)

    assert result.transitions == []
    assert not result.statuses[7].is_review_candidate

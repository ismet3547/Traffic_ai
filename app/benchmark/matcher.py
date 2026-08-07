"""Deterministic one-to-one temporal event matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.benchmark.models import EventMatch, MatchingConfig


class TemporalEvent(Protocol):
    event_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[EventMatch, ...]
    unmatched_ground_truth_ids: tuple[str, ...]
    unmatched_prediction_ids: tuple[str, ...]


def temporal_iou(first: TemporalEvent, second: TemporalEvent) -> float:
    intersection = max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )
    union = max(first.end_seconds, second.end_seconds) - min(
        first.start_seconds, second.start_seconds
    )
    return intersection / union if union > 0 else 0.0


def match_events(
    video_id: str,
    ground_truth: list[TemporalEvent],
    predictions: list[TemporalEvent],
    config: MatchingConfig,
) -> MatchResult:
    """Greedily select highest-IoU pairs with stable tie-breaking.

    A pair must meet the inclusive temporal-IoU threshold and, when configured,
    the inclusive start-time tolerance. Each event can participate in one match.
    Track hints are only enforced when the configuration requests it.
    """

    candidates: list[
        tuple[float, float, float, str, str, TemporalEvent, TemporalEvent]
    ] = []
    for truth in ground_truth:
        for prediction in predictions:
            overlap = temporal_iou(truth, prediction)
            start_error = prediction.start_seconds - truth.start_seconds
            if overlap + 1e-12 < config.minimum_temporal_iou:
                continue
            if (
                config.start_tolerance_seconds is not None
                and abs(start_error) > config.start_tolerance_seconds + 1e-12
            ):
                continue
            if config.require_track_association_if_available and not _tracks_compatible(
                truth, prediction
            ):
                continue
            duration_error = _duration(prediction) - _duration(truth)
            candidates.append(
                (
                    -overlap,
                    abs(start_error),
                    abs(duration_error),
                    prediction.event_id,
                    truth.event_id,
                    truth,
                    prediction,
                )
            )

    matched_truth: set[str] = set()
    matched_predictions: set[str] = set()
    matches: list[EventMatch] = []
    for _, _, _, _, _, truth, prediction in sorted(
        candidates, key=lambda item: item[:5]
    ):
        if (
            truth.event_id in matched_truth
            or prediction.event_id in matched_predictions
        ):
            continue
        matched_truth.add(truth.event_id)
        matched_predictions.add(prediction.event_id)
        matches.append(
            EventMatch(
                video_id=video_id,
                ground_truth_event_id=truth.event_id,
                predicted_event_id=prediction.event_id,
                temporal_iou=temporal_iou(truth, prediction),
                start_time_error_seconds=(
                    prediction.start_seconds - truth.start_seconds
                ),
                duration_error_seconds=_duration(prediction) - _duration(truth),
            )
        )

    matches.sort(key=lambda item: (item.video_id, item.ground_truth_event_id))
    return MatchResult(
        matches=tuple(matches),
        unmatched_ground_truth_ids=tuple(
            sorted(
                event.event_id
                for event in ground_truth
                if event.event_id not in matched_truth
            )
        ),
        unmatched_prediction_ids=tuple(
            sorted(
                event.event_id
                for event in predictions
                if event.event_id not in matched_predictions
            )
        ),
    )


def _duration(event: TemporalEvent) -> float:
    return event.end_seconds - event.start_seconds


def _tracks_compatible(truth: TemporalEvent, prediction: TemporalEvent) -> bool:
    hint = getattr(truth, "vehicle_track_hint", None)
    track_id = getattr(prediction, "track_id", None)
    if hint is None or track_id is None:
        return True
    return str(hint) == str(track_id)

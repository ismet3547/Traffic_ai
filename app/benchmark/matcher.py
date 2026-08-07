"""Deterministic maximum-cardinality, maximum-quality temporal matching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from typing import Protocol

from app.benchmark.models import ControlEventConfig, EventMatch, MatchingConfig


class TemporalEvent(Protocol):
    event_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[EventMatch, ...]
    unmatched_ground_truth_ids: tuple[str, ...]
    unmatched_prediction_ids: tuple[str, ...]


@dataclass(slots=True)
class _FlowEdge:
    target: int
    reverse_index: int
    capacity: int
    cost: float
    truth_id: str | None = None
    prediction_id: str | None = None


def interval_intersection_seconds(first: TemporalEvent, second: TemporalEvent) -> float:
    return max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )


def temporal_iou(first: TemporalEvent, second: TemporalEvent) -> float:
    intersection = interval_intersection_seconds(first, second)
    union = max(first.end_seconds, second.end_seconds) - min(
        first.start_seconds, second.start_seconds
    )
    return intersection / union if union > 0 else 0.0


def prediction_coverage(prediction: TemporalEvent, region: TemporalEvent) -> float:
    duration = prediction.end_seconds - prediction.start_seconds
    return (
        interval_intersection_seconds(prediction, region) / duration
        if duration > 0
        else 0.0
    )


def match_events(
    video_id: str,
    ground_truth: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    config: MatchingConfig,
) -> MatchResult:
    """Optimize valid edges by cardinality, then total IoU, deterministically.

    The residual network is augmented until no source-to-sink path remains, so
    cardinality is maximal. Successive shortest augmenting paths then minimize
    negative IoU cost for that cardinality. Invalid pairs are absent from the
    graph rather than represented by an "almost valid" finite cost.
    """

    eligible = []
    for truth in ground_truth:
        for prediction in predictions:
            rejection = matching_rejection_reason(truth, prediction, config)
            if rejection is None:
                eligible.append(
                    (
                        truth.event_id,
                        prediction.event_id,
                        temporal_iou(truth, prediction),
                    )
                )
    return _optimal_match(video_id, ground_truth, predictions, eligible)


def match_control_events(
    video_id: str,
    controls: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    config: ControlEventConfig,
) -> MatchResult:
    """Match negative controls using explicit coverage and IoU requirements."""

    eligible = []
    for control in controls:
        for prediction in predictions:
            coverage = prediction_coverage(prediction, control)
            overlap = temporal_iou(control, prediction)
            if (
                coverage + 1e-12 >= config.minimum_prediction_coverage
                and overlap + 1e-12 >= config.minimum_temporal_iou
            ):
                eligible.append((control.event_id, prediction.event_id, overlap))
    return _optimal_match(video_id, controls, predictions, eligible)


def matching_rejection_reason(
    truth: TemporalEvent,
    prediction: TemporalEvent,
    config: MatchingConfig,
) -> str | None:
    overlap = temporal_iou(truth, prediction)
    if overlap + 1e-12 < config.minimum_temporal_iou:
        return "TEMPORAL_IOU_BELOW_THRESHOLD"
    start_error = prediction.start_seconds - truth.start_seconds
    if (
        config.start_tolerance_seconds is not None
        and abs(start_error) > config.start_tolerance_seconds + 1e-12
    ):
        return "START_TOLERANCE_EXCEEDED"
    if config.require_track_association_if_available and not _tracks_compatible(
        truth, prediction
    ):
        return "TRACK_ASSOCIATION_MISMATCH"
    return None


def _optimal_match(
    video_id: str,
    ground_truth: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    eligible: list[tuple[str, str, float]],
) -> MatchResult:
    truths = sorted(ground_truth, key=lambda event: event.event_id)
    predicted = sorted(predictions, key=lambda event: event.event_id)
    truth_by_id = {event.event_id: event for event in truths}
    prediction_by_id = {event.event_id: event for event in predicted}
    if len(truth_by_id) != len(truths) or len(prediction_by_id) != len(predicted):
        raise ValueError("matching requires unique event IDs on each bipartite side")

    source = 0
    truth_offset = 1
    prediction_offset = truth_offset + len(truths)
    sink = prediction_offset + len(predicted)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    truth_node = {
        event.event_id: truth_offset + index for index, event in enumerate(truths)
    }
    prediction_node = {
        event.event_id: prediction_offset + index
        for index, event in enumerate(predicted)
    }
    for event in truths:
        _add_edge(graph, source, truth_node[event.event_id], 0.0)
    for event in predicted:
        _add_edge(graph, prediction_node[event.event_id], sink, 0.0)
    for truth_id, prediction_id, overlap in sorted(
        eligible, key=lambda item: (item[0], item[1])
    ):
        _add_edge(
            graph,
            truth_node[truth_id],
            prediction_node[prediction_id],
            -overlap,
            truth_id=truth_id,
            prediction_id=prediction_id,
        )

    while _augment_shortest_path(graph, source, sink):
        pass

    matched_pairs = []
    for node in (truth_node[event.event_id] for event in truths):
        for edge in graph[node]:
            if (
                edge.truth_id is not None
                and edge.prediction_id is not None
                and edge.capacity == 0
            ):
                matched_pairs.append((edge.truth_id, edge.prediction_id))
    matched_truth = {truth_id for truth_id, _ in matched_pairs}
    matched_predictions = {prediction_id for _, prediction_id in matched_pairs}
    matches = [
        _event_match(
            video_id,
            truth_by_id[truth_id],
            prediction_by_id[prediction_id],
        )
        for truth_id, prediction_id in sorted(matched_pairs)
    ]
    return MatchResult(
        matches=tuple(matches),
        unmatched_ground_truth_ids=tuple(
            event.event_id for event in truths if event.event_id not in matched_truth
        ),
        unmatched_prediction_ids=tuple(
            event.event_id
            for event in predicted
            if event.event_id not in matched_predictions
        ),
    )


def _add_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    cost: float,
    *,
    truth_id: str | None = None,
    prediction_id: str | None = None,
) -> None:
    forward = _FlowEdge(
        target=target,
        reverse_index=len(graph[target]),
        capacity=1,
        cost=cost,
        truth_id=truth_id,
        prediction_id=prediction_id,
    )
    reverse = _FlowEdge(
        target=source,
        reverse_index=len(graph[source]),
        capacity=0,
        cost=-cost,
    )
    graph[source].append(forward)
    graph[target].append(reverse)


def _augment_shortest_path(
    graph: list[list[_FlowEdge]], source: int, sink: int
) -> bool:
    distances = [inf] * len(graph)
    previous: list[tuple[int, int] | None] = [None] * len(graph)
    distances[source] = 0.0
    for _ in range(len(graph) - 1):
        changed = False
        for node, edges in enumerate(graph):
            if distances[node] == inf:
                continue
            for edge_index, edge in enumerate(edges):
                if edge.capacity <= 0:
                    continue
                candidate = distances[node] + edge.cost
                if candidate < distances[edge.target] - 1e-12:
                    distances[edge.target] = candidate
                    previous[edge.target] = (node, edge_index)
                    changed = True
        if not changed:
            break
    if previous[sink] is None:
        return False
    node = sink
    while node != source:
        step = previous[node]
        if step is None:  # pragma: no cover - guarded by complete predecessor path
            raise RuntimeError("incomplete augmenting path")
        previous_node, edge_index = step
        edge = graph[previous_node][edge_index]
        edge.capacity -= 1
        graph[node][edge.reverse_index].capacity += 1
        node = previous_node
    return True


def _event_match(
    video_id: str, truth: TemporalEvent, prediction: TemporalEvent
) -> EventMatch:
    return EventMatch(
        video_id=video_id,
        ground_truth_event_id=truth.event_id,
        predicted_event_id=prediction.event_id,
        temporal_iou=temporal_iou(truth, prediction),
        start_time_error_seconds=prediction.start_seconds - truth.start_seconds,
        duration_error_seconds=_duration(prediction) - _duration(truth),
    )


def _duration(event: TemporalEvent) -> float:
    return event.end_seconds - event.start_seconds


def _tracks_compatible(truth: TemporalEvent, prediction: TemporalEvent) -> bool:
    hint = getattr(truth, "vehicle_track_hint", None)
    track_id = getattr(prediction, "track_id", None)
    if hint is None or track_id is None:
        return True
    return str(hint) == str(track_id)

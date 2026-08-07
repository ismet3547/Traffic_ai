"""Deterministic, source-group-isolated dataset split assignment."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from app.benchmark.models import DatasetSplit
from app.dataset.models import (
    SplitAssignment,
    SplitAssignmentDocument,
    SplitCandidate,
)

SPLITS = (DatasetSplit.DEVELOPMENT, DatasetSplit.VALIDATION, DatasetSplit.TEST)


def assign_group_aware_splits(
    candidates: list[SplitCandidate],
    *,
    target_ratios: dict[DatasetSplit, float] | None = None,
    seed: int = 42,
) -> SplitAssignmentDocument:
    ratios = target_ratios or {
        DatasetSplit.DEVELOPMENT: 0.50,
        DatasetSplit.VALIDATION: 0.25,
        DatasetSplit.TEST: 0.25,
    }
    _validate_inputs(candidates, ratios)
    groups: dict[str, list[SplitCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.source_group_id].append(candidate)
    group_order = sorted(
        groups,
        key=lambda group_id: (
            -sum(item.duration_seconds for item in groups[group_id]),
            _stable_tie(seed, group_id),
            group_id,
        ),
    )
    total_duration = sum(item.duration_seconds for item in candidates)
    total_labels = Counter(
        label.value for item in candidates for label in set(item.labels)
    )
    total_tags = Counter(tag for item in candidates for tag in set(item.scenario_tags))
    durations: dict[DatasetSplit, float] = {split: 0.0 for split in SPLITS}
    label_counts: dict[DatasetSplit, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    tag_counts: dict[DatasetSplit, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    group_split: dict[str, DatasetSplit] = {}
    for group_id in group_order:
        members = groups[group_id]
        group_duration = sum(item.duration_seconds for item in members)
        group_labels = Counter(
            label.value for item in members for label in set(item.labels)
        )
        group_tags = Counter(tag for item in members for tag in set(item.scenario_tags))
        selected = min(
            SPLITS,
            key=lambda split: (
                _assignment_cost(
                    split,
                    ratios,
                    total_duration,
                    total_labels,
                    total_tags,
                    durations,
                    label_counts,
                    tag_counts,
                    group_duration,
                    group_labels,
                    group_tags,
                ),
                _stable_tie(seed, f"{group_id}:{split.value}"),
                split.value,
            ),
        )
        group_split[group_id] = selected
        durations[selected] += group_duration
        label_counts[selected].update(group_labels)
        tag_counts[selected].update(group_tags)
    assignments = [
        SplitAssignment(
            video_id=item.video_id,
            source_group_id=item.source_group_id,
            split=group_split[item.source_group_id],
        )
        for item in sorted(candidates, key=lambda candidate: candidate.video_id)
    ]
    return SplitAssignmentDocument(
        seed=seed,
        target_ratios=ratios,
        assignments=assignments,
    )


def _assignment_cost(
    split: DatasetSplit,
    ratios: dict[DatasetSplit, float],
    total_duration: float,
    total_labels: Counter[str],
    total_tags: Counter[str],
    durations: dict[DatasetSplit, float],
    label_counts: dict[DatasetSplit, Counter[str]],
    tag_counts: dict[DatasetSplit, Counter[str]],
    group_duration: float,
    group_labels: Counter[str],
    group_tags: Counter[str],
) -> float:
    ratio = ratios[split]
    duration_target = total_duration * ratio
    cost = abs(durations[split] + group_duration - duration_target) / max(
        duration_target, 1.0
    )
    for label, total in total_labels.items():
        target = total * ratio
        cost += (
            0.35
            * abs(label_counts[split][label] + group_labels[label] - target)
            / max(target, 1.0)
        )
    for tag, total in total_tags.items():
        target = total * ratio
        cost += (
            0.15
            * abs(tag_counts[split][tag] + group_tags[tag] - target)
            / max(target, 1.0)
        )
    return cost


def _validate_inputs(
    candidates: list[SplitCandidate], ratios: dict[DatasetSplit, float]
) -> None:
    identifiers = [item.video_id for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("split candidate video IDs must be unique")
    if set(ratios) != set(SPLITS):
        raise ValueError("target ratios must define development, validation, and test")
    if (
        any(value < 0 for value in ratios.values())
        or abs(sum(ratios.values()) - 1) > 1e-9
    ):
        raise ValueError("target split ratios must be non-negative and sum to 1")


def _stable_tie(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

"""Orchestrate conservative cached-prediction evaluation without inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.benchmark.agreement import compare_annotations
from app.benchmark.annotations import load_video_annotations
from app.benchmark.diagnostics import (
    diagnose_false_negative,
    diagnose_false_positive,
)
from app.benchmark.duration import resolve_video_duration
from app.benchmark.metrics import (
    EventEvaluation,
    combine_accounting,
    combine_evaluations,
    evaluate_events,
    policy_specific_metrics,
)
from app.benchmark.models import (
    ANNOTATION_ROLE_BY_LABEL,
    AnnotationConfidence,
    AnnotationRole,
    BenchmarkManifest,
    DurationValidationResult,
    FailureRecord,
    ManifestVideo,
    PredictionDocument,
    RuntimePerformance,
    annotation_role,
)


def evaluate_benchmark(
    manifest_path: str | Path,
    manifest: BenchmarkManifest,
    predictions: dict[str, PredictionDocument],
    *,
    videos: list[ManifestVideo] | None = None,
    reproducibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_videos = sorted(
        videos or [video for video in manifest.videos if video.enabled],
        key=lambda video: video.id,
    )
    headline_confidences = set(manifest.benchmark.headline_confidences)
    evaluations: dict[str, EventEvaluation] = {}
    annotations_by_video: dict[str, list[Any]] = {}
    documents_by_video: dict[str, list[Any]] = {}
    durations_by_video: dict[str, DurationValidationResult] = {}
    per_video: dict[str, Any] = {}
    failures: list[FailureRecord] = []
    ignored_predictions: list[dict[str, Any]] = []
    ignored_ground_truth: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    agreement = []

    for video in selected_videos:
        if video.id not in predictions:
            raise ValueError(f"prediction cache missing video ID: {video.id}")
        documents = load_video_annotations(manifest_path, video)
        primary = documents[0]
        annotations = primary.events
        documents_by_video[video.id] = documents
        annotations_by_video[video.id] = annotations
        duration = resolve_video_duration(
            manifest_path,
            video,
            primary,
            predictions[video.id],
            manifest.benchmark.duration_validation,
        )
        durations_by_video[video.id] = duration
        positive_events = [
            event
            for event in annotations
            if annotation_role(event.label) == AnnotationRole.POSITIVE
        ]
        selected_positive = [
            event
            for event in positive_events
            if event.confidence in headline_confidences
        ]
        non_headline_positive = [
            event
            for event in positive_events
            if event.confidence not in headline_confidences
        ]
        semantic_ignore_regions = [
            event
            for event in annotations
            if annotation_role(event.label) == AnnotationRole.IGNORE_REGION
        ]
        selected_controls = [
            event
            for event in annotations
            if annotation_role(event.label) == AnnotationRole.NEGATIVE_CONTROL
            and event.confidence in headline_confidences
        ]
        evaluation = evaluate_events(
            video.id,
            selected_positive,
            predictions[video.id].predictions,
            manifest.benchmark.matching,
            duration,
            ignored_annotations=semantic_ignore_regions,
            ignored_ground_truth=non_headline_positive,
            ignored_region_config=manifest.benchmark.ignored_regions,
            minimum_prediction_confidence=(
                manifest.benchmark.minimum_prediction_confidence
            ),
        )
        evaluations[video.id] = evaluation
        policy = policy_specific_metrics(
            selected_controls,
            list(evaluation.considered_predictions),
            manifest.benchmark.control_events,
        )
        _add_policy_counts(policy_counts, policy)
        per_video[video.id] = {
            "tags": sorted(video.tags),
            "split": video.split.value,
            "duration_seconds": duration.duration_seconds_used,
            "duration_validation": duration.model_dump(mode="json"),
            "metrics": evaluation.metrics.model_dump(mode="json"),
            "accounting": evaluation.accounting.model_dump(mode="json"),
            "matches": [item.model_dump(mode="json") for item in evaluation.matches],
            "ignored_predictions": [
                item.model_dump(mode="json") for item in evaluation.ignored_predictions
            ],
            "filtered_predictions": [
                item.model_dump(mode="json") for item in evaluation.filtered_predictions
            ],
            "ignored_ground_truth_events": [
                {
                    "event_id": event.event_id,
                    "label": event.label.value,
                    "confidence": event.confidence.value,
                    "reason": "positive annotation confidence excluded from headline",
                }
                for event in evaluation.ignored_ground_truth
            ],
            "matching_diagnostics": evaluation.matching_diagnostics,
            "policy_metrics": policy,
            "annotation_event_counts": _annotation_counts(annotations),
        }
        ignored_predictions.extend(
            {
                "video_id": video.id,
                **item.model_dump(mode="json"),
            }
            for item in evaluation.ignored_predictions
        )
        ignored_ground_truth.extend(
            {
                "video_id": video.id,
                "event_id": event.event_id,
                "label": event.label.value,
                "confidence": event.confidence.value,
                "reason": "positive annotation confidence excluded from headline",
            }
            for event in evaluation.ignored_ground_truth
        )
        failures.extend(
            diagnose_false_positive(
                prediction,
                annotations,
                sequence,
                evaluation.matching_diagnostics.get(prediction.event_id),
            )
            for sequence, prediction in enumerate(evaluation.false_positives, start=1)
        )
        failures.extend(
            diagnose_false_negative(
                truth,
                annotations,
                video.id,
                sequence,
                evaluation.matching_diagnostics.get(f"ground_truth:{truth.event_id}"),
            )
            for sequence, truth in enumerate(evaluation.false_negatives, start=1)
        )
        if len(documents) >= 2:
            for first_index in range(len(documents) - 1):
                for second_index in range(first_index + 1, len(documents)):
                    agreement.append(
                        compare_annotations(
                            documents[first_index],
                            documents[second_index],
                            manifest.benchmark.matching,
                        ).model_dump(mode="json")
                    )

    overall = combine_evaluations(list(evaluations.values()))
    overall_accounting = combine_accounting(list(evaluations.values()))
    scenario_metrics = _scenario_metrics(selected_videos, evaluations)
    confidence_metrics = _confidence_metrics(
        selected_videos,
        annotations_by_video,
        predictions,
        manifest,
        durations_by_video,
    )
    performance = _performance_summary(selected_videos, predictions)
    policy_metrics = _finish_policy_metrics(policy_counts)
    failure_breakdown = dict(
        sorted(Counter(item.suspected_failure_category for item in failures).items())
    )
    acceptance = _acceptance(
        overall.model_dump(mode="json"), manifest, durations_by_video
    )
    return {
        "benchmark_schema_version": "1.0",
        "report_title": (
            "SYNTHETIC INTEGRITY TEST - NOT REAL-WORLD PERFORMANCE"
            if manifest.synthetic
            else manifest.name
        ),
        "synthetic": manifest.synthetic,
        "accuracy_claim": (
            "Synthetic integrity fixture only; this is not measured real-world accuracy."
            if manifest.synthetic
            else "Metrics describe only the annotated videos in this manifest."
        ),
        "headline_annotation_confidences": sorted(
            confidence.value for confidence in headline_confidences
        ),
        "prediction_confidence_threshold": (
            manifest.benchmark.minimum_prediction_confidence
        ),
        "annotation_roles": {
            label.value: role.value
            for label, role in sorted(
                ANNOTATION_ROLE_BY_LABEL.items(), key=lambda item: item[0].value
            )
        },
        "metric_definitions": {
            "precision": "TP / (TP + FP); zero when the denominator is zero",
            "recall": "TP / (TP + FN); zero when the denominator is zero",
            "f1": "2 * precision * recall / (precision + recall); zero when both are zero",
            "temporal_iou": "intersection duration / interval union duration",
            "prediction_coverage": "intersection duration / prediction duration",
            "event_matching": "maximum valid cardinality, then maximum total temporal IoU",
            "control_overlap": "both configured prediction coverage and temporal IoU must pass",
            "real_time_factor": "video duration / processing time; values above 1 are faster than real time",
        },
        "overall_metrics": overall.model_dump(mode="json"),
        "accounting": overall_accounting.model_dump(mode="json"),
        "ignored_predictions": sorted(
            ignored_predictions,
            key=lambda item: (item["video_id"], item["prediction_id"]),
        ),
        "ignored_ground_truth_events": sorted(
            ignored_ground_truth,
            key=lambda item: (item["video_id"], item["event_id"]),
        ),
        "per_video_metrics": per_video,
        "scenario_metrics": scenario_metrics,
        "confidence_stratified_metrics": confidence_metrics,
        "policy_specific_metrics": policy_metrics,
        "failure_breakdown": failure_breakdown,
        "failures": [
            item.model_dump(mode="json")
            for item in sorted(
                failures,
                key=lambda item: (item.video_id, item.kind, item.failure_id),
            )
        ],
        "annotation_agreement": sorted(
            agreement,
            key=lambda item: (
                item["video_id"],
                item["annotator_a"],
                item["annotator_b"],
            ),
        ),
        "performance_metrics": performance,
        "acceptance": acceptance,
        "reproducibility": reproducibility or {},
    }


def _annotation_counts(annotations: list[Any]) -> dict[str, dict[str, int]]:
    by_confidence = Counter(event.confidence.value for event in annotations)
    by_label = Counter(event.label.value for event in annotations)
    by_role = Counter(annotation_role(event.label).value for event in annotations)
    return {
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_role": dict(sorted(by_role.items())),
    }


def _scenario_metrics(
    videos: list[ManifestVideo], evaluations: dict[str, EventEvaluation]
) -> dict[str, Any]:
    grouped: dict[str, list[EventEvaluation]] = defaultdict(list)
    for video in videos:
        for tag in video.tags:
            grouped[tag].append(evaluations[video.id])
    return {
        tag: combine_evaluations(group).model_dump(mode="json")
        for tag, group in sorted(grouped.items())
    }


def _confidence_metrics(
    videos: list[ManifestVideo],
    annotations: dict[str, list[Any]],
    predictions: dict[str, PredictionDocument],
    manifest: BenchmarkManifest,
    durations: dict[str, DurationValidationResult],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for confidence in AnnotationConfidence:
        evaluations = []
        for video in videos:
            positive_events = [
                event
                for event in annotations[video.id]
                if annotation_role(event.label) == AnnotationRole.POSITIVE
            ]
            selected = [
                event for event in positive_events if event.confidence == confidence
            ]
            non_selected = [
                event for event in positive_events if event.confidence != confidence
            ]
            semantic_ignore = [
                event
                for event in annotations[video.id]
                if annotation_role(event.label) == AnnotationRole.IGNORE_REGION
            ]
            evaluations.append(
                evaluate_events(
                    video.id,
                    selected,
                    predictions[video.id].predictions,
                    manifest.benchmark.matching,
                    durations[video.id],
                    ignored_annotations=semantic_ignore,
                    ignored_ground_truth=non_selected,
                    ignored_region_config=manifest.benchmark.ignored_regions,
                    minimum_prediction_confidence=(
                        manifest.benchmark.minimum_prediction_confidence
                    ),
                )
            )
        output[confidence.value] = combine_evaluations(evaluations).model_dump(
            mode="json"
        )
    return output


def _add_policy_counts(target: Counter[str], values: dict[str, float | int]) -> None:
    for key, value in values.items():
        if key.endswith(("_events", "_count")):
            target[key] += int(value)


def _finish_policy_metrics(counts: Counter[str]) -> dict[str, float | int]:
    output: dict[str, float | int] = dict(sorted(counts.items()))
    prefixes = sorted(
        key.removesuffix("_control_events")
        for key in counts
        if key.endswith("_control_events")
    )
    for prefix in prefixes:
        controls = counts[f"{prefix}_control_events"]
        failures = counts[f"{prefix}_false_positive_count"]
        output[f"{prefix}_false_positive_rate"] = (
            failures / controls if controls else 0.0
        )
        output[f"{prefix}_suppression_success_rate"] = (
            (controls - failures) / controls if controls else 0.0
        )
    output["geometry_fail_closed_success_rate"] = output.get(
        "geometry_suppression_success_rate", 0.0
    )
    return output


def _performance_summary(
    videos: list[ManifestVideo], predictions: dict[str, PredictionDocument]
) -> dict[str, Any]:
    records: list[RuntimePerformance] = []
    per_video: dict[str, Any] = {}
    for video in videos:
        performance = predictions[video.id].performance
        if performance is not None:
            records.append(performance)
            per_video[video.id] = performance.model_dump(mode="json")
    if not records:
        return {
            "available": False,
            "measurement_scope": "unavailable_in_prediction_cache",
            "per_video": {},
        }
    total_time = sum(item.total_processing_time_seconds for item in records)
    total_duration = sum(item.video_duration_seconds for item in records)
    total_frames = sum(item.frames_processed for item in records)
    return {
        "available": True,
        "measurement_scope": "end_to_end_inference_only",
        "total_processing_time_seconds": total_time,
        "video_duration_seconds": total_duration,
        "frames_processed": total_frames,
        "processing_fps": total_frames / total_time if total_time else 0.0,
        "real_time_factor": total_duration / total_time if total_time else 0.0,
        "average_frame_processing_time_ms": (
            total_time * 1000.0 / total_frames if total_frames else 0.0
        ),
        "per_video": per_video,
    }


def _acceptance(
    metrics: dict[str, Any],
    manifest: BenchmarkManifest,
    durations: dict[str, DurationValidationResult],
) -> dict[str, Any]:
    criteria = manifest.benchmark.acceptance
    checks = []
    if criteria.minimum_precision is not None:
        checks.append(
            {
                "metric": "precision",
                "operator": ">=",
                "threshold": criteria.minimum_precision,
                "actual": metrics["precision"],
                "passed": metrics["precision"] >= criteria.minimum_precision,
            }
        )
    if criteria.minimum_recall is not None:
        checks.append(
            {
                "metric": "recall",
                "operator": ">=",
                "threshold": criteria.minimum_recall,
                "actual": metrics["recall"],
                "passed": metrics["recall"] >= criteria.minimum_recall,
            }
        )
    if criteria.maximum_false_positives_per_hour is not None:
        actual = metrics["false_positives_per_video_hour"]
        checks.append(
            {
                "metric": "false_positives_per_video_hour",
                "operator": "<=",
                "threshold": criteria.maximum_false_positives_per_hour,
                "actual": actual,
                "passed": actual <= criteria.maximum_false_positives_per_hour,
            }
        )
    if (
        checks
        and manifest.benchmark.duration_validation.require_multiple_sources_for_acceptance
    ):
        unverified = sorted(
            video_id
            for video_id, duration in durations.items()
            if duration.duration_validation_status == "single_source_unverified"
        )
        checks.append(
            {
                "metric": "duration_denominator_validation",
                "operator": "multiple_consistent_sources_or_video_metadata",
                "threshold": 0,
                "actual": len(unverified),
                "passed": not unverified,
                "unverified_video_ids": unverified,
            }
        )
    return {
        "configured": bool(checks),
        "passed": all(item["passed"] for item in checks) if checks else None,
        "checks": checks,
        "note": "No production-readiness threshold is assumed when criteria are null.",
    }

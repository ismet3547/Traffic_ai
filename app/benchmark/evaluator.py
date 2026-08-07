"""Orchestrate cached-prediction evaluation without invoking inference."""

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
from app.benchmark.metrics import (
    EventEvaluation,
    combine_evaluations,
    evaluate_events,
    policy_specific_metrics,
)
from app.benchmark.models import (
    AnnotationConfidence,
    BenchmarkManifest,
    ManifestVideo,
    PredictionDocument,
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
    per_video: dict[str, Any] = {}
    failures = []
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
        duration = _video_duration(
            video, primary.video_duration_seconds, predictions[video.id]
        )
        selected = [
            event for event in annotations if event.confidence in headline_confidences
        ]
        ignored = [
            event
            for event in annotations
            if event.confidence not in headline_confidences
        ]
        evaluation = evaluate_events(
            video.id,
            selected,
            predictions[video.id].predictions,
            manifest.benchmark.matching,
            duration,
            ignored_annotations=ignored,
        )
        evaluations[video.id] = evaluation
        policy = policy_specific_metrics(
            selected,
            predictions[video.id].predictions,
            manifest.benchmark.matching,
        )
        _add_policy_counts(policy_counts, policy)
        per_video[video.id] = {
            "tags": sorted(video.tags),
            "split": video.split.value,
            "duration_seconds": duration,
            "metrics": evaluation.metrics.model_dump(mode="json"),
            "matches": [item.model_dump(mode="json") for item in evaluation.matches],
            "ignored_prediction_ids": list(evaluation.ignored_prediction_ids),
            "policy_metrics": policy,
            "annotation_event_counts": _annotation_counts(annotations),
        }
        fp_failures = [
            diagnose_false_positive(prediction, annotations, sequence)
            for sequence, prediction in enumerate(evaluation.false_positives, start=1)
        ]
        fn_failures = [
            diagnose_false_negative(truth, annotations, video.id, sequence)
            for sequence, truth in enumerate(evaluation.false_negatives, start=1)
        ]
        failures.extend(fp_failures)
        failures.extend(fn_failures)
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
    scenario_metrics = _scenario_metrics(selected_videos, evaluations)
    confidence_metrics = _confidence_metrics(
        selected_videos,
        annotations_by_video,
        predictions,
        manifest,
        documents_by_video,
    )
    performance = _performance_summary(selected_videos, predictions)
    policy_metrics = _finish_policy_metrics(policy_counts)
    failure_breakdown = dict(
        sorted(Counter(item.suspected_failure_category for item in failures).items())
    )
    acceptance = _acceptance(overall.model_dump(mode="json"), manifest)
    return {
        "benchmark_schema_version": "1.0",
        "report_title": (
            "SYNTHETIC EXAMPLE - NOT REAL-WORLD PERFORMANCE"
            if manifest.synthetic
            else manifest.name
        ),
        "synthetic": manifest.synthetic,
        "accuracy_claim": (
            "Synthetic fixture validation only; this is not measured real-world accuracy."
            if manifest.synthetic
            else "Metrics describe only the annotated videos in this manifest."
        ),
        "headline_annotation_confidences": sorted(
            confidence.value for confidence in headline_confidences
        ),
        "metric_definitions": {
            "precision": "TP / (TP + FP); zero when the denominator is zero",
            "recall": "TP / (TP + FN); zero when the denominator is zero",
            "f1": "2 * precision * recall / (precision + recall); zero when both are zero",
            "temporal_iou": "intersection duration / union duration",
            "real_time_factor": "video duration / processing time; values above 1 are faster than real time",
        },
        "overall_metrics": overall.model_dump(mode="json"),
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


def _video_duration(
    video: ManifestVideo,
    annotation_duration: float | None,
    prediction: PredictionDocument,
) -> float:
    values = (
        video.duration_seconds,
        annotation_duration,
        (
            prediction.performance.video_duration_seconds
            if prediction.performance is not None
            else None
        ),
    )
    for value in values:
        if value is not None and value > 0:
            return value
    raise ValueError(
        f"video {video.id!r} needs duration_seconds in the manifest, annotation, "
        "or prediction performance metadata"
    )


def _annotation_counts(annotations: list[Any]) -> dict[str, dict[str, int]]:
    by_confidence = Counter(event.confidence.value for event in annotations)
    by_label = Counter(event.label.value for event in annotations)
    return {
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_label": dict(sorted(by_label.items())),
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
    documents: dict[str, list[Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for confidence in AnnotationConfidence:
        evaluations = []
        for video in videos:
            selected = [
                event
                for event in annotations[video.id]
                if event.confidence == confidence
            ]
            ignored = [
                event
                for event in annotations[video.id]
                if event.confidence != confidence
            ]
            duration = _video_duration(
                video,
                documents[video.id][0].video_duration_seconds,
                predictions[video.id],
            )
            evaluations.append(
                evaluate_events(
                    video.id,
                    selected,
                    predictions[video.id].predictions,
                    manifest.benchmark.matching,
                    duration,
                    ignored_annotations=ignored,
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
    records = [
        predictions[video.id].performance
        for video in videos
        if predictions[video.id].performance is not None
    ]
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
        "per_video": {
            video.id: predictions[video.id].performance.model_dump(mode="json")
            for video in videos
            if predictions[video.id].performance is not None
        },
    }


def _acceptance(metrics: dict[str, Any], manifest: BenchmarkManifest) -> dict[str, Any]:
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
    return {
        "configured": bool(checks),
        "passed": all(item["passed"] for item in checks) if checks else None,
        "checks": checks,
        "note": "No production-readiness threshold is assumed when criteria are null.",
    }

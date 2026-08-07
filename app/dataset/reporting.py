"""Actual dataset coverage aggregation without invented balance claims."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from app.dataset.agreement_integrity import validate_supplied_agreements
from app.dataset.io import write_json_model
from app.dataset.models import (
    AgreementReport,
    CoverageReport,
    DatasetAnnotation,
    IntakeRegistry,
)


def build_coverage_report(
    registry: IntakeRegistry,
    annotations: dict[str, list[DatasetAnnotation]],
    agreements: list[AgreementReport] | None = None,
) -> CoverageReport:
    labels: Counter[str] = Counter()
    confidences: Counter[str] = Counter()
    scenario_tags: Counter[str] = Counter()
    day_night: Counter[str] = Counter()
    traffic: Counter[str] = Counter()
    cameras: Counter[str] = Counter()
    vehicle_classes: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for record in registry.videos:
        scenario_tags.update(record.scenario_tags)
        day_night[_category(record.scenario_tags, ("daylight", "night"))] += 1
        traffic[
            _category(
                record.scenario_tags,
                ("free_flow", "moderate_traffic", "dense_traffic", "stop_and_go"),
            )
        ] += 1
        cameras[
            _category(
                record.scenario_tags,
                ("fixed_camera", "camera_motion", "moving_camera"),
            )
        ] += 1
        vehicle_classes.update(item.value for item in record.vehicle_classes)
        splits[record.split.value if record.split is not None else "unassigned"] += 1
        for document in annotations.get(record.video_id, []):
            for event in document.events:
                labels[event.label.value] += 1
                confidences[event.confidence.value] += 1
                if event.evidence.vehicle_class is not None:
                    vehicle_classes[event.evidence.vehicle_class.value] += 1
    agreement_values = validate_supplied_agreements(
        registry, annotations, agreements or []
    )
    positive_event_reports = [
        item
        for item in agreement_values
        if item.annotation_a_event_count + item.annotation_b_event_count > 0
    ]
    agreement_statistics: dict[str, float | int | str | None] = {
        "report_count": len(agreement_values),
        "zero_event_both_annotators_video_count": (
            len(agreement_values) - len(positive_event_reports)
        ),
        "positive_event_video_count": len(positive_event_reports),
        "agreement_mode": (
            agreement_values[0].agreement_mode.value if agreement_values else None
        ),
        "agreement_protocol_version": (
            agreement_values[0].agreement_protocol_version if agreement_values else None
        ),
        "agreement_config_fingerprint": (
            agreement_values[0].agreement_config_fingerprint
            if agreement_values
            else None
        ),
        "mean_event_detection_agreement": (
            mean(item.event_detection_agreement for item in agreement_values)
            if agreement_values
            else None
        ),
        "mean_label_agreement": (
            mean(item.label_agreement for item in agreement_values)
            if agreement_values
            else None
        ),
        "positive_event_mean_event_detection_agreement": (
            mean(item.event_detection_agreement for item in positive_event_reports)
            if positive_event_reports
            else None
        ),
        "mean_temporal_boundary_agreement": (
            mean(item.temporal_boundary_agreement for item in agreement_values)
            if agreement_values
            else None
        ),
        "total_disagreements": sum(
            item.disagreement_count for item in agreement_values
        ),
    }
    return CoverageReport(
        total_clips=len(registry.videos),
        total_duration_seconds=sum(item.duration_seconds for item in registry.videos),
        labels=sorted(labels),
        label_counts=dict(sorted(labels.items())),
        confidence_counts=dict(sorted(confidences.items())),
        scenario_tag_counts=dict(sorted(scenario_tags.items())),
        day_night_distribution=dict(sorted(day_night.items())),
        traffic_density_distribution=dict(sorted(traffic.items())),
        camera_configuration_distribution=dict(sorted(cameras.items())),
        vehicle_class_distribution=dict(sorted(vehicle_classes.items())),
        split_counts=dict(sorted(splits.items())),
        agreement_statistics=agreement_statistics,
    )


def write_coverage_report(
    report: CoverageReport, output_directory: str | Path
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = write_json_model(report, output / "dataset_coverage.json")
    markdown_path = output / "dataset_coverage.md"
    markdown_path.write_text(coverage_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def coverage_markdown(report: CoverageReport) -> str:
    lines = [
        "# Dataset Coverage",
        "",
        "Counts below describe actual registered metadata and supplied annotations; imbalance is not hidden.",
        "",
        f"- Total clips: {report.total_clips}",
        f"- Total duration: {report.total_duration_seconds:.3f} seconds",
        f"- Observed labels: {', '.join(report.labels) if report.labels else 'none'}",
    ]
    for title, values in (
        ("Labels", report.label_counts),
        ("Confidence", report.confidence_counts),
        ("Scenario tags", report.scenario_tag_counts),
        ("Day/night", report.day_night_distribution),
        ("Traffic density", report.traffic_density_distribution),
        ("Camera configuration", report.camera_configuration_distribution),
        ("Vehicle classes", report.vehicle_class_distribution),
        ("Splits", report.split_counts),
        ("Agreement", report.agreement_statistics),
    ):
        lines.extend(["", f"## {title}", ""])
        if values:
            lines.extend(f"- `{key}`: {value}" for key, value in sorted(values.items()))
        else:
            lines.append("No data recorded.")
    lines.append("")
    return "\n".join(lines)


def _category(tags: list[str], options: tuple[str, ...]) -> str:
    for option in options:
        if option in tags:
            return option
    return "unknown"

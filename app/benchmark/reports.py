"""Deterministic JSON/Markdown reports, baselines, and failure bundles."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from app.benchmark.annotations import resolve_manifest_path
from app.benchmark.models import BaselineTolerances, ManifestVideo


def write_reports(
    report: dict[str, Any], output_directory: str | Path
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "benchmark_report.json"
    markdown_path = output / "benchmark_report.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall_metrics"]
    lines = [
        f"# {report['report_title']}",
        "",
        report["accuracy_claim"],
        "",
        "All system detections are review candidates. Human review is required; enforcement action is none.",
        "",
        "## Overall metrics",
        "",
        "| TP | FP | FN | Precision | Recall | F1 | FP/video hour |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['true_positives']} | {overall['false_positives']} | "
            f"{overall['false_negatives']} | {overall['precision']:.4f} | "
            f"{overall['recall']:.4f} | {overall['f1']:.4f} | "
            f"{overall['false_positives_per_video_hour']:.4f} |"
        ),
        "",
        "Headline annotation confidence: "
        + ", ".join(report["headline_annotation_confidences"]),
        "",
        "## Scenario metrics",
        "",
        "| Tag | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, metrics in sorted(report["scenario_metrics"].items()):
        lines.append(
            f"| {tag} | {metrics['true_positives']} | {metrics['false_positives']} | "
            f"{metrics['false_negatives']} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-video metrics",
            "",
            "| Video | Split | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for video_id, entry in sorted(report["per_video_metrics"].items()):
        metrics = entry["metrics"]
        lines.append(
            f"| {video_id} | {entry['split']} | {metrics['true_positives']} | "
            f"{metrics['false_positives']} | {metrics['false_negatives']} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} |"
        )
    lines.extend(["", "## Confidence-stratified metrics", ""])
    for confidence, metrics in sorted(report["confidence_stratified_metrics"].items()):
        lines.append(
            f"- `{confidence}`: TP={metrics['true_positives']}, "
            f"FP={metrics['false_positives']}, "
            f"FN={metrics['false_negatives']}, F1={metrics['f1']:.4f}"
        )
    lines.extend(["", "## Policy-specific suppression", ""])
    for key, value in sorted(report["policy_specific_metrics"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Suspected failure breakdown", ""])
    if report["failure_breakdown"]:
        for category, count in sorted(report["failure_breakdown"].items()):
            lines.append(f"- `{category}`: {count}")
    else:
        lines.append("No headline FP/FN failures.")
    lines.extend(["", "## Failure review", ""])
    if report["failures"]:
        lines.extend(
            [
                "| Failure | Video | Kind | Suspected category | Time | Confidence | Artifacts |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        ranked_failures = sorted(
            report["failures"],
            key=lambda item: (
                -float((item.get("prediction") or {}).get("confidence", 0.0)),
                item["video_id"],
                item["failure_id"],
            ),
        )
        for failure in ranked_failures[:20]:
            confidence = (failure.get("prediction") or {}).get("confidence")
            confidence_text = f"{confidence:.3f}" if confidence is not None else "N/A"
            artifacts = failure.get("artifact_directory") or "N/A"
            lines.append(
                f"| {failure['failure_id']} | {failure['video_id']} | "
                f"{failure['kind']} | {failure['suspected_failure_category']} | "
                f"{failure['timestamp_seconds']:.3f}s | {confidence_text} | "
                f"{artifacts} |"
            )
    else:
        lines.append("No headline failures to inspect.")
    lines.extend(["", "## Performance", ""])
    performance = report["performance_metrics"]
    if performance.get("available"):
        lines.extend(
            [
                f"- Processing FPS: {performance['processing_fps']:.3f}",
                (
                    f"- Real-time factor: {performance['real_time_factor']:.3f}x "
                    "(video duration / processing time; larger is faster)"
                ),
                f"- Measurement scope: {performance['measurement_scope']}",
            ]
        )
    else:
        lines.append("Performance metadata is unavailable in this prediction cache.")
    lines.extend(["", "## Acceptance criteria", ""])
    acceptance = report["acceptance"]
    if acceptance["configured"]:
        lines.append(f"Overall result: `{'PASS' if acceptance['passed'] else 'FAIL'}`")
        for check in acceptance["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- `{check['metric']}`: {check['actual']:.6f} "
                f"{check['operator']} {check['threshold']:.6f} - **{status}**"
            )
    else:
        lines.append("Not configured; no production-readiness threshold is assumed.")
    lines.extend(["", "## Reproducibility", ""])
    for key, value in sorted(report.get("reproducibility", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    if report.get("baseline_comparison") is not None:
        baseline = report["baseline_comparison"]
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                f"Regression detected: `{baseline['regression_detected']}`",
            ]
        )
        for key, value in sorted(baseline["deltas"].items()):
            lines.append(f"- `{key}` delta: {value:.6f}")
        for key, value in sorted(baseline["policy_metric_deltas"].items()):
            lines.append(f"- Policy `{key}` delta: {value:.6f}")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Suspected failure categories are transparent heuristics, not proven root causes.",
            "No real-world accuracy claim should be made until a sufficiently diverse, independently annotated test set has been evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_with_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any],
    tolerances: BaselineTolerances,
) -> dict[str, Any]:
    current_metrics = report["overall_metrics"]
    baseline_metrics = baseline["overall_metrics"]
    current_performance = report.get("performance_metrics", {})
    baseline_performance = baseline.get("performance_metrics", {})
    current_policy = report.get("policy_specific_metrics", {})
    baseline_policy = baseline.get("policy_specific_metrics", {})
    deltas = {
        "precision": current_metrics["precision"] - baseline_metrics["precision"],
        "recall": current_metrics["recall"] - baseline_metrics["recall"],
        "f1": current_metrics["f1"] - baseline_metrics["f1"],
        "false_positives_per_video_hour": (
            current_metrics["false_positives_per_video_hour"]
            - baseline_metrics["false_positives_per_video_hour"]
        ),
        "processing_fps": (
            float(current_performance.get("processing_fps", 0.0))
            - float(baseline_performance.get("processing_fps", 0.0))
        ),
    }
    regressions = {
        "precision": deltas["precision"] < -tolerances.precision,
        "recall": deltas["recall"] < -tolerances.recall,
        "f1": deltas["f1"] < -tolerances.f1,
        "false_positives_per_video_hour": (
            deltas["false_positives_per_video_hour"]
            > tolerances.false_positives_per_hour
        ),
        "processing_fps": (
            bool(current_performance.get("available"))
            and bool(baseline_performance.get("available"))
            and deltas["processing_fps"] < -tolerances.processing_fps
        ),
    }
    policy_deltas = {
        key: float(current_policy[key]) - float(baseline_policy[key])
        for key in sorted(set(current_policy) & set(baseline_policy))
        if key.endswith("_rate")
    }
    return {
        "deltas": deltas,
        "policy_metric_deltas": policy_deltas,
        "regressions": regressions,
        "regression_detected": any(regressions.values()),
        "tolerances": tolerances.model_dump(mode="json"),
    }


def attach_failure_artifacts(
    report: dict[str, Any],
    output_directory: str | Path,
    manifest_path: str | Path,
    videos: list[ManifestVideo],
    *,
    clip_padding_seconds: float,
) -> None:
    video_by_id = {video.id: video for video in videos}
    root = Path(output_directory) / "failures"
    for failure in report["failures"]:
        video_id = failure["video_id"]
        directory = root / _safe_name(video_id) / failure["failure_id"]
        directory.mkdir(parents=True, exist_ok=True)
        prediction = failure.get("prediction") or {}
        copied_image = _copy_if_file(
            prediction.get("representative_frame_path"),
            directory / "representative.jpg",
        )
        copied_clip = _copy_if_file(
            prediction.get("event_video_clip_path"), directory / "clip.mp4"
        )
        video = video_by_id.get(video_id)
        source = (
            resolve_manifest_path(manifest_path, video.path)
            if video is not None and video.path is not None
            else None
        )
        if (
            source is not None
            and source.is_file()
            and (not copied_image or not copied_clip)
        ):
            _extract_failure_media(
                source,
                failure["timestamp_seconds"],
                clip_padding_seconds,
                directory,
                need_image=not copied_image,
                need_clip=not copied_clip,
            )
        failure["artifact_directory"] = str(
            directory.relative_to(Path(output_directory))
        ).replace("\\", "/")
        with (directory / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")


def _copy_if_file(value: Any, destination: Path) -> bool:
    if not isinstance(value, str):
        return False
    source = Path(value)
    if not source.is_file():
        return False
    shutil.copy2(source, destination)
    return True


def _extract_failure_media(
    source: Path,
    timestamp: float,
    padding: float,
    directory: Path,
    *,
    need_image: bool,
    need_clip: bool,
) -> None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - runtime dependency in normal installs
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if need_image:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = capture.read()
        if ok:
            cv2.imwrite(str(directory / "representative.jpg"), frame)
    if need_clip and width > 0 and height > 0:
        start = max(0.0, timestamp - padding)
        end = timestamp + padding
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        writer = cv2.VideoWriter(
            str(directory / "clip.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        while writer.isOpened():
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if position > end:
                break
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
    capture.release()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "video"

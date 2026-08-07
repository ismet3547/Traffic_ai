from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.benchmark.adapter import (
    load_prediction_document,
    prediction_document_from_run,
)
from app.benchmark.annotations import load_manifest
from app.benchmark.evaluator import evaluate_benchmark
from app.benchmark.models import BaselineTolerances, RuntimePerformance
from app.benchmark.reports import (
    compare_with_baseline,
    render_markdown,
    write_reports,
)
from app.benchmark.runner import build_reproducibility_snapshot, select_videos
from app.tools.run_benchmark import main as run_benchmark_main

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/benchmark/manifests/synthetic_manifest.yaml"
PREDICTIONS = ROOT / "data/benchmark/predictions"


def _synthetic_report(tmp_path: Path):
    manifest = load_manifest(MANIFEST)
    videos = select_videos(manifest, "validation")
    predictions = {
        video.id: load_prediction_document(PREDICTIONS / f"{video.id}.json")
        for video in videos
    }
    reproducibility = build_reproducibility_snapshot(
        MANIFEST,
        manifest,
        videos,
        tmp_path,
        git_commit="abc123",
    )
    report = evaluate_benchmark(
        MANIFEST,
        manifest,
        predictions,
        videos=videos,
        reproducibility=reproducibility,
    )
    return manifest, report


def test_synthetic_integration_metrics_and_per_tag_breakdown(tmp_path) -> None:
    _, report = _synthetic_report(tmp_path)
    overall = report["overall_metrics"]
    assert overall["true_positives"] == 2
    assert overall["false_positives"] == 1
    assert overall["false_negatives"] == 0
    assert overall["precision"] == pytest.approx(2 / 3)
    assert overall["recall"] == 1.0
    assert overall["f1"] == 0.8
    assert report["scenario_metrics"]["daylight"] == overall
    assert report["policy_specific_metrics"]["overtake_false_positive_rate"] == 1.0


def test_report_serialization_is_deterministic_and_labeled_synthetic(tmp_path) -> None:
    _, report = _synthetic_report(tmp_path)
    first_json, first_markdown = write_reports(report, tmp_path / "first")
    second_json, second_markdown = write_reports(report, tmp_path / "second")
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_bytes() == second_markdown.read_bytes()
    assert "SYNTHETIC EXAMPLE - NOT REAL-WORLD PERFORMANCE" in render_markdown(report)


def test_reproducibility_snapshot_contains_config_hash_and_versions(tmp_path) -> None:
    _, report = _synthetic_report(tmp_path)
    reproducibility = report["reproducibility"]
    assert reproducibility["git_commit"] == "abc123"
    assert len(reproducibility["config_hash_sha256"]) == 64
    assert reproducibility["annotation_schema_versions"] == ["1.0"]
    assert (tmp_path / "resolved_config.yaml").is_file()


def test_baseline_comparison_uses_directional_tolerances(tmp_path) -> None:
    _, report = _synthetic_report(tmp_path)
    baseline = json.loads(json.dumps(report))
    baseline["overall_metrics"]["precision"] = 0.8
    comparison = compare_with_baseline(
        report,
        baseline,
        BaselineTolerances(precision=0.01),
    )
    assert comparison["deltas"]["precision"] == pytest.approx(-0.8 + 2 / 3)
    assert comparison["regressions"]["precision"] is True
    assert comparison["regression_detected"] is True


def test_runtime_event_adapter_reads_only_pending_review_index(tmp_path) -> None:
    run = tmp_path / "run"
    event_directory = run / "events" / "event_1"
    event_directory.mkdir(parents=True)
    record = {
        "event_id": "event_1",
        "event_type": "left_lane_review_candidate",
        "review_status": "pending_human_review",
        "track_id": 17,
        "event_start_timestamp_seconds": 12.0,
        "event_end_timestamp_seconds": 20.0,
        "duration_seconds": 8.0,
        "confidence_score": 0.7,
        "review_reason_codes": ["LEFT_LANE_DURATION_EXCEEDED"],
        "geometry_integrity": {"status": "trusted", "reason_codes": []},
        "overtaking_assessment": {"status": "not_overtaking"},
        "traffic_context": {"congestion_level": "free_flow"},
        "candidate_lifecycle": {"state": "finalized"},
    }
    (run / "events.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (run / "cancelled_events.jsonl").write_text("{}\n", encoding="utf-8")
    performance = RuntimePerformance(
        total_processing_time_seconds=5,
        video_duration_seconds=10,
        frames_processed=100,
        processing_fps=20,
        real_time_factor=2,
        average_frame_processing_time_ms=50,
    )
    document = prediction_document_from_run(
        run,
        video_id="video_a",
        source_file="video.mp4",
        performance=performance,
    )
    assert len(document.predictions) == 1
    assert document.predictions[0].geometry_status == "trusted"
    assert document.predictions[0].overtaking_status == "not_overtaking"
    assert document.cancelled_event_count == 1


def test_skip_inference_cli_evaluates_prediction_cache(tmp_path, capsys) -> None:
    exit_code = run_benchmark_main(
        [
            "--manifest",
            str(MANIFEST),
            "--output",
            str(tmp_path),
            "--predictions-dir",
            str(PREDICTIONS),
            "--skip-inference",
            "--no-failure-artifacts",
            "--split",
            "validation",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TP=2 FP=1 FN=0" in output
    report = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert report["overall_metrics"]["f1"] == 0.8

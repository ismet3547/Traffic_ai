from __future__ import annotations

import copy

import pytest

from app.benchmark.duration import validate_duration_evidence
from app.benchmark.fingerprints import canonical_sha256, dataset_fingerprint_payload
from app.benchmark.models import (
    BaselineTolerances,
    DurationEvidence,
    DurationValidationConfig,
    ManifestVideo,
)
from app.benchmark.reports import compare_with_baseline


def _duration(source: str, seconds: float, confidence: str):
    return DurationEvidence(source=source, seconds=seconds, confidence=confidence)


def _report(dataset: str = "dataset-a", evaluation: str = "evaluation-a"):
    return {
        "overall_metrics": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.7466666667,
            "false_positives_per_video_hour": 1.2,
        },
        "performance_metrics": {"available": True, "processing_fps": 20.0},
        "policy_specific_metrics": {"overtake_false_positive_rate": 0.1},
        "reproducibility": {
            "dataset_fingerprint": dataset,
            "evaluation_fingerprint": evaluation,
            "production_config_hash_sha256": "production-a",
        },
    }


def test_gross_manifest_vs_video_metadata_duration_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="DURATION_EVIDENCE_MISMATCH"):
        validate_duration_evidence(
            "video_a",
            [
                _duration("manifest", 3600.0, "low"),
                _duration("video_metadata", 600.0, "high"),
            ],
            DurationValidationConfig(
                absolute_tolerance_seconds=1.0, relative_tolerance=0.01
            ),
        )


def test_duration_sources_within_tolerance_are_accepted() -> None:
    result = validate_duration_evidence(
        "video_a",
        [
            _duration("manifest", 600.0, "low"),
            _duration("video_metadata", 600.5, "high"),
            _duration("annotation", 600.2, "medium"),
        ],
        DurationValidationConfig(
            absolute_tolerance_seconds=1.0, relative_tolerance=0.001
        ),
    )
    assert result.duration_seconds_used == 600.5
    assert result.duration_source == "video_metadata"
    assert result.duration_validation_status == "verified_video_metadata"
    assert result.denominator_confidence == "high"


def test_cache_only_single_duration_source_is_explicitly_unverified() -> None:
    result = validate_duration_evidence(
        "video_a",
        [_duration("annotation", 600.0, "medium")],
        DurationValidationConfig(),
    )
    assert result.duration_validation_status == "single_source_unverified"
    assert result.denominator_confidence == "low"


def test_same_dataset_and_evaluation_protocol_are_comparable() -> None:
    current = _report()
    baseline = copy.deepcopy(current)
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is True
    assert comparison["reason_codes"] == []
    assert comparison["deltas"]["precision"] == 0.0


def test_changed_annotation_hash_is_not_comparable(tmp_path) -> None:
    video = ManifestVideo(id="a", annotation="a.json")
    first_payload = dataset_fingerprint_payload(
        tmp_path / "manifest.yaml",
        [video],
        {"a:0:a.json": "annotation-hash-a"},
        ["1.0"],
    )
    second_payload = dataset_fingerprint_payload(
        tmp_path / "manifest.yaml",
        [video],
        {"a:0:a.json": "annotation-hash-b"},
        ["1.0"],
    )
    current = _report(dataset=canonical_sha256(first_payload))
    baseline = _report(dataset=canonical_sha256(second_payload))
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is False
    assert comparison["reason_codes"] == ["DATASET_FINGERPRINT_MISMATCH"]
    assert comparison["deltas"] == {}


def test_changed_video_set_is_not_comparable(tmp_path) -> None:
    video_a = ManifestVideo(id="a", annotation="a.json")
    video_b = ManifestVideo(id="b", annotation="b.json")
    annotation_hashes = {
        "a:0:a.json": "annotation-hash-a",
        "b:0:b.json": "annotation-hash-b",
    }
    first_payload = dataset_fingerprint_payload(
        tmp_path / "manifest.yaml", [video_a], annotation_hashes, ["1.0"]
    )
    second_payload = dataset_fingerprint_payload(
        tmp_path / "manifest.yaml", [video_a, video_b], annotation_hashes, ["1.0"]
    )
    comparison = compare_with_baseline(
        _report(dataset=canonical_sha256(first_payload)),
        _report(dataset=canonical_sha256(second_payload)),
        BaselineTolerances(),
    )
    assert comparison["comparison_valid"] is False
    assert comparison["reason_codes"] == ["DATASET_FINGERPRINT_MISMATCH"]
    assert comparison["deltas"] == {}


def test_changed_matching_threshold_fingerprint_is_not_comparable() -> None:
    current = _report(evaluation=canonical_sha256({"minimum_temporal_iou": 0.3}))
    baseline = _report(evaluation=canonical_sha256({"minimum_temporal_iou": 0.5}))
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is False
    assert comparison["reason_codes"] == ["EVALUATION_CONFIG_MISMATCH"]


def test_changed_production_config_only_remains_comparable() -> None:
    current = _report()
    baseline = copy.deepcopy(current)
    baseline["reproducibility"]["production_config_hash_sha256"] = "production-b"
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is True


def test_incomparable_override_shows_deltas_with_prominent_warning() -> None:
    comparison = compare_with_baseline(
        _report(dataset="dataset-a"),
        _report(dataset="dataset-b"),
        BaselineTolerances(),
        allow_incomparable=True,
    )
    assert comparison["comparison_valid"] is False
    assert comparison["override_used"] is True
    assert comparison["warning"] == "NON-COMPARABLE BASELINE OVERRIDE"
    assert comparison["deltas"]["precision"] == 0.0

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
    PredictionDocument,
    VideoIdentity,
    VideoIdentityMode,
)
from app.benchmark.protocol import current_evaluation_protocol
from app.benchmark.reports import compare_with_baseline


def _duration(source: str, seconds: float, confidence: str):
    return DurationEvidence(source=source, seconds=seconds, confidence=confidence)


def _report(dataset: str = "dataset-a", evaluation: str = "evaluation-a"):
    protocol = current_evaluation_protocol().model_dump(mode="json")
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
            "dataset_identity_status": "verified",
            "source_video_identities": {
                "a": {
                    "video_id": "a",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "identity_mode": "full_sha256",
                    "verified": True,
                    "reason_codes": [],
                    "source_path": None,
                }
            },
            "evaluation_fingerprint": evaluation,
            "evaluation_protocol": protocol,
            "production_config_hash_sha256": "production-a",
        },
    }


def _dataset_payload(videos: list[ManifestVideo], annotation_hashes: dict[str, str]):
    identities = {
        video.id: VideoIdentity(
            video_id=video.id,
            sha256=(video.id[0] * 64),
            size_bytes=1,
            identity_mode=VideoIdentityMode.FULL_SHA256,
            verified=True,
        )
        for video in videos
    }
    predictions = {
        video.id: PredictionDocument(
            video_id=video.id,
            source_video_sha256=identities[video.id].sha256,
            source_video_size_bytes=1,
        )
        for video in videos
    }
    annotations = {
        f"{video.id}:0": {
            "sha256": annotation_hashes[video.id],
            "schema_version": "1.0",
        }
        for video in videos
    }
    return dataset_fingerprint_payload(videos, annotations, identities, predictions)


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


def test_changed_annotation_hash_is_not_comparable() -> None:
    video = ManifestVideo(id="a", annotation="a.json")
    first_payload = _dataset_payload([video], {"a": "annotation-hash-a"})
    second_payload = _dataset_payload([video], {"a": "annotation-hash-b"})
    current = _report(dataset=canonical_sha256(first_payload))
    baseline = _report(dataset=canonical_sha256(second_payload))
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is False
    assert comparison["reason_codes"] == ["DATASET_FINGERPRINT_MISMATCH"]
    assert comparison["deltas"] == {}


def test_changed_video_set_is_not_comparable() -> None:
    video_a = ManifestVideo(id="a", annotation="a.json")
    video_b = ManifestVideo(id="b", annotation="b.json")
    annotation_hashes = {
        "a": "annotation-hash-a",
        "b": "annotation-hash-b",
    }
    first_payload = _dataset_payload([video_a], annotation_hashes)
    second_payload = _dataset_payload([video_a, video_b], annotation_hashes)
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

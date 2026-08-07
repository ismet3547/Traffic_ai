from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.benchmark.fingerprints import (
    canonical_sha256,
    dataset_fingerprint_payload,
    dataset_identity_status,
    evaluation_fingerprint_payload,
    resolve_video_identities,
    streaming_file_sha256,
)
from app.benchmark.models import (
    BaselineTolerances,
    BenchmarkSettings,
    EvaluationProtocolIdentity,
    ManifestVideo,
    MatchingConfig,
    PredictionDocument,
    VideoIdentity,
    VideoIdentityMode,
)
from app.benchmark.protocol import current_evaluation_protocol
from app.benchmark.reports import compare_with_baseline
from app.benchmark.runner import run_video_inference


def _video(video_id: str, path: str | None = None) -> ManifestVideo:
    return ManifestVideo(
        id=video_id,
        path=path,
        annotation=f"{video_id}.json",
        duration_seconds=600.0,
    )


def _prediction(
    video_id: str,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> PredictionDocument:
    return PredictionDocument(
        video_id=video_id,
        source_video_sha256=sha256,
        source_video_size_bytes=size_bytes,
    )


def _verified_identity(
    video_id: str,
    sha256: str,
    size_bytes: int,
    *,
    mode: VideoIdentityMode = VideoIdentityMode.FULL_SHA256,
) -> VideoIdentity:
    return VideoIdentity(
        video_id=video_id,
        sha256=sha256,
        size_bytes=size_bytes,
        identity_mode=mode,
        verified=True,
    )


def _dataset_payload(
    videos: list[ManifestVideo],
    identities: dict[str, VideoIdentity],
    predictions: dict[str, PredictionDocument],
    *,
    annotation_hashes: dict[str, str] | None = None,
):
    annotation_hashes = annotation_hashes or {}
    annotations = {
        f"{video.id}:0": {
            "sha256": annotation_hashes.get(video.id, f"annotation-{video.id}"),
            "schema_version": "1.0",
        }
        for video in videos
    }
    return dataset_fingerprint_payload(videos, annotations, identities, predictions)


def _report(
    dataset_payload: dict,
    identities: dict[str, VideoIdentity],
    *,
    protocol: EvaluationProtocolIdentity | None = None,
    settings: BenchmarkSettings | None = None,
    production_config_hash: str = "production-a",
):
    active_protocol = protocol or current_evaluation_protocol()
    evaluation_payload = evaluation_fingerprint_payload(
        settings or BenchmarkSettings(), active_protocol
    )
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
            "dataset_fingerprint": canonical_sha256(dataset_payload),
            "dataset_identity_status": dataset_identity_status(identities).value,
            "source_video_identities": {
                key: value.model_dump(mode="json")
                for key, value in sorted(identities.items())
            },
            "evaluation_fingerprint": canonical_sha256(evaluation_payload),
            "evaluation_protocol": active_protocol.model_dump(mode="json"),
            "production_config_hash_sha256": production_config_hash,
        },
    }


def test_same_video_bytes_produce_same_sha256(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same synthetic video bytes")
    second.write_bytes(b"same synthetic video bytes")
    assert streaming_file_sha256(first) == streaming_file_sha256(second)


def test_changed_one_byte_changes_video_sha256(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"video-A")
    second.write_bytes(b"video-B")
    assert streaming_file_sha256(first) != streaming_file_sha256(second)


def test_same_video_content_at_different_paths_has_same_dataset_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "old-name.mp4"
    second = tmp_path / "renamed.mp4"
    content = b"identical footage"
    first.write_bytes(content)
    second.write_bytes(content)
    sha256 = streaming_file_sha256(first)
    prediction = _prediction("same", sha256, len(content))
    first_video = _video("same", first.name)
    second_video = _video("same", second.name)
    first_identity = resolve_video_identities(
        tmp_path / "manifest.yaml", [first_video], {"same": prediction}
    )
    second_identity = resolve_video_identities(
        tmp_path / "manifest.yaml", [second_video], {"same": prediction}
    )
    first_payload = _dataset_payload(
        [first_video], first_identity, {"same": prediction}
    )
    second_payload = _dataset_payload(
        [second_video], second_identity, {"same": prediction}
    )
    assert canonical_sha256(first_payload) == canonical_sha256(second_payload)


def test_same_id_duration_and_annotation_but_changed_video_bytes_changes_dataset() -> (
    None
):
    video = _video("same")
    first_sha = canonical_sha256("video bytes A")
    second_sha = canonical_sha256("video bytes B")
    first_prediction = _prediction("same", first_sha, 13)
    second_prediction = _prediction("same", second_sha, 13)
    first_identities = {"same": _verified_identity("same", first_sha, 13)}
    second_identities = {"same": _verified_identity("same", second_sha, 13)}
    first_payload = _dataset_payload(
        [video], first_identities, {"same": first_prediction}
    )
    second_payload = _dataset_payload(
        [video], second_identities, {"same": second_prediction}
    )
    comparison = compare_with_baseline(
        _report(second_payload, second_identities),
        _report(first_payload, first_identities),
        BaselineTolerances(),
    )
    assert canonical_sha256(first_payload) != canonical_sha256(second_payload)
    assert comparison["comparison_valid"] is False
    assert "DATASET_FINGERPRINT_MISMATCH" in comparison["reason_codes"]


def test_changed_annotation_bytes_change_dataset_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "annotation-a.json"
    second = tmp_path / "annotation-b.json"
    first.write_bytes(b'{"events": []}')
    second.write_bytes(b'{"events": [1]}')
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    first_payload = _dataset_payload(
        [video],
        identities,
        {"same": prediction},
        annotation_hashes={"same": streaming_file_sha256(first)},
    )
    second_payload = _dataset_payload(
        [video],
        identities,
        {"same": prediction},
        annotation_hashes={"same": streaming_file_sha256(second)},
    )
    assert canonical_sha256(first_payload) != canonical_sha256(second_payload)


def test_legacy_cache_without_video_hash_is_unverified(tmp_path: Path) -> None:
    video = _video("legacy")
    prediction = PredictionDocument(schema_version="1.0", video_id="legacy")
    identities = resolve_video_identities(
        tmp_path / "manifest.yaml", [video], {"legacy": prediction}
    )
    assert identities["legacy"].identity_mode == VideoIdentityMode.UNVERIFIED
    assert identities["legacy"].verified is False
    assert dataset_identity_status(identities).value == "unverified"


def test_cache_preserved_sha256_is_verified_when_raw_video_is_unavailable(
    tmp_path: Path,
) -> None:
    sha256 = canonical_sha256("source video")
    prediction = _prediction("cached", sha256, 12)
    identities = resolve_video_identities(
        tmp_path / "manifest.yaml", [_video("cached")], {"cached": prediction}
    )
    assert identities["cached"].identity_mode == VideoIdentityMode.CACHED_FULL_SHA256
    assert identities["cached"].verified is True
    assert identities["cached"].sha256 == sha256


def test_raw_and_cache_preserved_full_hash_have_same_content_fingerprint() -> None:
    video = _video("same")
    sha256 = canonical_sha256("source video")
    prediction = _prediction("same", sha256, 12)
    raw_identity = {"same": _verified_identity("same", sha256, 12)}
    cached_identity = {
        "same": _verified_identity(
            "same", sha256, 12, mode=VideoIdentityMode.CACHED_FULL_SHA256
        )
    }
    raw_payload = _dataset_payload([video], raw_identity, {"same": prediction})
    cached_payload = _dataset_payload([video], cached_identity, {"same": prediction})
    assert canonical_sha256(raw_payload) == canonical_sha256(cached_payload)


def test_new_inference_cache_automatically_contains_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source bytes used for inference")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    def fake_run(command, **_kwargs):
        output_base = Path(command[command.index("--output-dir") + 1])
        run_directory = output_base / "run-001"
        run_directory.mkdir(parents=True)
        (run_directory / "events.jsonl").write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.benchmark.runner.subprocess.run", fake_run)
    monkeypatch.setattr("app.benchmark.runner.probe_video", lambda _path: (10, 1.0))
    document = run_video_inference(
        tmp_path / "manifest.yaml",
        ManifestVideo(
            id="video",
            path=str(source),
            annotation="annotation.json",
            config=str(config),
        ),
        tmp_path / "output",
        git_commit="abc123",
    )
    assert document.schema_version == "1.1"
    assert document.source_video_sha256 == streaming_file_sha256(source)
    assert document.source_video_size_bytes == source.stat().st_size


def test_raw_video_hash_disagreement_with_cache_fails_before_evaluation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "same.mp4"
    source.write_bytes(b"current video B")
    cached_bytes = b"original video A"
    prediction = _prediction(
        "same", hashlib.sha256(cached_bytes).hexdigest(), len(cached_bytes)
    )
    with pytest.raises(ValueError, match="PREDICTION_CACHE_SOURCE_MISMATCH"):
        resolve_video_identities(
            tmp_path / "manifest.yaml",
            [_video("same", source.name)],
            {"same": prediction},
        )


def test_same_verified_dataset_and_protocol_are_comparable() -> None:
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    payload = _dataset_payload([video], identities, {"same": prediction})
    report = _report(payload, identities)
    comparison = compare_with_baseline(
        report, copy.deepcopy(report), BaselineTolerances()
    )
    assert comparison["comparison_valid"] is True
    assert comparison["comparison_mode"] == "strict_comparable"


def test_changed_production_config_only_remains_comparable() -> None:
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    payload = _dataset_payload([video], identities, {"same": prediction})
    current = _report(payload, identities, production_config_hash="production-b")
    baseline = _report(payload, identities, production_config_hash="production-a")
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is True


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        ("matcher_semantics_version", "maximum_iou_greedy_v1"),
        ("metric_semantics_version", "legacy_implicit_accounting_v1"),
        ("annotation_ontology_version", "legacy_roles_v0"),
    ],
)
def test_changed_protocol_component_is_not_comparable(
    field: str, legacy_value: str
) -> None:
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    payload = _dataset_payload([video], identities, {"same": prediction})
    current_protocol = current_evaluation_protocol()
    legacy_protocol = current_protocol.model_copy(update={field: legacy_value})
    comparison = compare_with_baseline(
        _report(payload, identities, protocol=current_protocol),
        _report(payload, identities, protocol=legacy_protocol),
        BaselineTolerances(),
    )
    assert comparison["comparison_valid"] is False
    assert "EVALUATION_PROTOCOL_MISMATCH" in comparison["reason_codes"]


def test_same_thresholds_but_greedy_v1_and_max_cardinality_v2_are_incomparable() -> (
    None
):
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    payload = _dataset_payload([video], identities, {"same": prediction})
    settings = BenchmarkSettings(matching=MatchingConfig(minimum_temporal_iou=0.3))
    current_protocol = current_evaluation_protocol()
    greedy_protocol = current_protocol.model_copy(
        update={"matcher_semantics_version": "maximum_iou_greedy_v1"}
    )
    current = _report(payload, identities, protocol=current_protocol, settings=settings)
    baseline = _report(payload, identities, protocol=greedy_protocol, settings=settings)
    assert (
        current["reproducibility"]["evaluation_fingerprint"]
        != baseline["reproducibility"]["evaluation_fingerprint"]
    )
    comparison = compare_with_baseline(current, baseline, BaselineTolerances())
    assert comparison["comparison_valid"] is False
    assert comparison["reason_codes"] == ["EVALUATION_PROTOCOL_MISMATCH"]


def test_legacy_baseline_without_protocol_identity_is_non_comparable() -> None:
    video = _video("same")
    video_sha = canonical_sha256("video")
    prediction = _prediction("same", video_sha, 5)
    identities = {"same": _verified_identity("same", video_sha, 5)}
    payload = _dataset_payload([video], identities, {"same": prediction})
    current = _report(payload, identities)
    legacy = copy.deepcopy(current)
    legacy["reproducibility"].pop("evaluation_protocol")
    comparison = compare_with_baseline(current, legacy, BaselineTolerances())
    assert comparison["comparison_valid"] is False
    assert "LEGACY_BASELINE_IDENTITY_INCOMPLETE" in comparison["reason_codes"]


def test_explicit_override_displays_deltas_but_remains_non_comparable() -> None:
    video = _video("same")
    first_sha = canonical_sha256("video-a")
    second_sha = canonical_sha256("video-b")
    first_prediction = _prediction("same", first_sha, 7)
    second_prediction = _prediction("same", second_sha, 7)
    first_identities = {"same": _verified_identity("same", first_sha, 7)}
    second_identities = {"same": _verified_identity("same", second_sha, 7)}
    comparison = compare_with_baseline(
        _report(
            _dataset_payload([video], first_identities, {"same": first_prediction}),
            first_identities,
        ),
        _report(
            _dataset_payload([video], second_identities, {"same": second_prediction}),
            second_identities,
        ),
        BaselineTolerances(),
        allow_incomparable=True,
    )
    assert comparison["comparison_valid"] is False
    assert comparison["comparison_mode"] == "forced_non_comparable"
    assert comparison["warning"] == "NON-COMPARABLE BASELINE OVERRIDE"
    assert comparison["deltas"]["precision"] == 0.0


def test_canonical_fingerprint_ordering_is_deterministic() -> None:
    first = {"videos": [{"id": "a", "values": {"x": 1, "y": 2}}]}
    second = {"videos": [{"values": {"y": 2, "x": 1}, "id": "a"}]}
    assert canonical_sha256(first) == canonical_sha256(second)


def test_manifest_video_order_does_not_change_dataset_fingerprint() -> None:
    videos = [_video("a"), _video("b")]
    first_sha = canonical_sha256("a")
    second_sha = canonical_sha256("b")
    predictions = {
        "a": _prediction("a", first_sha, 1),
        "b": _prediction("b", second_sha, 1),
    }
    identities = {
        "a": _verified_identity("a", first_sha, 1),
        "b": _verified_identity("b", second_sha, 1),
    }
    first = _dataset_payload(videos, identities, predictions)
    second = _dataset_payload(list(reversed(videos)), identities, predictions)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_one_unverified_video_makes_strict_comparison_invalid() -> None:
    videos = [_video("verified"), _video("legacy")]
    sha256 = canonical_sha256("video")
    predictions = {
        "verified": _prediction("verified", sha256, 5),
        "legacy": _prediction("legacy"),
    }
    identities = {
        "verified": _verified_identity("verified", sha256, 5),
        "legacy": VideoIdentity(
            video_id="legacy",
            identity_mode=VideoIdentityMode.UNVERIFIED,
            verified=False,
            reason_codes=["PREDICTION_CACHE_SOURCE_IDENTITY_MISSING"],
        ),
    }
    report = _report(_dataset_payload(videos, identities, predictions), identities)
    comparison = compare_with_baseline(
        report, copy.deepcopy(report), BaselineTolerances()
    )
    assert comparison["comparison_valid"] is False
    assert "DATASET_IDENTITY_UNVERIFIED" in comparison["reason_codes"]


def test_all_video_identities_verified_produce_verified_dataset_status() -> None:
    identities = {
        "raw": _verified_identity("raw", canonical_sha256("raw"), 3),
        "cached": _verified_identity(
            "cached",
            canonical_sha256("cached"),
            6,
            mode=VideoIdentityMode.CACHED_FULL_SHA256,
        ),
    }
    assert dataset_identity_status(identities).value == "verified"

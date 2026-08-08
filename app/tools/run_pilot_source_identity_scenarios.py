"""Run deterministic Phase 4.3.3 local-source identity scenarios."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import cast

from app.dataset.models import (
    PermissionStatus,
    SourceType,
    VideoIntakeRecord,
    VideoResolution,
)
from app.dataset.pilot import (
    TERMINAL_PILOT_STATES,
    PilotStageCounts,
    assess_local_source_identity,
    derive_pilot_state,
)
from app.dataset.pilot_review import (
    AgreementReviewStatus,
    FailureReviewCoverage,
    ScaleUpDecision,
    ScaleUpDecisionStatus,
)


def main() -> int:
    registered_bytes = b"registered source AAA"
    different_size_bytes = b"replacement BBB with a different size"
    same_size_replacement = bytearray(registered_bytes)
    same_size_replacement[len(same_size_replacement) // 2] ^= 0xFF
    same_size_bytes = bytes(same_size_replacement)
    record = _record(registered_bytes)

    with tempfile.TemporaryDirectory(prefix="traffic_ai_source_identity_") as value:
        source_path = Path(value) / "clip_001.mp4"
        source_path.write_bytes(registered_bytes)
        scenario_a = _scenario(source_path, record)

        source_path.write_bytes(different_size_bytes)
        scenario_b = _scenario(source_path, record)

        source_path.write_bytes(same_size_bytes)
        scenario_c = _scenario(source_path, record)

        source_path.write_bytes(registered_bytes)
        scenario_d = _scenario(source_path, record)

        source_path.unlink()
        scenario_e = _scenario(source_path, record)

    output = {
        "synthetic": True,
        "warning": "Synthetic integrity fixture; this is not real pilot evidence.",
        "registered_identity": {
            "size_bytes": len(registered_bytes),
            "sha256": hashlib.sha256(registered_bytes).hexdigest(),
        },
        "replacement_hashes": {
            "different_size_bbb_sha256": hashlib.sha256(
                different_size_bytes
            ).hexdigest(),
            "same_size_ccc_sha256": hashlib.sha256(same_size_bytes).hexdigest(),
        },
        "scenarios": {
            "A_registered_AAA_local_AAA": scenario_a,
            "B_same_path_BBB_different_size": scenario_b,
            "C_same_path_CCC_same_size": scenario_c,
            "D_original_AAA_restored": scenario_d,
            "E_local_file_deleted": scenario_e,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if _passed(output) else 1


def _record(source_bytes: bytes) -> VideoIntakeRecord:
    return VideoIntakeRecord(
        video_id="clip_001",
        source_group_id="synthetic_source_identity",
        source_type=SourceType.OWN_CAPTURE,
        source_reference="synthetic Phase 4.3.3 lifecycle",
        acquisition_date=date(2026, 8, 8),
        license_or_permission_status=PermissionStatus.VERIFIED,
        redistribution_allowed=False,
        benchmark_use_allowed=True,
        source_video_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_video_size_bytes=len(source_bytes),
        source_identity_verified=True,
        duration_seconds=1.0,
        resolution=VideoResolution(width=16, height=16),
        fps=1.0,
        original_filename="clip_001.mp4",
    )


def _scenario(source_path: Path, record: VideoIntakeRecord) -> dict[str, object]:
    assessment = assess_local_source_identity(source_path, record)
    state = derive_pilot_state(
        _complete_counts(),
        _complete_agreement_review(),
        True,
        _complete_failure_review(),
        _valid_decision(),
        has_completion_blocker=not assessment.identity_verified,
    )
    return {
        "source_present": assessment.present,
        "source_valid": assessment.identity_verified,
        "actual_size_bytes": assessment.actual_size_bytes,
        "actual_sha256": assessment.actual_sha256,
        "state": state.value,
        "pilot_executed": state in TERMINAL_PILOT_STATES,
        "blocker_codes": (
            [] if assessment.reason_code is None else [assessment.reason_code]
        ),
    }


def _complete_counts() -> PilotStageCounts:
    return PilotStageCounts(
        selected_clips=1,
        registered_clips=1,
        real_world_confirmed_clips=1,
        total_duration_seconds=1.0,
        double_annotated_clips=1,
        agreement_ready_clips=1,
        adjudicated_clips=1,
        benchmark_exported_clips=1,
        inference_complete_clips=1,
        benchmark_complete_clips=1,
    )


def _complete_agreement_review() -> AgreementReviewStatus:
    return AgreementReviewStatus(
        required=True,
        complete=True,
        stale=False,
        required_report_count=3,
        reviewed_report_count=3,
        missing_count=0,
        unknown_count=0,
        artifact_present=True,
        artifact_content_sha256="a" * 64,
    )


def _complete_failure_review() -> FailureReviewCoverage:
    return FailureReviewCoverage(
        required_count=1,
        reviewed_count=1,
        missing_count=0,
        duplicate_count=0,
        unknown_count=0,
        stale_count=0,
        coverage_ratio=1.0,
        complete=True,
        artifact_present=True,
        artifact_content_sha256="b" * 64,
        message="Synthetic failure review complete.",
    )


def _valid_decision() -> ScaleUpDecisionStatus:
    return ScaleUpDecisionStatus(
        present=True,
        valid=True,
        stale=False,
        decision=ScaleUpDecision.GO,
        artifact_content_sha256="c" * 64,
    )


def _passed(output: dict[str, object]) -> bool:
    scenarios = cast(dict[str, dict[str, object]], output["scenarios"])
    expected = {
        "A_registered_AAA_local_AAA": (True, "COMPLETE_GO", []),
        "B_same_path_BBB_different_size": (
            False,
            "BASELINE_FROZEN",
            ["LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"],
        ),
        "C_same_path_CCC_same_size": (
            False,
            "BASELINE_FROZEN",
            ["LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH"],
        ),
        "D_original_AAA_restored": (True, "COMPLETE_GO", []),
        "E_local_file_deleted": (
            False,
            "BASELINE_FROZEN",
            ["LOCAL_VIDEO_MISSING"],
        ),
    }
    return all(
        scenario["source_valid"] is source_valid
        and scenario["state"] == state
        and scenario["blocker_codes"] == blockers
        and scenario["pilot_executed"] is source_valid
        for name, (source_valid, state, blockers) in expected.items()
        for scenario in [scenarios[name]]
    )


if __name__ == "__main__":
    raise SystemExit(main())

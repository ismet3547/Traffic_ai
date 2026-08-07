from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import OutputConfig
from app.events import writer as writer_module
from app.events.writer import EventArtifactWriter
from app.models import (
    CandidateDecisionRecord,
    CandidateTransition,
    CongestionLevel,
    GlobalTrafficContext,
    NeighborVehicles,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    VehicleTrafficContext,
    VideoInfo,
)


class _FakeVideoWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames = 0

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames += 1

    def release(self) -> None:
        self.path.write_bytes(f"frames={self.frames}".encode())


class _FakeCV2:
    IMWRITE_JPEG_QUALITY = 1

    @staticmethod
    def imwrite(path: str, frame: np.ndarray, params: list[int]) -> bool:
        Path(path).write_bytes(b"image")
        return True

    @staticmethod
    def VideoWriter_fourcc(*codec: str) -> int:
        return 0

    @staticmethod
    def VideoWriter(
        path: str, fourcc: int, fps: float, dimensions: tuple[int, int]
    ) -> _FakeVideoWriter:
        return _FakeVideoWriter(path)


def test_writes_finalized_human_review_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(writer_module, "_cv2", lambda: _FakeCV2)
    artifact_writer = EventArtifactWriter(
        run_directory=tmp_path,
        output_config=OutputConfig(
            clip_pre_event_seconds=0.5,
            clip_max_duration_seconds=2.0,
        ),
        video_info=VideoInfo(width=32, height=24, fps=2.0, frame_count=20),
        source_video_name="source.mp4",
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    traffic = GlobalTrafficContext(
        congestion_level=CongestionLevel.FREE_FLOW,
        traffic_density=0.2,
        active_vehicle_count=2,
        lane_vehicle_counts={"left": 1, "right": 1},
        average_normalized_motion_per_second=0.05,
        confidence=0.8,
    )
    vehicle_context = VehicleTrafficContext(
        track_id=4,
        neighbors=NeighborVehicles(),
        nearby_vehicle_count=1,
        adjacent_right_lane_id="right",
        right_lane_available=True,
        right_lane_available_seconds=2.2,
        right_lane_confidence=0.8,
    )
    overtaking = OvertakingAssessment(
        track_id=4,
        status=OvertakingStatus.NOT_OVERTAKING,
        state=OvertakeState.NONE,
        confidence=0.78,
        evidence=("no_active_overtaking_sequence_detected",),
    )
    artifact_writer.process_frame(frame, [])
    artifact_writer.process_frame(
        frame,
        [
            CandidateTransition(
                transition="started",
                track_id=4,
                lane_id="left",
                start_timestamp_seconds=1.0,
                timestamp_seconds=3.0,
                duration_seconds=2.0,
                confidence_score=0.8,
                review_reason_codes=(
                    "LEFT_LANE_DURATION_EXCEEDED",
                    "RIGHT_LANE_AVAILABLE",
                ),
                policy_version="2.0",
                traffic_context=traffic,
                vehicle_traffic_context=vehicle_context,
                overtaking_assessment=overtaking,
                behavior_classification="possible_left_lane_occupation",
                evidence_confidence_score=0.79,
            )
        ],
    )
    artifact_writer.process_frame(
        frame,
        [
            CandidateTransition(
                transition="ended",
                track_id=4,
                lane_id="left",
                start_timestamp_seconds=1.0,
                timestamp_seconds=4.0,
                duration_seconds=3.0,
                confidence_score=0.85,
                end_reason="left_lane_exit",
            )
        ],
    )

    event_directory = tmp_path / "events" / "left_lane_track_4_0000001000"
    metadata = json.loads((event_directory / "metadata.json").read_text())
    assert (event_directory / "representative.jpg").is_file()
    assert (event_directory / "event.mp4").is_file()
    assert metadata["event_type"] == "left_lane_review_candidate"
    assert metadata["schema_version"] == "3.0"
    assert metadata["review_status"] == "pending_human_review"
    assert metadata["candidate_lifecycle"]["state"] == "finalized"
    assert metadata["human_review_required"] is True
    assert metadata["enforcement_action"] == "none"
    assert metadata["duration_seconds"] == 3.0
    assert metadata["end_reason"] == "left_lane_exit"
    assert metadata["policy_version"] == "2.0"
    assert metadata["behavior_classification"] == "possible_left_lane_occupation"
    assert metadata["evidence_confidence_score"] == 0.79
    assert metadata["traffic_context"]["congestion_level"] == "free_flow"
    assert metadata["traffic_context"]["right_lane_available_seconds"] == 2.2
    assert metadata["overtaking_assessment"]["status"] == "not_overtaking"
    assert metadata["review_reason_codes"] == [
        "LEFT_LANE_DURATION_EXCEEDED",
        "RIGHT_LANE_AVAILABLE",
    ]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_cancelled_event_is_preserved_but_not_pending_review(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(writer_module, "_cv2", lambda: _FakeCV2)
    artifact_writer = EventArtifactWriter(
        run_directory=tmp_path,
        output_config=OutputConfig(
            clip_pre_event_seconds=0.5,
            clip_max_duration_seconds=2.0,
        ),
        video_info=VideoInfo(width=32, height=24, fps=2.0, frame_count=20),
        source_video_name="source.mp4",
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    artifact_writer.process_frame(
        frame,
        [
            CandidateTransition(
                transition="started",
                track_id=8,
                lane_id="left",
                start_timestamp_seconds=1.0,
                timestamp_seconds=3.0,
                duration_seconds=2.0,
                confidence_score=0.8,
                lifecycle_state="candidate_active",
                candidate_started_at=3.0,
            )
        ],
    )
    artifact_writer.process_frame(
        frame,
        [
            CandidateTransition(
                transition="cancelled",
                track_id=8,
                lane_id="left",
                start_timestamp_seconds=1.0,
                timestamp_seconds=4.0,
                duration_seconds=3.0,
                confidence_score=0.8,
                lifecycle_state="cancelled",
                candidate_started_at=3.0,
                cancelled_at=4.0,
                cancellation_reason="OVERTAKING_CONFIRMED",
                decision_history=(
                    CandidateDecisionRecord(
                        timestamp_seconds=3.0,
                        decision="candidate_started",
                        reason_codes=("LEFT_LANE_DURATION_EXCEEDED",),
                    ),
                    CandidateDecisionRecord(
                        timestamp_seconds=4.0,
                        decision="candidate_cancelled",
                        reason_codes=("OVERTAKING_CONFIRMED",),
                    ),
                ),
            )
        ],
    )

    metadata_path = (
        tmp_path / "events" / "left_lane_track_8_0000001000" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["review_status"] == "cancelled"
    assert metadata["candidate_lifecycle"]["cancellation_reason"] == (
        "OVERTAKING_CONFIRMED"
    )
    assert [item["decision"] for item in metadata["decision_history"]] == [
        "candidate_started",
        "candidate_cancelled",
    ]
    assert not (tmp_path / "events.jsonl").exists()
    assert len((tmp_path / "cancelled_events.jsonl").read_text().splitlines()) == 1
    assert artifact_writer.completed_count == 0
    assert artifact_writer.cancelled_count == 1

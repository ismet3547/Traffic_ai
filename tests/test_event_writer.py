from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import OutputConfig
from app.events import writer as writer_module
from app.events.writer import EventArtifactWriter
from app.models import CandidateTransition, VideoInfo


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
    assert metadata["review_status"] == "pending_human_review"
    assert metadata["human_review_required"] is True
    assert metadata["enforcement_action"] == "none"
    assert metadata["duration_seconds"] == 3.0
    assert metadata["end_reason"] == "left_lane_exit"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1

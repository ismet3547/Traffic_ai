"""Write representative images, bounded clips, and JSON review records."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import OutputConfig
from app.models import CandidateTransition, EventMetadata, VideoInfo

LOGGER = logging.getLogger(__name__)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return cv2


@dataclass(slots=True)
class _Recording:
    metadata: EventMetadata
    directory: Path
    writer: object
    written_frames: int = 0


class EventArtifactWriter:
    def __init__(
        self,
        run_directory: Path,
        output_config: OutputConfig,
        video_info: VideoInfo,
        source_video_name: str,
    ) -> None:
        self._run_directory = run_directory
        self._events_directory = run_directory / "events"
        self._events_directory.mkdir(parents=True, exist_ok=True)
        self._index_path = run_directory / "events.jsonl"
        self._config = output_config
        self._video_info = video_info
        self._source_video_name = source_video_name
        pre_event_frames = max(0, round(output_config.clip_pre_event_seconds * video_info.fps))
        self._prebuffer: deque[np.ndarray] = deque(maxlen=pre_event_frames)
        self._active: dict[int, _Recording] = {}
        self._completed_count = 0

    @property
    def completed_count(self) -> int:
        return self._completed_count

    def process_frame(
        self,
        frame: np.ndarray,
        transitions: list[CandidateTransition],
    ) -> None:
        for transition in transitions:
            if transition.transition == "started":
                self._start(transition, frame)

        for recording in self._active.values():
            self._write_clip_frame(recording, frame)

        for transition in transitions:
            if transition.transition == "ended":
                self._finish(transition)

        if self._prebuffer.maxlen:
            self._prebuffer.append(frame.copy())

    def finalize(self, transitions: list[CandidateTransition]) -> None:
        for transition in transitions:
            if transition.transition == "ended":
                self._finish(transition)
        for track_id in list(self._active):
            recording = self._active[track_id]
            fallback = CandidateTransition(
                transition="ended",
                track_id=track_id,
                lane_id=recording.metadata.lane_id,
                start_timestamp_seconds=recording.metadata.event_start_timestamp_seconds,
                timestamp_seconds=recording.metadata.candidate_created_timestamp_seconds,
                duration_seconds=recording.metadata.duration_seconds,
                confidence_score=recording.metadata.confidence_score,
                end_reason="pipeline_stopped",
            )
            self._finish(fallback)

    def _start(self, transition: CandidateTransition, frame: np.ndarray) -> None:
        if transition.track_id in self._active:
            return
        cv2 = _cv2()
        start_ms = round(transition.start_timestamp_seconds * 1000)
        event_id = f"left_lane_track_{transition.track_id}_{start_ms:010d}"
        event_directory = self._events_directory / event_id
        event_directory.mkdir(parents=True, exist_ok=False)
        image_path = event_directory / "representative.jpg"
        clip_path = event_directory / "event.mp4"
        image_params = [cv2.IMWRITE_JPEG_QUALITY, self._config.representative_image_quality]
        if not cv2.imwrite(str(image_path), frame, image_params):
            raise RuntimeError(f"could not write representative image: {image_path}")

        fourcc = cv2.VideoWriter_fourcc(*self._config.codec)
        writer = cv2.VideoWriter(
            str(clip_path),
            fourcc,
            self._video_info.fps,
            (self._video_info.width, self._video_info.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create event clip: {clip_path}")

        metadata = EventMetadata(
            event_id=event_id,
            track_id=transition.track_id,
            event_start_timestamp_seconds=transition.start_timestamp_seconds,
            candidate_created_timestamp_seconds=transition.timestamp_seconds,
            duration_seconds=transition.duration_seconds,
            lane_id=transition.lane_id,
            confidence_score=transition.confidence_score,
            source_video_name=self._source_video_name,
            representative_frame=image_path.name,
            event_video_clip=clip_path.name,
        )
        recording = _Recording(metadata=metadata, directory=event_directory, writer=writer)
        self._active[transition.track_id] = recording
        for buffered_frame in self._prebuffer:
            self._write_clip_frame(recording, buffered_frame)
        self._write_metadata(recording)
        LOGGER.info("Review candidate started: %s", event_id)

    def _write_clip_frame(self, recording: _Recording, frame: np.ndarray) -> None:
        max_frames = max(
            1, round(self._config.clip_max_duration_seconds * self._video_info.fps)
        )
        if recording.written_frames >= max_frames:
            return
        recording.writer.write(frame)
        recording.written_frames += 1

    def _finish(self, transition: CandidateTransition) -> None:
        recording = self._active.pop(transition.track_id, None)
        if recording is None:
            return
        recording.writer.release()
        recording.metadata.event_end_timestamp_seconds = transition.timestamp_seconds
        recording.metadata.duration_seconds = transition.duration_seconds
        recording.metadata.confidence_score = transition.confidence_score
        recording.metadata.end_reason = transition.end_reason
        self._write_metadata(recording)
        with self._index_path.open("a", encoding="utf-8") as index:
            index.write(recording.metadata.model_dump_json() + "\n")
        self._completed_count += 1
        LOGGER.info("Review candidate finalized: %s", recording.metadata.event_id)

    @staticmethod
    def _write_metadata(recording: _Recording) -> None:
        metadata_path = recording.directory / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(
                recording.metadata.model_dump(mode="json"),
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")

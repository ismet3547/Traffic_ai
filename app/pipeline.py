"""Core source-agnostic traffic analysis loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.detection import Detector
from app.events import EventArtifactWriter
from app.lanes import LaneAssigner
from app.rules import LeftLaneRuleEngine
from app.tracking import VehicleTracker
from app.video.annotation import DebugAnnotator
from app.video.protocols import FrameSource, VideoSink

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    frames_processed: int
    duration_seconds: float
    review_candidates: int


class TrafficAnalysisPipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        tracker: VehicleTracker,
        lane_assigner: LaneAssigner,
        rule_engine: LeftLaneRuleEngine,
        event_writer: EventArtifactWriter,
        annotator: DebugAnnotator,
        debug_sink: VideoSink | None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._lane_assigner = lane_assigner
        self._rule_engine = rule_engine
        self._event_writer = event_writer
        self._annotator = annotator
        self._debug_sink = debug_sink

    def run(self) -> PipelineSummary:
        frame_count = 0
        last_timestamp = 0.0
        try:
            for packet in self._source:
                detections = self._detector.detect(packet.image)
                vehicles = self._tracker.update(detections)
                observations = self._lane_assigner.assign(
                    vehicles,
                    frame_width=self._source.info.width,
                    frame_height=self._source.info.height,
                )
                evaluation = self._rule_engine.evaluate(
                    observations, packet.timestamp_seconds
                )
                annotated = self._annotator.annotate(
                    packet.image, observations, evaluation.statuses
                )
                self._event_writer.process_frame(annotated, evaluation.transitions)
                if self._debug_sink is not None:
                    self._debug_sink.write(annotated)
                frame_count += 1
                last_timestamp = packet.timestamp_seconds
                if frame_count % max(1, round(self._source.info.fps * 10)) == 0:
                    LOGGER.info(
                        "Processed %d frames (video time %.1fs)",
                        frame_count,
                        last_timestamp,
                    )

            self._event_writer.finalize(self._rule_engine.finalize())
        finally:
            # Also closes any partial candidate clip if inference is interrupted.
            self._event_writer.finalize([])
            self._source.close()
            if self._debug_sink is not None:
                self._debug_sink.close()

        return PipelineSummary(
            frames_processed=frame_count,
            duration_seconds=last_timestamp,
            review_candidates=self._event_writer.completed_count,
        )

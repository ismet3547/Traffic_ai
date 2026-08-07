"""Command-line entry point for prerecorded highway video analysis."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from app.config import AppConfig, load_config
from app.detection import UltralyticsDetector
from app.events import EventArtifactWriter
from app.lanes import LaneAssigner
from app.pipeline import TrafficAnalysisPipeline
from app.rules import LeftLaneRuleEngine
from app.tracking import ByteTrackVehicleTracker
from app.video import OpenCVVideoSink, OpenCVVideoSource
from app.video.annotation import DebugAnnotator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate human-review candidates for prolonged left-lane occupancy."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="MP4 path; overrides video.input_path")
    parser.add_argument("--output-dir", help="Base output path; overrides output.directory")
    parser.add_argument("--model", help="YOLO model path/name; overrides detector.model_path")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, 0, or cuda:0")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    config = _apply_overrides(config, args)
    input_path = Path(config.video.input_path).expanduser().resolve()
    output_base = Path(config.output.directory).expanduser().resolve()
    LOGGER.info("Input: %s", input_path)

    source = OpenCVVideoSource(input_path)
    try:
        run_directory = _make_run_directory(output_base, input_path.stem)
        LOGGER.info("Output: %s", run_directory)
        detector = UltralyticsDetector(config.detector)
        tracker = ByteTrackVehicleTracker(config.tracker, source.info.fps)
        lane_assigner = LaneAssigner(config.lanes)
        rule_engine = LeftLaneRuleEngine(config.rules.left_lane)
        annotator = DebugAnnotator(lane_assigner)
        event_writer = EventArtifactWriter(
            run_directory=run_directory,
            output_config=config.output,
            video_info=source.info,
            source_video_name=input_path.name,
        )
        debug_sink = (
            OpenCVVideoSink(
                run_directory / config.output.debug_video_name,
                source.info,
                config.output.codec,
            )
            if config.output.debug_video_enabled
            else None
        )
        pipeline = TrafficAnalysisPipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            lane_assigner=lane_assigner,
            rule_engine=rule_engine,
            event_writer=event_writer,
            annotator=annotator,
            debug_sink=debug_sink,
        )
        summary = pipeline.run()
    except Exception:
        # Pipeline.run also closes it; OpenCV release is safe to call twice.
        source.close()
        raise
    LOGGER.info(
        "Done: %d frames, %.1fs video, %d review candidate(s)",
        summary.frames_processed,
        summary.duration_seconds,
        summary.review_candidates,
    )
    return 0


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    video = config.video.model_copy(
        update={"input_path": args.input} if args.input else {}
    )
    detector_updates = {}
    if args.model:
        detector_updates["model_path"] = args.model
    if args.device:
        detector_updates["device"] = args.device
    detector = config.detector.model_copy(update=detector_updates)
    output = config.output.model_copy(
        update={"directory": args.output_dir} if args.output_dir else {}
    )
    return config.model_copy(
        update={"video": video, "detector": detector, "output": output}
    )


def _make_run_directory(base: Path, video_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{video_stem}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{video_stem}_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())

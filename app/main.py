"""Command-line entry point for prerecorded highway video analysis."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.camera_motion import CameraPoseValidator, build_camera_motion_estimator
from app.config import AppConfig, load_config
from app.context import RightLaneOpportunityTracker, TrafficContextAnalyzer
from app.detection import UltralyticsDetector
from app.events import EventArtifactWriter
from app.geometry import GeometryIntegrityPolicy
from app.lanes import LaneAssigner
from app.motion import LaneTransitionDetector, MotionHistoryStore
from app.overtaking import ContextualOvertakingPolicy, NoOvertakingPolicy
from app.physical_measurements import PhysicalMeasurementPolicy
from app.pipeline import TrafficAnalysisPipeline
from app.positioning import build_road_coordinate_transformer
from app.rules import ContextualLeftLaneDecisionPolicy, LeftLaneRuleEngine
from app.speed import RollingSpeedEstimator
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
    parser.add_argument(
        "--output-dir", help="Base output path; overrides output.directory"
    )
    parser.add_argument(
        "--model", help="YOLO model path/name; overrides detector.model_path"
    )
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
        lane_transition_detector = LaneTransitionDetector(config.lane_change)
        road_position_estimator = build_road_coordinate_transformer(
            config.calibration, config.road_position
        )
        calibration_status = road_position_estimator.calibration_quality
        LOGGER.info(
            "Calibration: mode=%s matrix_valid=%s numerically_stable=%s "
            "validation=%s confidence=%.2f basis=%s reasons=%s",
            calibration_status.mode,
            calibration_status.matrix_valid,
            calibration_status.numerically_stable,
            calibration_status.validation_mode,
            calibration_status.confidence,
            calibration_status.confidence_basis,
            ",".join(calibration_status.reason_codes) or "none",
        )
        if calibration_status.validation_mode == "FIT_POINTS_ONLY":
            LOGGER.warning(
                "Homography is mathematically solvable but physically unverified; "
                "physical speed/gap output is disabled by default"
            )
        speed_estimator = RollingSpeedEstimator(config.speed_estimation)
        camera_motion_estimator = build_camera_motion_estimator(config.camera_motion)
        camera_pose_validator = CameraPoseValidator(config.camera_pose_validation)
        physical_measurement_policy = PhysicalMeasurementPolicy(
            config.physical_measurements, config.calibration
        )
        geometry_integrity_policy = GeometryIntegrityPolicy(
            config.geometry_integrity, config.lanes
        )
        motion_history = MotionHistoryStore(config.traffic_context)
        traffic_context_analyzer = TrafficContextAnalyzer(
            config.lanes.lane_ids_left_to_right,
            config.traffic_context,
            config.congestion,
        )
        right_lane_opportunities = RightLaneOpportunityTracker(
            config.right_lane_opportunity
        )
        overtaking_policy = (
            ContextualOvertakingPolicy(
                config.overtaking,
                config.traffic_context,
                config.rules.left_lane.left_lane_id,
            )
            if config.rules.left_lane.overtaking_clearance_mode == "contextual"
            else NoOvertakingPolicy()
        )
        decision_policy = ContextualLeftLaneDecisionPolicy(
            config.rules.left_lane,
            config.traffic_context,
            config.right_lane_opportunity,
            config.calibration,
        )
        rule_engine = LeftLaneRuleEngine(
            config.rules.left_lane,
            decision_policy=decision_policy,
            lifecycle_config=config.candidate_lifecycle,
        )
        annotator = DebugAnnotator(lane_assigner, config.output)
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
            lane_transition_detector=lane_transition_detector,
            road_position_estimator=road_position_estimator,
            speed_estimator=speed_estimator,
            camera_motion_estimator=camera_motion_estimator,
            camera_pose_validator=camera_pose_validator,
            physical_measurement_policy=physical_measurement_policy,
            motion_history=motion_history,
            traffic_context_analyzer=traffic_context_analyzer,
            right_lane_opportunities=right_lane_opportunities,
            overtaking_policy=overtaking_policy,
            rule_engine=rule_engine,
            event_writer=event_writer,
            annotator=annotator,
            debug_sink=debug_sink,
            geometry_integrity_policy=geometry_integrity_policy,
        )
        summary = pipeline.run()
    except Exception:
        # Pipeline.run also closes it; OpenCV release is safe to call twice.
        source.close()
        raise
    LOGGER.info(
        "Done: %d frames, %.1fs video, %d review candidate(s), %d cancelled",
        summary.frames_processed,
        summary.duration_seconds,
        summary.review_candidates,
        summary.cancelled_candidates,
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{video_stem}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{video_stem}_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())

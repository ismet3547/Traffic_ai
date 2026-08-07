"""Bounded robust speed estimator over road-plane samples."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

import numpy as np

from app.config import SpeedEstimationConfig
from app.models import CameraMotionEstimate, RoadPosition, SpeedEstimate


@dataclass(frozen=True, slots=True)
class _PositionSample:
    timestamp_seconds: float
    world_xy: tuple[float, float] | None
    normalized_xy: tuple[float, float] | None


class RollingSpeedEstimator:
    """Estimates approximate speed only when calibrated meter positions exist."""

    def __init__(self, config: SpeedEstimationConfig) -> None:
        self._config = config
        self._histories: dict[int, deque[_PositionSample]] = {}

    def update(
        self,
        timestamp_seconds: float,
        positions: dict[int, RoadPosition],
        camera_motion: CameraMotionEstimate | None = None,
    ) -> dict[int, SpeedEstimate]:
        estimates: dict[int, SpeedEstimate] = {}
        for track_id, position in positions.items():
            estimates[track_id] = self._update_track(
                timestamp_seconds, position, camera_motion
            )
        cutoff = timestamp_seconds - self._config.maximum_window_seconds * 2.0
        for track_id, history in list(self._histories.items()):
            if (
                track_id not in positions
                and history
                and history[-1].timestamp_seconds < cutoff
            ):
                del self._histories[track_id]
        return estimates

    def _update_track(
        self,
        timestamp_seconds: float,
        position: RoadPosition,
        camera_motion: CameraMotionEstimate | None,
    ) -> SpeedEstimate:
        track_id = position.track_id
        history = self._histories.setdefault(track_id, deque())
        if history:
            gap = timestamp_seconds - history[-1].timestamp_seconds
            if gap <= 0:
                return self._unavailable(track_id, "invalid_timestamp", len(history))
            if gap > self._config.tracker_gap_grace_seconds:
                history.clear()

        sample = _PositionSample(
            timestamp_seconds=timestamp_seconds,
            world_xy=position.world_position,
            normalized_xy=position.normalized_position,
        )
        if position.world_position is not None and history and history[-1].world_xy:
            jump = math.dist(position.world_position, history[-1].world_xy)
            if jump > self._config.max_position_jump_meters:
                history.clear()
                history.append(sample)
                return self._unavailable(track_id, "rejected_position_jump", 1)
        history.append(sample)
        cutoff = timestamp_seconds - self._config.maximum_window_seconds
        while len(history) > 1 and history[0].timestamp_seconds < cutoff:
            history.popleft()

        if not self._config.enabled:
            return self._unavailable(track_id, "disabled", len(history))
        if (
            camera_motion is not None
            and camera_motion.valid
            and camera_motion.level == "high"
        ):
            return self._unavailable(
                track_id, "unavailable_camera_motion_high", len(history)
            )
        if not position.calibrated or position.world_position is None:
            rate = self._normalized_rate(history)
            return SpeedEstimate(
                track_id=track_id,
                speed_mps=None,
                speed_kph=None,
                speed_confidence=0.0,
                speed_mode="unavailable_uncalibrated",
                normalized_motion_rate=rate,
                sample_count=len(history),
            )
        if (
            len(history) < self._config.minimum_samples
            or history[-1].timestamp_seconds - history[0].timestamp_seconds
            < self._config.minimum_window_seconds
        ):
            return self._unavailable(track_id, "insufficient_history", len(history))

        speed_mps = self._world_speed(history)
        if speed_mps is None or not math.isfinite(speed_mps):
            return self._unavailable(track_id, "insufficient_history", len(history))
        speed_kph = speed_mps * 3.6
        if speed_kph > self._config.max_reasonable_speed_kph:
            history.clear()
            history.append(sample)
            return self._unavailable(track_id, "rejected_unreasonable_speed", 1)
        span = history[-1].timestamp_seconds - history[0].timestamp_seconds
        coverage = min(1.0, span / self._config.minimum_window_seconds)
        sample_quality = min(1.0, len(history) / self._config.minimum_samples)
        confidence = min(
            1.0, position.calibration_confidence * coverage * sample_quality
        )
        return SpeedEstimate(
            track_id=track_id,
            speed_mps=speed_mps,
            speed_kph=speed_kph,
            speed_confidence=confidence,
            speed_mode="approximate_calibrated",
            normalized_motion_rate=self._normalized_rate(history),
            sample_count=len(history),
        )

    def _world_speed(self, history: deque[_PositionSample]) -> float | None:
        samples = [sample for sample in history if sample.world_xy is not None]
        if len(samples) < 2:
            return None
        if self._config.smoothing == "linear_regression":
            times = np.asarray(
                [
                    sample.timestamp_seconds - samples[0].timestamp_seconds
                    for sample in samples
                ]
            )
            if float(np.ptp(times)) <= 0:
                return None
            xs = np.asarray([sample.world_xy[0] for sample in samples])  # type: ignore[index]
            ys = np.asarray([sample.world_xy[1] for sample in samples])  # type: ignore[index]
            vx = float(np.polyfit(times, xs, 1)[0])
            vy = float(np.polyfit(times, ys, 1)[0])
            return math.hypot(vx, vy)
        segment_speeds = []
        for previous, current in pairwise(samples):
            elapsed = current.timestamp_seconds - previous.timestamp_seconds
            if elapsed <= 0 or previous.world_xy is None or current.world_xy is None:
                continue
            segment_speeds.append(
                math.dist(previous.world_xy, current.world_xy) / elapsed
            )
        return median(segment_speeds) if segment_speeds else None

    @staticmethod
    def _normalized_rate(history: deque[_PositionSample]) -> float | None:
        rates: list[float] = []
        for previous, current in pairwise(history):
            elapsed = current.timestamp_seconds - previous.timestamp_seconds
            if (
                elapsed > 0
                and previous.normalized_xy is not None
                and current.normalized_xy is not None
            ):
                rates.append(
                    math.dist(previous.normalized_xy, current.normalized_xy) / elapsed
                )
        return median(rates) if rates else None

    @staticmethod
    def _unavailable(track_id: int, mode: str, samples: int) -> SpeedEstimate:
        return SpeedEstimate(
            track_id=track_id,
            speed_mps=None,
            speed_kph=None,
            speed_confidence=0.0,
            speed_mode=mode,
            normalized_motion_rate=None,
            sample_count=samples,
        )

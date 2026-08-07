"""Overtaking-policy contract, independent of traffic-rule decisions."""

from __future__ import annotations

from typing import Protocol

from app.models import (
    LaneObservation,
    LaneTransition,
    OvertakingAssessment,
    TrafficFrameContext,
)
from app.motion import MotionHistoryStore


class OvertakingClearancePolicy(Protocol):
    def update(
        self,
        timestamp_seconds: float,
        observations: list[LaneObservation],
        transitions: list[LaneTransition],
        context: TrafficFrameContext,
        history: MotionHistoryStore,
    ) -> dict[int, OvertakingAssessment]:
        """Assess overtaking behavior for visible tracks."""
        ...

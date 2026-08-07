"""Shared domain models."""

from .domain import (
    BoundingBox,
    CandidateTransition,
    Detection,
    EventMetadata,
    FramePacket,
    LaneObservation,
    RuleEvaluation,
    TrackedVehicle,
    VehicleRuleStatus,
    VideoInfo,
)

__all__ = [
    "BoundingBox",
    "CandidateTransition",
    "Detection",
    "EventMetadata",
    "FramePacket",
    "LaneObservation",
    "RuleEvaluation",
    "TrackedVehicle",
    "VehicleRuleStatus",
    "VideoInfo",
]

from __future__ import annotations

from app.candidates import CandidateLifecycleManager
from app.config import CandidateLifecycleConfig
from app.models import (
    BehaviorClassification,
    CandidateDecision,
    CandidateLifecycleState,
)


def _eligible() -> CandidateDecision:
    return CandidateDecision(
        True,
        BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION,
        0.9,
        ("LEFT_LANE_DURATION_EXCEEDED", "RIGHT_LANE_AVAILABLE"),
    )


def _suppressed(reason: str) -> CandidateDecision:
    return CandidateDecision(
        False,
        BehaviorClassification.INSUFFICIENT_EVIDENCE,
        0.7,
        suppression_reason=reason,
    )


def _manager(**updates: float) -> CandidateLifecycleManager:
    values = {
        "invalidation_grace_seconds": 2.0,
        "suspension_grace_seconds": 3.0,
        "evidence_settle_seconds": 2.0,
        "max_event_duration_seconds": 30.0,
        "restart_cooldown_seconds": 1.0,
    }
    values.update(updates)
    return CandidateLifecycleManager(CandidateLifecycleConfig(**values))


def test_candidate_starts_and_remains_active_while_collecting() -> None:
    manager = _manager()
    assert (
        manager.update(1, 0.0, _suppressed("DURATION_BELOW_THRESHOLD")).state
        == CandidateLifecycleState.ACCUMULATING
    )
    started = manager.update(1, 2.0, _eligible())
    collecting = manager.update(1, 20.0, _eligible())
    assert started.transition == "started"
    assert collecting.transition is None
    assert collecting.state == CandidateLifecycleState.CANDIDATE_ACTIVE


def test_candidate_suspends_then_resumes_within_grace() -> None:
    manager = _manager(suspension_grace_seconds=3.0)
    manager.update(1, 0.0, _eligible())
    suspended = manager.update(1, 1.0, _suppressed("RIGHT_LANE_UNAVAILABLE"))
    resumed = manager.update(1, 2.0, _eligible())
    assert suspended.transition == "suspended"
    assert resumed.transition == "resumed"


def test_persistent_congestion_cancels_candidate() -> None:
    manager = _manager(invalidation_grace_seconds=2.0)
    manager.update(1, 0.0, _eligible())
    manager.update(1, 1.0, _suppressed("CONGESTION"))
    cancelled = manager.update(1, 3.0, _suppressed("CONGESTION"))
    assert cancelled.transition == "cancelled"
    assert cancelled.cancellation_reason == "CONGESTION"


def test_close_enters_pending_state_and_only_finalizes_after_settle() -> None:
    manager = _manager(evidence_settle_seconds=2.0)
    manager.update(1, 0.0, _eligible())
    pending = manager.request_close(1, 1.0, "left_lane_exit")
    early = manager.update(1, 2.9, _eligible())
    final = manager.update(1, 3.0, _eligible())
    assert pending is not None and pending.transition == "pending_close"
    assert early.transition is None
    assert final.transition == "finalized"
    assert final.close_reason == "left_lane_exit"


def test_delayed_overtaking_confirmation_cancels_before_finalization() -> None:
    manager = _manager(evidence_settle_seconds=2.0)
    manager.update(1, 10.0, _eligible())
    manager.request_close(1, 15.0, "left_lane_exit")
    cancelled = manager.update(1, 16.0, _suppressed("OVERTAKING_CONFIRMED"))
    assert cancelled.transition == "cancelled"
    assert cancelled.cancellation_reason == "OVERTAKING_CONFIRMED"


def test_video_end_force_close_finalizes_eligible_active_event() -> None:
    manager = _manager()
    manager.update(1, 0.0, _eligible())
    final = manager.force_close(1, 4.0, "video_ended", _eligible())
    assert final is not None and final.transition == "finalized"


def test_finalized_candidate_is_immutable_and_idempotent() -> None:
    manager = _manager(evidence_settle_seconds=0.0)
    manager.update(1, 0.0, _eligible())
    manager.request_close(1, 1.0, "left_lane_exit")
    final = manager.update(1, 1.0, _eligible())
    later = manager.update(1, 2.0, _suppressed("OVERTAKING_CONFIRMED"))
    second_close = manager.force_close(1, 3.0, "video_ended", _eligible())
    assert final.transition == "finalized"
    assert later.transition is None and later.state == CandidateLifecycleState.FINALIZED
    assert second_close is not None and second_close.transition is None


def test_cancelled_episode_can_restart_after_cooldown() -> None:
    manager = _manager(invalidation_grace_seconds=0.5, restart_cooldown_seconds=0.5)
    manager.update(1, 0.0, _eligible())
    manager.update(1, 0.5, _suppressed("OVERTAKING_CONFIRMED"))
    manager.update(1, 1.0, _suppressed("OVERTAKING_CONFIRMED"))
    restarted = manager.update(1, 1.6, _eligible())
    assert restarted.transition == "started"
    assert restarted.candidate_started_at == 1.6

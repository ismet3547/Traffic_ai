"""Explicit, bounded lifecycle for review-candidate evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config import CandidateLifecycleConfig
from app.models import (
    CandidateDecision,
    CandidateDecisionRecord,
    CandidateLifecycleState,
)

_HARD_INVALIDATIONS = {
    "CONGESTION",
    "ACTIVE_OVERTAKE",
    "OVERTAKING_CONFIRMED",
    "CALIBRATION_UNRELIABLE",
    "CAMERA_MOTION_HIGH",
    "UNSTABLE_TRACK",
}


@dataclass(frozen=True, slots=True)
class LifecycleUpdate:
    state: CandidateLifecycleState
    transition: str | None
    candidate_started_at: float | None
    suspended_at: float | None
    finalized_at: float | None
    cancelled_at: float | None
    cancellation_reason: str | None
    decision_history: tuple[CandidateDecisionRecord, ...]


@dataclass(slots=True)
class _LifecycleRecord:
    state: CandidateLifecycleState = CandidateLifecycleState.IDLE
    candidate_started_at: float | None = None
    suspended_at: float | None = None
    invalid_since: float | None = None
    invalid_reason: str | None = None
    finalized_at: float | None = None
    cancelled_at: float | None = None
    cancellation_reason: str | None = None
    cooldown_until: float = 0.0
    decision_history: deque[CandidateDecisionRecord] = field(default_factory=deque)


class CandidateLifecycleManager:
    """Separates candidate start eligibility from continuation validity."""

    def __init__(self, config: CandidateLifecycleConfig) -> None:
        self._config = config
        self._records: dict[int, _LifecycleRecord] = {}

    def update(
        self,
        track_id: int,
        timestamp_seconds: float,
        decision: CandidateDecision,
    ) -> LifecycleUpdate:
        record = self._records.setdefault(track_id, self._new_record())
        transition: str | None = None

        if record.state == CandidateLifecycleState.IDLE:
            record.state = CandidateLifecycleState.ACCUMULATING
            self._append(record, timestamp_seconds, "evidence_accumulating", ())

        if record.state == CandidateLifecycleState.FINALIZED:
            return self._snapshot(record, None)

        if record.state == CandidateLifecycleState.CANCELLED:
            if timestamp_seconds < record.cooldown_until or not decision.eligible:
                return self._snapshot(record, None)
            self._reset_episode(record)
            record.state = CandidateLifecycleState.ACCUMULATING

        if record.state == CandidateLifecycleState.ACCUMULATING:
            if decision.eligible:
                record.state = CandidateLifecycleState.CANDIDATE_ACTIVE
                record.candidate_started_at = timestamp_seconds
                self._append(
                    record,
                    timestamp_seconds,
                    "candidate_started",
                    decision.reason_codes,
                )
                transition = "started"
            return self._snapshot(record, transition)

        if record.state == CandidateLifecycleState.CANDIDATE_ACTIVE:
            if not decision.eligible:
                record.state = CandidateLifecycleState.SUSPENDED
                record.suspended_at = timestamp_seconds
                record.invalid_since = timestamp_seconds
                record.invalid_reason = decision.suppression_reason
                self._append(
                    record,
                    timestamp_seconds,
                    "candidate_suspended",
                    _reason_tuple(decision.suppression_reason),
                )
                return self._snapshot(record, "suspended")
            if (
                record.candidate_started_at is not None
                and timestamp_seconds - record.candidate_started_at
                >= self._config.finalize_after_seconds
            ):
                record.state = CandidateLifecycleState.FINALIZED
                record.finalized_at = timestamp_seconds
                self._append(
                    record,
                    timestamp_seconds,
                    "candidate_finalized",
                    decision.reason_codes,
                )
                return self._snapshot(record, "finalized")
            return self._snapshot(record, None)

        if record.state == CandidateLifecycleState.SUSPENDED:
            grace = self._grace_for(record.invalid_reason)
            invalid_since = (
                record.invalid_since
                if record.invalid_since is not None
                else timestamp_seconds
            )
            invalid_duration = timestamp_seconds - invalid_since
            if decision.eligible and invalid_duration < grace - 1e-9:
                record.state = CandidateLifecycleState.CANDIDATE_ACTIVE
                record.invalid_since = None
                record.invalid_reason = None
                self._append(
                    record,
                    timestamp_seconds,
                    "candidate_resumed",
                    decision.reason_codes,
                )
                return self._snapshot(record, "resumed")
            if invalid_duration + 1e-9 >= grace:
                return self._cancel(record, timestamp_seconds, record.invalid_reason)
            return self._snapshot(record, None)

        return self._snapshot(record, transition)

    def close(
        self, track_id: int, timestamp_seconds: float, reason: str
    ) -> LifecycleUpdate | None:
        record = self._records.get(track_id)
        if record is None:
            return None
        if record.state == CandidateLifecycleState.CANDIDATE_ACTIVE:
            record.state = CandidateLifecycleState.FINALIZED
            record.finalized_at = timestamp_seconds
            self._append(record, timestamp_seconds, "candidate_finalized", (reason,))
            return self._snapshot(record, "finalized")
        if record.state == CandidateLifecycleState.SUSPENDED:
            return self._cancel(
                record, timestamp_seconds, record.invalid_reason or reason
            )
        return self._snapshot(record, None)

    def state(self, track_id: int) -> CandidateLifecycleState:
        record = self._records.get(track_id)
        return record.state if record else CandidateLifecycleState.IDLE

    def remove(self, track_id: int) -> None:
        self._records.pop(track_id, None)

    def _cancel(
        self,
        record: _LifecycleRecord,
        timestamp_seconds: float,
        reason: str | None,
    ) -> LifecycleUpdate:
        record.state = CandidateLifecycleState.CANCELLED
        record.cancelled_at = timestamp_seconds
        record.cancellation_reason = reason or "INSUFFICIENT_CONTEXT"
        record.cooldown_until = (
            timestamp_seconds + self._config.restart_cooldown_seconds
        )
        self._append(
            record,
            timestamp_seconds,
            "candidate_cancelled",
            _reason_tuple(record.cancellation_reason),
        )
        return self._snapshot(record, "cancelled")

    def _grace_for(self, reason: str | None) -> float:
        return (
            self._config.invalidation_grace_seconds
            if reason in _HARD_INVALIDATIONS
            else self._config.suspension_grace_seconds
        )

    def _new_record(self) -> _LifecycleRecord:
        return _LifecycleRecord(
            decision_history=deque(maxlen=self._config.maximum_decision_history_entries)
        )

    @staticmethod
    def _reset_episode(record: _LifecycleRecord) -> None:
        record.candidate_started_at = None
        record.suspended_at = None
        record.invalid_since = None
        record.invalid_reason = None
        record.finalized_at = None
        record.cancelled_at = None
        record.cancellation_reason = None
        record.decision_history.clear()

    @staticmethod
    def _append(
        record: _LifecycleRecord,
        timestamp_seconds: float,
        decision: str,
        reasons: tuple[str, ...],
    ) -> None:
        record.decision_history.append(
            CandidateDecisionRecord(timestamp_seconds, decision, reasons)
        )

    @staticmethod
    def _snapshot(record: _LifecycleRecord, transition: str | None) -> LifecycleUpdate:
        return LifecycleUpdate(
            state=record.state,
            transition=transition,
            candidate_started_at=record.candidate_started_at,
            suspended_at=record.suspended_at,
            finalized_at=record.finalized_at,
            cancelled_at=record.cancelled_at,
            cancellation_reason=record.cancellation_reason,
            decision_history=tuple(record.decision_history),
        )


def _reason_tuple(reason: str | None) -> tuple[str, ...]:
    return (reason,) if reason else ()

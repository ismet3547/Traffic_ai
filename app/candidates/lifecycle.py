"""Explicit, bounded lifecycle for review-candidate evidence."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from app.config import CandidateLifecycleConfig
from app.models import (
    CandidateDecision,
    CandidateDecisionRecord,
    CandidateLifecycleState,
)

LOGGER = logging.getLogger(__name__)

_HARD_INVALIDATIONS = {
    "CONGESTION",
    "ACTIVE_OVERTAKE",
    "OVERTAKING_CONFIRMED",
    "CALIBRATION_UNRELIABLE",
    "CAMERA_MOTION_HIGH",
    "UNSTABLE_TRACK",
    "GEOMETRY_INTEGRITY_LOST",
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
    close_requested_at: float | None
    close_reason: str | None
    decision_history: tuple[CandidateDecisionRecord, ...]


@dataclass(slots=True)
class _LifecycleRecord:
    state: CandidateLifecycleState = CandidateLifecycleState.IDLE
    candidate_started_at: float | None = None
    suspended_at: float | None = None
    invalid_since: float | None = None
    invalid_reason: str | None = None
    close_requested_at: float | None = None
    close_reason: str | None = None
    finalized_at: float | None = None
    cancelled_at: float | None = None
    cancellation_reason: str | None = None
    cooldown_until: float = 0.0
    latest_decision: CandidateDecision | None = None
    decision_history: deque[CandidateDecisionRecord] = field(default_factory=deque)


class CandidateLifecycleManager:
    """Collect until a real close trigger, settle, then make one terminal decision."""

    def __init__(self, config: CandidateLifecycleConfig) -> None:
        self._config = config
        self._records: dict[int, _LifecycleRecord] = {}

    def update(
        self, track_id: int, timestamp_seconds: float, decision: CandidateDecision
    ) -> LifecycleUpdate:
        record = self._records.setdefault(track_id, self._new_record())
        record.latest_decision = decision
        if record.state in {
            CandidateLifecycleState.FINALIZED,
            CandidateLifecycleState.CANCELLED,
        }:
            if (
                record.state == CandidateLifecycleState.CANCELLED
                and timestamp_seconds >= record.cooldown_until
                and decision.eligible
            ):
                self._reset_episode(record)
                record.state = CandidateLifecycleState.ACCUMULATING
            else:
                return self._snapshot(record, None)

        if record.state == CandidateLifecycleState.IDLE:
            record.state = CandidateLifecycleState.ACCUMULATING
            self._append(record, timestamp_seconds, "evidence_accumulating", ())

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
                return self._snapshot(record, "started")
            return self._snapshot(record, None)

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
                >= self._config.max_event_duration_seconds
            ):
                return self.request_close(
                    track_id, timestamp_seconds, "maximum_evidence_window"
                )  # type: ignore[return-value]
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

        if record.state == CandidateLifecycleState.PENDING_CLOSE:
            if decision.suppression_reason == "GEOMETRY_INTEGRITY_LOST":
                if record.invalid_since is None:
                    record.invalid_since = timestamp_seconds
                    record.invalid_reason = "GEOMETRY_INTEGRITY_LOST"
                    record.suspended_at = timestamp_seconds
                    self._append(
                        record,
                        timestamp_seconds,
                        "candidate_suspended",
                        ("GEOMETRY_INTEGRITY_LOST",),
                    )
                    return self._snapshot(record, "suspended")
                if (
                    timestamp_seconds - record.invalid_since + 1e-9
                    >= self._config.invalidation_grace_seconds
                ):
                    return self._cancel(
                        record, timestamp_seconds, "GEOMETRY_INTEGRITY_LOST"
                    )
                return self._snapshot(record, None)
            if decision.suppression_reason == "OVERTAKING_CONFIRMED":
                return self._cancel(record, timestamp_seconds, "OVERTAKING_CONFIRMED")
            close_at = (
                record.close_requested_at
                if record.close_requested_at is not None
                else timestamp_seconds
            )
            if (
                timestamp_seconds - close_at + 1e-9
                < self._config.evidence_settle_seconds
            ):
                return self._snapshot(record, None)
            if decision.eligible:
                return self._finalize(record, timestamp_seconds)
            return self._cancel(
                record,
                timestamp_seconds,
                decision.suppression_reason
                or record.invalid_reason
                or "INSUFFICIENT_CONTEXT",
            )

        return self._snapshot(record, None)

    def request_close(
        self, track_id: int, timestamp_seconds: float, reason: str
    ) -> LifecycleUpdate | None:
        record = self._records.get(track_id)
        if record is None:
            return None
        if record.state in {
            CandidateLifecycleState.FINALIZED,
            CandidateLifecycleState.CANCELLED,
        }:
            return self._snapshot(record, None)
        if record.state not in {
            CandidateLifecycleState.CANDIDATE_ACTIVE,
            CandidateLifecycleState.SUSPENDED,
            CandidateLifecycleState.PENDING_CLOSE,
        }:
            return self._snapshot(record, None)
        if record.state != CandidateLifecycleState.PENDING_CLOSE:
            record.state = CandidateLifecycleState.PENDING_CLOSE
            record.close_requested_at = timestamp_seconds
            record.close_reason = reason
            self._append(
                record, timestamp_seconds, "candidate_close_requested", (reason,)
            )
            return self._snapshot(record, "pending_close")
        return self._snapshot(record, None)

    def force_close(
        self,
        track_id: int,
        timestamp_seconds: float,
        reason: str,
        decision: CandidateDecision | None = None,
    ) -> LifecycleUpdate | None:
        """Deterministic end-of-source close when no later evidence can arrive."""

        record = self._records.get(track_id)
        if record is None:
            return None
        if record.state in {
            CandidateLifecycleState.FINALIZED,
            CandidateLifecycleState.CANCELLED,
        }:
            return self._snapshot(record, None)
        final_decision = decision or record.latest_decision
        if record.candidate_started_at is None:
            return self._snapshot(record, None)
        if final_decision is not None and final_decision.eligible:
            record.close_requested_at = record.close_requested_at or timestamp_seconds
            record.close_reason = record.close_reason or reason
            return self._finalize(record, timestamp_seconds)
        return self._cancel(
            record,
            timestamp_seconds,
            (final_decision.suppression_reason if final_decision else None)
            or record.invalid_reason
            or reason,
        )

    # Compatibility alias: close now requests a close; it is not terminal.
    def close(
        self, track_id: int, timestamp_seconds: float, reason: str
    ) -> LifecycleUpdate | None:
        return self.request_close(track_id, timestamp_seconds, reason)

    def state(self, track_id: int) -> CandidateLifecycleState:
        record = self._records.get(track_id)
        return record.state if record else CandidateLifecycleState.IDLE

    def remove(self, track_id: int) -> None:
        self._records.pop(track_id, None)

    def _finalize(
        self, record: _LifecycleRecord, timestamp_seconds: float
    ) -> LifecycleUpdate:
        record.state = CandidateLifecycleState.FINALIZED
        record.finalized_at = timestamp_seconds
        reasons = _reason_tuple(record.close_reason)
        self._append(record, timestamp_seconds, "candidate_finalized", reasons)
        return self._snapshot(record, "finalized")

    def _cancel(
        self, record: _LifecycleRecord, timestamp_seconds: float, reason: str | None
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
        if record.cancellation_reason == "OVERTAKING_CONFIRMED":
            LOGGER.warning(
                "Review candidate cancelled by later exculpatory overtaking evidence"
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
        record.close_requested_at = None
        record.close_reason = None
        record.finalized_at = None
        record.cancelled_at = None
        record.cancellation_reason = None
        record.latest_decision = None
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
            close_requested_at=record.close_requested_at,
            close_reason=record.close_reason,
            decision_history=tuple(record.decision_history),
        )


def _reason_tuple(reason: str | None) -> tuple[str, ...]:
    return (reason,) if reason else ()

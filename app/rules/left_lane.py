"""Stateful left-lane rule with an explicit candidate evidence lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from app.candidates import CandidateLifecycleManager, LifecycleUpdate
from app.config import CandidateLifecycleConfig, LeftLaneRuleConfig
from app.models import (
    BehaviorClassification,
    CandidateDecision,
    CandidateLifecycleState,
    CandidateTransition,
    LaneObservation,
    OvertakeState,
    OvertakingAssessment,
    OvertakingStatus,
    RuleEvaluation,
    SpeedEstimate,
    TrafficFrameContext,
    VehicleRuleStatus,
    VehicleTrafficContext,
)
from app.rules.policy import ContextualLeftLaneDecisionPolicy


class NoOvertakingClearancePolicy:
    """Deprecated Phase 1 compatibility shim."""

    def is_cleared(
        self, observation: LaneObservation, timestamp_seconds: float
    ) -> bool:
        del observation, timestamp_seconds
        return False


@dataclass(slots=True)
class _LeftLaneState:
    entered_at: float
    last_seen_at: float
    confidence_sum: float
    observation_count: int
    last_overtake_completed_at: float | None = None
    candidate_evidence_started_at: float | None = None
    last_candidate_ended_at: float | None = None
    last_decision: CandidateDecision | None = None
    latest_context: TrafficFrameContext | None = None
    latest_assessment: OvertakingAssessment | None = None

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.observation_count


class LeftLaneRuleEngine:
    def __init__(
        self,
        config: LeftLaneRuleConfig,
        decision_policy: ContextualLeftLaneDecisionPolicy | None = None,
        lifecycle_config: CandidateLifecycleConfig | None = None,
    ) -> None:
        self._config = config
        self._decision_policy = decision_policy
        self._legacy_mode = decision_policy is None and lifecycle_config is None
        self._lifecycle = CandidateLifecycleManager(
            lifecycle_config or CandidateLifecycleConfig()
        )
        self._states: dict[int, _LeftLaneState] = {}

    def evaluate(
        self,
        observations: list[LaneObservation],
        timestamp_seconds: float,
        traffic_context: TrafficFrameContext | None = None,
        overtaking_assessments: dict[int, OvertakingAssessment] | None = None,
        history_durations: dict[int, float] | None = None,
    ) -> RuleEvaluation:
        statuses: dict[int, VehicleRuleStatus] = {}
        transitions: list[CandidateTransition] = []
        seen_ids = {observation.vehicle.track_id for observation in observations}
        assessments = overtaking_assessments or {}
        durations = history_durations or {}

        if not self._config.enabled:
            self._states.clear()
            for observation in observations:
                statuses[observation.vehicle.track_id] = self._status_outside(
                    observation,
                    assessments.get(observation.vehicle.track_id),
                    traffic_context,
                )
            return RuleEvaluation(
                statuses=statuses,
                transitions=[],
                traffic_context=traffic_context.global_context
                if traffic_context
                else None,
            )

        for observation in observations:
            track_id = observation.vehicle.track_id
            assessment = assessments.get(track_id)
            if observation.lane_id != self._config.left_lane_id:
                transition = self._close_state(
                    track_id, timestamp_seconds, "left_lane_exit"
                )
                if transition:
                    transitions.append(transition)
                statuses[track_id] = self._status_outside(
                    observation, assessment, traffic_context
                )
                continue

            state = self._states.get(track_id)
            if state is None:
                state = _LeftLaneState(
                    entered_at=timestamp_seconds,
                    last_seen_at=timestamp_seconds,
                    confidence_sum=observation.vehicle.confidence,
                    observation_count=1,
                )
                self._states[track_id] = state
            else:
                state.last_seen_at = timestamp_seconds
                state.confidence_sum += observation.vehicle.confidence
                state.observation_count += 1

            state.latest_context = traffic_context
            state.latest_assessment = assessment
            duration = max(0.0, timestamp_seconds - state.entered_at)
            vehicle_context = (
                traffic_context.vehicles.get(track_id) if traffic_context else None
            )
            speed = (
                (traffic_context.speeds or {}).get(track_id)
                if traffic_context is not None
                else None
            )
            decision = self._decide(
                duration,
                state.mean_confidence,
                durations.get(track_id, 0.0),
                traffic_context,
                vehicle_context,
                assessment,
                speed,
            )
            state.last_decision = decision
            lifecycle = self._lifecycle.update(track_id, timestamp_seconds, decision)
            if lifecycle.transition == "started":
                state.candidate_evidence_started_at = max(
                    state.entered_at, state.last_candidate_ended_at or state.entered_at
                )
            if lifecycle.transition is not None:
                transitions.append(
                    self._transition(
                        lifecycle,
                        track_id,
                        state,
                        timestamp_seconds,
                        duration,
                    )
                )
                if lifecycle.transition in {"cancelled", "finalized"}:
                    state.last_candidate_ended_at = timestamp_seconds

            if (
                assessment is not None
                and assessment.status == OvertakingStatus.OVERTAKING_CONFIRMED
                and assessment.completed_at is not None
                and assessment.completed_at != state.last_overtake_completed_at
            ):
                state.entered_at = assessment.completed_at
                state.confidence_sum = observation.vehicle.confidence
                state.observation_count = 1
                state.last_overtake_completed_at = assessment.completed_at

            lifecycle_state = lifecycle.state
            is_candidate = lifecycle_state in {
                CandidateLifecycleState.CANDIDATE_ACTIVE,
                CandidateLifecycleState.SUSPENDED,
                CandidateLifecycleState.FINALIZED,
            }
            right_gap = None
            if vehicle_context is not None:
                right_gap = vehicle_context.right_lane_front_gap
                if right_gap is None:
                    right_gap = vehicle_context.right_lane_rear_gap
            position = (
                traffic_context.positions.get(track_id) if traffic_context else None
            )
            statuses[track_id] = VehicleRuleStatus(
                track_id=track_id,
                lane_id=observation.lane_id,
                left_lane_duration_seconds=duration,
                is_review_candidate=is_candidate,
                behavior_classification=(
                    BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION.value
                    if is_candidate
                    else decision.classification.value
                ),
                suppression_reason=(
                    decision.suppression_reason
                    if lifecycle_state == CandidateLifecycleState.SUSPENDED
                    or not is_candidate
                    else None
                ),
                overtake_state=assessment.state.value
                if assessment
                else OvertakeState.NONE.value,
                overtaking_status=(
                    assessment.status.value
                    if assessment
                    else OvertakingStatus.NOT_ASSESSED.value
                ),
                right_lane_available_seconds=(
                    vehicle_context.right_lane_available_seconds
                    if vehicle_context
                    else 0.0
                ),
                evidence_confidence=decision.evidence_confidence,
                related_track_ids=assessment.related_track_ids if assessment else (),
                candidate_lifecycle_state=lifecycle_state.value,
                speed_kph=speed.speed_kph if speed else None,
                speed_mode=speed.speed_mode if speed else "unavailable_uncalibrated",
                coordinate_mode=position.coordinate_mode
                if position
                else "normalized_image",
                right_lane_gap=right_gap,
            )

        for track_id, state in list(self._states.items()):
            if track_id in seen_ids:
                continue
            if (
                timestamp_seconds - state.last_seen_at
                >= self._config.track_lost_grace_seconds
            ):
                transition = self._close_state(
                    track_id, state.last_seen_at, "track_lost"
                )
                if transition:
                    transitions.append(transition)

        return RuleEvaluation(
            statuses=statuses,
            transitions=transitions,
            traffic_context=traffic_context.global_context if traffic_context else None,
        )

    def finalize(self) -> list[CandidateTransition]:
        transitions: list[CandidateTransition] = []
        for track_id, state in list(self._states.items()):
            transition = self._close_state(track_id, state.last_seen_at, "video_ended")
            if transition:
                transitions.append(transition)
        return transitions

    def _decide(
        self,
        duration: float,
        mean_confidence: float,
        history_duration: float,
        traffic_context: TrafficFrameContext | None,
        vehicle_context: VehicleTrafficContext | None,
        assessment: OvertakingAssessment | None,
        speed: SpeedEstimate | None,
    ) -> CandidateDecision:
        if self._decision_policy is not None:
            return self._decision_policy.decide(
                left_lane_duration_seconds=duration,
                mean_detector_confidence=mean_confidence,
                history_duration_seconds=history_duration,
                traffic=traffic_context.global_context if traffic_context else None,
                vehicle_context=vehicle_context,
                overtaking=assessment,
                speed=speed,
            )
        eligible = (
            duration >= self._config.occupancy_threshold_seconds
            and mean_confidence >= self._config.minimum_mean_confidence
        )
        return CandidateDecision(
            eligible=eligible,
            classification=(
                BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION
                if eligible
                else BehaviorClassification.TEMPORARY_LEFT_LANE_USE
            ),
            evidence_confidence=mean_confidence,
            reason_codes=("LEFT_LANE_DURATION_EXCEEDED",) if eligible else (),
            suppression_reason=None if eligible else "DURATION_BELOW_THRESHOLD",
        )

    def _transition(
        self,
        lifecycle: LifecycleUpdate,
        track_id: int,
        state: _LeftLaneState,
        timestamp_seconds: float,
        duration: float,
        end_reason: str | None = None,
    ) -> CandidateTransition:
        frame_context = state.latest_context
        kind = lifecycle.transition or "suspended"
        if self._legacy_mode and kind == "finalized":
            kind = "ended"
        if end_reason is None and lifecycle.transition == "finalized":
            end_reason = "evidence_window_complete"
        if end_reason is None and lifecycle.transition == "cancelled":
            end_reason = lifecycle.cancellation_reason
        position = frame_context.positions.get(track_id) if frame_context else None
        speed = (frame_context.speeds or {}).get(track_id) if frame_context else None
        evidence_start = (
            state.candidate_evidence_started_at
            if state.candidate_evidence_started_at is not None
            else state.entered_at
        )
        return CandidateTransition(
            transition=kind,
            track_id=track_id,
            lane_id=self._config.left_lane_id,
            start_timestamp_seconds=evidence_start,
            timestamp_seconds=timestamp_seconds,
            duration_seconds=max(duration, timestamp_seconds - evidence_start),
            confidence_score=state.mean_confidence,
            end_reason=end_reason,
            review_reason_codes=state.last_decision.reason_codes
            if state.last_decision
            else (),
            policy_version=self._config.policy_version,
            traffic_context=frame_context.global_context if frame_context else None,
            vehicle_traffic_context=(
                frame_context.vehicles.get(track_id) if frame_context else None
            ),
            overtaking_assessment=state.latest_assessment,
            behavior_classification=(
                state.last_decision.classification.value
                if state.last_decision
                else None
            ),
            evidence_confidence_score=(
                state.last_decision.evidence_confidence if state.last_decision else None
            ),
            lifecycle_state=lifecycle.state.value,
            candidate_started_at=lifecycle.candidate_started_at,
            suspended_at=lifecycle.suspended_at,
            finalized_at=lifecycle.finalized_at,
            cancelled_at=lifecycle.cancelled_at,
            cancellation_reason=lifecycle.cancellation_reason,
            decision_history=lifecycle.decision_history,
            position=position,
            speed_estimate=speed,
            calibration_status=(
                frame_context.global_context.calibration_status
                if frame_context
                else None
            ),
            camera_motion=(
                frame_context.global_context.camera_motion if frame_context else None
            ),
        )

    def _close_state(
        self, track_id: int, timestamp_seconds: float, reason: str
    ) -> CandidateTransition | None:
        state = self._states.pop(track_id, None)
        if state is None:
            return None
        lifecycle = self._lifecycle.close(track_id, timestamp_seconds, reason)
        transition = None
        if lifecycle is not None and lifecycle.transition is not None:
            transition = self._transition(
                lifecycle,
                track_id,
                state,
                timestamp_seconds,
                max(0.0, timestamp_seconds - state.entered_at),
                reason,
            )
        self._lifecycle.remove(track_id)
        return transition

    @staticmethod
    def _status_outside(
        observation: LaneObservation,
        assessment: OvertakingAssessment | None,
        traffic_context: TrafficFrameContext | None,
    ) -> VehicleRuleStatus:
        vehicle_context = (
            traffic_context.vehicles.get(observation.vehicle.track_id)
            if traffic_context
            else None
        )
        position = (
            traffic_context.positions.get(observation.vehicle.track_id)
            if traffic_context
            else None
        )
        speed = (
            (traffic_context.speeds or {}).get(observation.vehicle.track_id)
            if traffic_context
            else None
        )
        return VehicleRuleStatus(
            track_id=observation.vehicle.track_id,
            lane_id=observation.lane_id,
            left_lane_duration_seconds=0.0,
            is_review_candidate=False,
            behavior_classification=BehaviorClassification.TEMPORARY_LEFT_LANE_USE.value,
            overtake_state=assessment.state.value
            if assessment
            else OvertakeState.NONE.value,
            overtaking_status=(
                assessment.status.value
                if assessment
                else OvertakingStatus.NOT_ASSESSED.value
            ),
            right_lane_available_seconds=(
                vehicle_context.right_lane_available_seconds if vehicle_context else 0.0
            ),
            related_track_ids=assessment.related_track_ids if assessment else (),
            candidate_lifecycle_state=CandidateLifecycleState.IDLE.value,
            speed_kph=speed.speed_kph if speed else None,
            speed_mode=speed.speed_mode if speed else "unavailable_uncalibrated",
            coordinate_mode=position.coordinate_mode
            if position
            else "normalized_image",
        )

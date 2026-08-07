"""Stateful left-lane rule with close-and-settle candidate semantics."""

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
    PhysicalMeasurementPermission,
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
    close_reason: str | None = None

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
        effective_lifecycle = lifecycle_config or CandidateLifecycleConfig(
            evidence_settle_seconds=0.0 if self._legacy_mode else 2.0
        )
        self._lifecycle_config = effective_lifecycle
        self._lifecycle = CandidateLifecycleManager(effective_lifecycle)
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
        assessments = overtaking_assessments or {}
        durations = history_durations or {}
        observation_by_id = {item.vehicle.track_id: item for item in observations}

        if not self._config.enabled:
            self._states.clear()
            return RuleEvaluation(
                statuses={
                    item.vehicle.track_id: self._outside_status(
                        item, assessments.get(item.vehicle.track_id), traffic_context
                    )
                    for item in observations
                },
                transitions=[],
                traffic_context=traffic_context.global_context
                if traffic_context
                else None,
            )

        for observation in observations:
            track_id = observation.vehicle.track_id
            assessment = assessments.get(track_id)
            state = self._states.get(track_id)
            if observation.lane_id != self._config.left_lane_id:
                if state is None:
                    statuses[track_id] = self._outside_status(
                        observation, assessment, traffic_context
                    )
                    continue
                state.last_seen_at = timestamp_seconds
                state.latest_context = traffic_context
                state.latest_assessment = assessment
                decision = self._decision_for_existing(
                    state,
                    track_id,
                    timestamp_seconds,
                    traffic_context,
                    assessment,
                    durations,
                )
                geometry = (
                    traffic_context.global_context.geometry_integrity
                    if traffic_context
                    else None
                )
                if geometry is not None and not geometry.lane_assignment_allowed:
                    state.last_decision = decision
                    if self._lifecycle.state(track_id) in {
                        CandidateLifecycleState.IDLE,
                        CandidateLifecycleState.ACCUMULATING,
                    }:
                        # Unverified time cannot contribute to a later left-lane
                        # duration finding after geometry recovers.
                        state.entered_at = timestamp_seconds
                        state.confidence_sum = observation.vehicle.confidence
                        state.observation_count = 1
                    lifecycle = self._lifecycle.update(
                        track_id, timestamp_seconds, decision
                    )
                    if lifecycle.transition is not None:
                        transitions.append(
                            self._transition(
                                lifecycle,
                                track_id,
                                state,
                                timestamp_seconds,
                                timestamp_seconds - state.entered_at,
                            )
                        )
                    statuses[track_id] = self._status(
                        observation, state, lifecycle.state, decision
                    )
                    if lifecycle.state == CandidateLifecycleState.CANCELLED:
                        self._states.pop(track_id, None)
                        self._lifecycle.remove(track_id)
                    continue
                if (
                    state.last_decision is not None
                    and state.last_decision.eligible
                    and decision.suppression_reason
                    in {
                        "DURATION_BELOW_THRESHOLD",
                        "RIGHT_LANE_UNAVAILABLE",
                        "INSUFFICIENT_CONTEXT",
                    }
                ):
                    # Lane exit is a close trigger, not by itself exculpatory.
                    # Retain the last eligible assessment while allowing later
                    # congestion/overtaking/camera evidence to invalidate it.
                    decision = state.last_decision
                state.last_decision = decision
                transitions.extend(
                    self._advance_and_close(
                        track_id, state, timestamp_seconds, decision, "left_lane_exit"
                    )
                )
                lifecycle_state = self._lifecycle.state(track_id)
                statuses[track_id] = self._status(
                    observation, state, lifecycle_state, decision
                )
                if lifecycle_state in {
                    CandidateLifecycleState.CANCELLED,
                    CandidateLifecycleState.FINALIZED,
                }:
                    self._states.pop(track_id, None)
                    self._lifecycle.remove(track_id)
                continue

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
            decision = self._decision_for_existing(
                state,
                track_id,
                timestamp_seconds,
                traffic_context,
                assessment,
                durations,
            )
            state.last_decision = decision
            lifecycle = self._lifecycle.update(track_id, timestamp_seconds, decision)
            if lifecycle.transition == "started":
                state.candidate_evidence_started_at = max(
                    state.entered_at, state.last_candidate_ended_at or state.entered_at
                )
            if lifecycle.transition == "pending_close":
                state.close_reason = lifecycle.close_reason
            if lifecycle.transition is not None:
                transitions.append(
                    self._transition(
                        lifecycle, track_id, state, timestamp_seconds, duration
                    )
                )
            if lifecycle.state in {
                CandidateLifecycleState.CANCELLED,
                CandidateLifecycleState.FINALIZED,
            }:
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
            statuses[track_id] = self._status(
                observation, state, lifecycle.state, decision
            )

        for track_id, state in list(self._states.items()):
            if track_id in observation_by_id:
                continue
            missing_for = timestamp_seconds - state.last_seen_at
            close_after = (
                self._config.track_lost_grace_seconds
                if self._legacy_mode
                else max(
                    self._config.track_lost_grace_seconds,
                    self._lifecycle_config.track_loss_close_seconds,
                )
            )
            if missing_for + 1e-9 < close_after and state.close_reason is None:
                continue
            decision = state.last_decision or self._legacy_decision(
                state, timestamp_seconds
            )
            transitions.extend(
                self._advance_and_close(
                    track_id,
                    state,
                    state.last_seen_at if self._legacy_mode else timestamp_seconds,
                    decision,
                    "track_lost",
                )
            )
            if self._lifecycle.state(track_id) in {
                CandidateLifecycleState.CANCELLED,
                CandidateLifecycleState.FINALIZED,
            }:
                self._states.pop(track_id, None)
                self._lifecycle.remove(track_id)

        for transition in transitions:
            state = self._states.get(transition.track_id)
            terminal_closed_episode = transition.transition in {
                "finalized",
                "ended",
            } or (
                transition.transition == "cancelled"
                and state is not None
                and state.close_reason is not None
            )
            if terminal_closed_episode:
                self._states.pop(transition.track_id, None)
                self._lifecycle.remove(transition.track_id)

        return RuleEvaluation(
            statuses=statuses,
            transitions=transitions,
            traffic_context=traffic_context.global_context if traffic_context else None,
        )

    def finalize(self) -> list[CandidateTransition]:
        transitions: list[CandidateTransition] = []
        for track_id, state in list(self._states.items()):
            lifecycle = self._lifecycle.force_close(
                track_id, state.last_seen_at, "video_ended", state.last_decision
            )
            if lifecycle is not None and lifecycle.transition is not None:
                transitions.append(
                    self._transition(
                        lifecycle,
                        track_id,
                        state,
                        state.last_seen_at,
                        max(0.0, state.last_seen_at - state.entered_at),
                        "video_ended",
                    )
                )
            self._states.pop(track_id, None)
            self._lifecycle.remove(track_id)
        return transitions

    def _advance_and_close(
        self,
        track_id: int,
        state: _LeftLaneState,
        timestamp_seconds: float,
        decision: CandidateDecision,
        reason: str,
    ) -> list[CandidateTransition]:
        output: list[CandidateTransition] = []
        lifecycle = self._lifecycle.update(track_id, timestamp_seconds, decision)
        if lifecycle.transition is not None:
            output.append(
                self._transition(
                    lifecycle,
                    track_id,
                    state,
                    timestamp_seconds,
                    timestamp_seconds - state.entered_at,
                )
            )
        if lifecycle.state not in {
            CandidateLifecycleState.FINALIZED,
            CandidateLifecycleState.CANCELLED,
        }:
            close = self._lifecycle.request_close(track_id, timestamp_seconds, reason)
            if close is not None and close.transition is not None:
                state.close_reason = reason
                if not self._legacy_mode:
                    output.append(
                        self._transition(
                            close,
                            track_id,
                            state,
                            timestamp_seconds,
                            timestamp_seconds - state.entered_at,
                            reason,
                        )
                    )
                if self._lifecycle_config.evidence_settle_seconds == 0:
                    settled = self._lifecycle.update(
                        track_id, timestamp_seconds, decision
                    )
                    if settled.transition is not None:
                        output.append(
                            self._transition(
                                settled,
                                track_id,
                                state,
                                timestamp_seconds,
                                timestamp_seconds - state.entered_at,
                                reason,
                            )
                        )
        return output

    def _decision_for_existing(
        self,
        state: _LeftLaneState,
        track_id: int,
        timestamp_seconds: float,
        traffic_context: TrafficFrameContext | None,
        assessment: OvertakingAssessment | None,
        history_durations: dict[int, float],
    ) -> CandidateDecision:
        vehicle_context = (
            traffic_context.vehicles.get(track_id) if traffic_context else None
        )
        speed = (
            (traffic_context.speeds or {}).get(track_id) if traffic_context else None
        )
        return self._decide(
            max(0.0, timestamp_seconds - state.entered_at),
            state.mean_confidence,
            history_durations.get(track_id, 0.0),
            traffic_context,
            vehicle_context,
            assessment,
            speed,
        )

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
                duration,
                mean_confidence,
                history_duration,
                traffic_context.global_context if traffic_context else None,
                vehicle_context,
                assessment,
                speed,
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

    def _legacy_decision(
        self, state: _LeftLaneState, timestamp_seconds: float
    ) -> CandidateDecision:
        return self._decide(
            max(0.0, timestamp_seconds - state.entered_at),
            state.mean_confidence,
            0.0,
            state.latest_context,
            None,
            state.latest_assessment,
            None,
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
        position = frame_context.positions.get(track_id) if frame_context else None
        speed = (frame_context.speeds or {}).get(track_id) if frame_context else None
        evidence_start = (
            state.candidate_evidence_started_at
            if state.candidate_evidence_started_at is not None
            else state.entered_at
        )
        global_context = frame_context.global_context if frame_context else None
        permission = global_context.physical_measurements if global_context else None
        if (
            position is not None
            and not position.calibrated
            and position.physical_measurement_reason_codes
        ):
            permission = PhysicalMeasurementPermission(
                allowed=False,
                confidence=0.0,
                status=position.physical_measurement_status,
                reason_codes=position.physical_measurement_reason_codes,
            )
        return CandidateTransition(
            transition=kind,  # type: ignore[arg-type]
            track_id=track_id,
            lane_id=self._config.left_lane_id,
            start_timestamp_seconds=evidence_start,
            timestamp_seconds=timestamp_seconds,
            duration_seconds=max(duration, timestamp_seconds - evidence_start),
            confidence_score=state.mean_confidence,
            end_reason=end_reason
            or lifecycle.cancellation_reason
            or lifecycle.close_reason,
            review_reason_codes=state.last_decision.reason_codes
            if state.last_decision
            else (),
            policy_version=self._config.policy_version,
            traffic_context=global_context,
            vehicle_traffic_context=frame_context.vehicles.get(track_id)
            if frame_context
            else None,
            overtaking_assessment=state.latest_assessment,
            behavior_classification=state.last_decision.classification.value
            if state.last_decision
            else None,
            evidence_confidence_score=state.last_decision.evidence_confidence
            if state.last_decision
            else None,
            lifecycle_state=lifecycle.state.value,
            candidate_started_at=lifecycle.candidate_started_at,
            suspended_at=lifecycle.suspended_at,
            finalized_at=lifecycle.finalized_at,
            cancelled_at=lifecycle.cancelled_at,
            cancellation_reason=lifecycle.cancellation_reason,
            close_requested_at=lifecycle.close_requested_at,
            close_reason=lifecycle.close_reason,
            decision_history=lifecycle.decision_history,
            position=position,
            speed_estimate=speed,
            calibration_quality=global_context.calibration_quality
            if global_context
            else None,
            camera_motion=global_context.camera_motion if global_context else None,
            camera_pose=global_context.camera_pose if global_context else None,
            physical_measurements=permission,
            geometry_integrity=global_context.geometry_integrity
            if global_context
            else None,
        )

    def _status(
        self,
        observation: LaneObservation,
        state: _LeftLaneState,
        lifecycle: CandidateLifecycleState,
        decision: CandidateDecision,
    ) -> VehicleRuleStatus:
        frame_context = state.latest_context
        vehicle = (
            frame_context.vehicles.get(observation.vehicle.track_id)
            if frame_context
            else None
        )
        position = (
            frame_context.positions.get(observation.vehicle.track_id)
            if frame_context
            else None
        )
        speed = (
            (frame_context.speeds or {}).get(observation.vehicle.track_id)
            if frame_context
            else None
        )
        assessment = state.latest_assessment
        is_candidate = lifecycle in {
            CandidateLifecycleState.CANDIDATE_ACTIVE,
            CandidateLifecycleState.SUSPENDED,
            CandidateLifecycleState.PENDING_CLOSE,
            CandidateLifecycleState.FINALIZED,
        }
        right_gap = (
            (vehicle.right_lane_front_gap or vehicle.right_lane_rear_gap)
            if vehicle
            else None
        )
        permission = (
            frame_context.global_context.physical_measurements
            if frame_context
            else None
        )
        if (
            position is not None
            and not position.calibrated
            and position.physical_measurement_reason_codes
        ):
            permission = PhysicalMeasurementPermission(
                allowed=False,
                confidence=0.0,
                status=position.physical_measurement_status,
                reason_codes=position.physical_measurement_reason_codes,
            )
        return VehicleRuleStatus(
            track_id=observation.vehicle.track_id,
            lane_id=observation.lane_id,
            left_lane_duration_seconds=max(0.0, state.last_seen_at - state.entered_at),
            is_review_candidate=is_candidate,
            behavior_classification=(
                BehaviorClassification.POSSIBLE_LEFT_LANE_OCCUPATION.value
                if is_candidate
                else decision.classification.value
            ),
            suppression_reason=decision.suppression_reason
            if lifecycle == CandidateLifecycleState.SUSPENDED or not is_candidate
            else None,
            overtake_state=assessment.state.value
            if assessment
            else OvertakeState.NONE.value,
            overtaking_status=assessment.status.value
            if assessment
            else OvertakingStatus.NOT_ASSESSED.value,
            right_lane_available_seconds=vehicle.right_lane_available_seconds
            if vehicle
            else 0.0,
            evidence_confidence=decision.evidence_confidence,
            related_track_ids=assessment.related_track_ids if assessment else (),
            candidate_lifecycle_state=lifecycle.value,
            speed_kph=speed.speed_kph if speed else None,
            speed_mode=speed.speed_mode
            if speed
            else "unavailable_physical_measurements",
            coordinate_mode=position.coordinate_mode
            if position
            else "normalized_image",
            right_lane_gap=right_gap,
            physical_measurement_status=permission.status
            if permission
            else "unavailable",
            physical_measurement_reason_codes=permission.reason_codes
            if permission
            else (),
            geometry_integrity_status=(
                frame_context.global_context.geometry_integrity.status.value
                if frame_context
                and frame_context.global_context.geometry_integrity is not None
                else "unverified"
            ),
            geometry_reason_codes=(
                frame_context.global_context.geometry_integrity.reason_codes
                if frame_context
                and frame_context.global_context.geometry_integrity is not None
                else ()
            ),
        )

    @staticmethod
    def _outside_status(
        observation: LaneObservation,
        assessment: OvertakingAssessment | None,
        traffic_context: TrafficFrameContext | None,
    ) -> VehicleRuleStatus:
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
            overtaking_status=assessment.status.value
            if assessment
            else OvertakingStatus.NOT_ASSESSED.value,
            related_track_ids=assessment.related_track_ids if assessment else (),
            candidate_lifecycle_state=CandidateLifecycleState.IDLE.value,
            speed_kph=speed.speed_kph if speed else None,
            speed_mode=speed.speed_mode
            if speed
            else "unavailable_physical_measurements",
            coordinate_mode=position.coordinate_mode
            if position
            else "normalized_image",
            geometry_integrity_status=(
                traffic_context.global_context.geometry_integrity.status.value
                if traffic_context
                and traffic_context.global_context.geometry_integrity is not None
                else "unverified"
            ),
            geometry_reason_codes=(
                traffic_context.global_context.geometry_integrity.reason_codes
                if traffic_context
                and traffic_context.global_context.geometry_integrity is not None
                else ()
            ),
        )

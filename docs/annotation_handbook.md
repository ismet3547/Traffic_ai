# Traffic Behavior Annotation Handbook

Protocol: `pilot-1`  
Handbook: `1.0`  
Dataset target: `pilot-0.1`

## Objective and non-goals

Annotate observable highway traffic behavior for a human-review-candidate benchmark. This is not a legal judgment and must never be used as an automatic penalty decision. Do not record faces, plates, drivers, identity, or other personal information. A vehicle reference is local to one clip and uses only a neutral ID such as `vehicle_03`; a description may say “white sedan,” never a plate number.

Annotators must work independently and must not see another annotator's work or any model prediction before their pass is locked. **Model predictions must not be shown to annotators before ground truth is finalized.**

## General event procedure

1. Watch enough context before and after the episode; frame-step near uncertain boundaries.
2. Assign one annotation-local `vehicle_ref` consistently. Add a representative box and timestamp when identity could be confused.
3. Mark the behavioral interval, choose the best label, then record confidence and observable evidence. Unknown evidence remains `unknown` or omitted—never guessed.
4. Overlap is allowed when separate episodes or vehicles genuinely overlap. Do not split one continuous episode merely to improve apparent precision.
5. Record non-personal ambiguity in notes. Never consult system output.

Boundary comparisons use a default one-second tolerance. This tolerance is a measurement convention, not permission to annotate carelessly.

## Labels

### `unnecessary_left_lane_occupation`

Definition: a vehicle remains in the leftmost through lane for a sustained, observable period without an active passing need while a reasonable opportunity to return right exists.

- Include when the right lane is realistically usable, relevant traffic is free enough to evaluate, and no visible pass or other necessity explains continued left-lane use.
- Exclude active or slow overtakes, consecutive overtakes without a safe return gap, congestion, blocked right lanes, brief transitions, uncertain geometry, and insufficient visibility.
- Example: a clearly tracked sedan continues left beside an empty right lane after any earlier pass has ended.
- Counterexample: a sedan takes time to pass a long truck, then returns right.
- Ambiguity: “sustained” is contextual. Begin when the behavior becomes evaluable as suspicious, not necessarily when the tires first cross into the lane. Use insufficient evidence if the relevant right-lane context is not observable.

### `legitimate_overtaking`

Definition: the target uses the left lane as part of an observable pass or a sequence of passes where no reasonable return gap occurs between them.

- Include entry to pass, gaining on and moving ahead of another vehicle, and a reasonable completion interval.
- Exclude merely travelling faster with no observable vehicle being passed, or staying left long after a safe return becomes clear.
- Example: a car moves left behind a truck, advances past it, and returns right.
- Counterexample: a car cruises left with the right lane empty and no relative-order change.
- Ambiguity: a very slow pass can still be legitimate. Start when passing context becomes observable; end when the maneuver ends or the vehicle clearly transitions into another behavioral episode.

### `congestion_left_lane_use`

Definition: left-lane occupancy occurs within dense, queued, stop-and-go, or constrained traffic where ordinary return-right opportunity cannot be evaluated reliably.

- Include sustained queueing or dense parallel flow that removes meaningful lane choice.
- Exclude moderate but freely moving traffic with a clearly usable right lane.
- Example: all lanes crawl with short gaps.
- Counterexample: several vehicles are visible but safe gaps persist in the right lane.
- Ambiguity: traffic density alone is insufficient; consider whether it actually constrains the target.

### `temporary_left_lane_use`

Definition: a short left-lane episode that is observable but too brief or transitional to characterize as unnecessary occupation.

- Include brief lane changes, positioning transitions, and short observations ending before context becomes evaluable.
- Exclude a clear pass, sustained suspicious occupancy, or a clip whose visibility is inadequate.
- Example: a vehicle enters left for two seconds and immediately returns without a stable episode.
- Counterexample: a long clear left-lane cruise with an open right lane.
- Ambiguity: do not use this label simply because the clip was cut too short; use insufficient evidence when the missing context is decisive.

### `right_lane_unavailable`

Definition: the right lane is visibly not a reasonable return option because of vehicles, closure, obstruction, merge conflict, or inadequate safe gaps.

- Include blocked lanes and clearly unsafe front/rear gaps.
- Exclude a merely less convenient but visibly reasonable opportunity.
- Example: a line of closely spaced trucks prevents a safe return.
- Counterexample: an empty right lane adjacent to the target.
- Ambiguity: perspective can hide rear gaps; use unknown evidence or insufficient evidence rather than assuming availability.

### `insufficient_evidence`

Definition: the relevant behavior cannot be resolved from the available visual evidence.

- Include decisive occlusion, clipped context, uncertain target association, or unresolved adjudication ambiguity.
- Exclude difficult but still observable behavior; use the appropriate behavior label with medium confidence.
- Example: the target disappears behind an obstruction during the only possible passing interval.
- Counterexample: a clearly visible slow pass.
- Ambiguity: confidence describes certainty within a chosen label; this label describes missing evidence that prevents choosing a behavioral label.

### `geometry_invalid`

Definition: lane position or road topology cannot be interpreted reliably for the relevant interval.

- Include severe perspective ambiguity, unclear/non-through lane boundaries, or a curve that makes leftmost-lane identity unreliable.
- Exclude ordinary curvature whose lanes remain visually traceable.
- Example: the road leaves the calibrated/visible area and lane order becomes indeterminate.
- Counterexample: a gentle curve with clear markings.
- Ambiguity: distinguish geometry failure from temporary vehicle occlusion.

### `camera_motion_invalid`

Definition: camera shake, pan, zoom, or viewpoint change prevents reliable temporal traffic interpretation.

- Include motion that breaks target/lane continuity or invalidates static geometry.
- Exclude small vibration that does not affect interpretation.
- Example: a drone changes altitude and yaw enough that the prior road plane no longer applies.
- Counterexample: a stable fixed roadside camera.
- Ambiguity: label the affected interval, not an entire clip when only a local segment is unusable.

## Evidence and confidence

Evidence fields record only what is observable: entered left from right, a vehicle being passed, right-lane availability, congestion, return right, visibility, and coarse vehicle class. Values are true, false, or unknown; omission is also valid. Vehicle class is descriptive coverage metadata, not identity.

- `high`: clear visual evidence, stable target identity, and little material ambiguity.
- `medium`: the label is probable, but some secondary context or a boundary is partially uncertain.
- `low`: substantial uncertainty remains. If uncertainty prevents selecting a behavior at all, use `insufficient_evidence`; do not use low confidence as a substitute.

## Context guidance

Right-lane opportunity requires a usable through lane and a plausible safe front/rear gap over time. A one-frame visual opening is not enough. Never infer that a lane is clear solely because no detector—or no human—noticed a vehicle.

Overtaking evidence is strongest when entry, a vehicle being passed, relative-order change, and return right are visible. None is individually mandatory. Multiple consecutive passes may form one legitimate episode when there is no reasonable intervening return opportunity.

Congestion concerns constraint, not a fixed vehicle count. Queueing and short gaps across lanes usually make an unnecessary-occupation judgment inappropriate.

Occlusion, poor resolution, night glare, lane-boundary proximity, stopped vehicles, merges, roadworks, camera shake, and zoom must be represented honestly. Use diagnostic labels when geometry/camera failure is the dominant fact and `insufficient_evidence` when the target behavior is simply unknowable.

## Independent annotation, agreement, and adjudication

Two annotators receive the same raw clip and handbook but separate output locations and anonymous IDs. They do not see each other's labels. Each pass is validated and locked before comparison. Official agreement uses one-to-one temporal event matching under the canonical protocol below; exact timestamps are not required. Event, label, boundary, confidence, vehicle-reference, and visibility disagreements remain separate diagnostics. Cohen's kappa, when reported, applies only to matched event-label pairs.

### Agreement provenance

An agreement report belongs to one exact source video and two exact locked annotation revisions. It records the source SHA-256, both anonymous annotator IDs, both canonical annotation content hashes, ontology and handbook versions, agreement protocol version, deterministic pair identity, and report content hash. The annotation revision hash includes lock metadata and audited override history but not the JSON file's filesystem path. Any annotation edit and relock requires agreement to be recomputed.

The A/B pair identity is order-independent. Supplying the same report twice, or supplying both A/B and B/A orientations, is a release error rather than extra evidence. Validation/test release also requires the agreement hashes to match the two original annotation hashes embedded in adjudication.

Agreement quality is a macro average over exactly one provenance-validated current report per validation/test video. Reports for unknown clips, stale revisions, different sources or protocols, and incomplete report subsets cannot enter the calculation. If both passes contain zero events, event-detection agreement is `1.0`, matched-event label/boundary/confidence agreement is `0.0`, and temporal IoU is unavailable.

**A high agreement score is meaningless if it was computed from different annotation revisions than the ground truth being released.**

**Agreement quality is calculated only from provenance-validated reports for the exact release set.**

### Canonical agreement protocol

All official validation/test reports use protocol `2`, config version `1`: minimum temporal IoU `0.30`, boundary tolerance `1.0` second, and required vehicle-reference matching. The IoU threshold rejects incidental overlap, the boundary tolerance matches the current timestamping precision, and the vehicle rule ensures agreement refers to the same observed target. These values are dataset-wide and cannot be relaxed per clip.

Reports store the complete config and its canonical SHA-256 fingerprint. Agreement identity includes the protocol and config fingerprint. Release validation ignores the report as a source of grading policy, independently recomputes with the canonical config, and rejects exploratory, old-protocol, noncanonical, or mixed-config reports. Changing any official agreement semantic requires a new protocol/config identity and regeneration of agreement and adjudication artifacts.

Exploratory reports are allowed for development research and must be marked `exploratory`; they are never official quality evidence. **An agreement report cannot choose its own grading rules for an official dataset release. All validation/test agreement metrics in a release are computed under one canonical agreement protocol.**

A positive-event clip is one where at least one annotator recorded an event. Release metadata reports both overall macro event-detection agreement and the positive-event subset, plus zero-event and positive-event clip counts. Zero-event clips remain in the overall calculation. Optional quality thresholds apply to the overall macro metrics; the positive-event metric is informational in this protocol version.

An adjudicator reviews human disagreement, not model output. The artifact embeds both original locked annotations, the agreement report, an explicit decision for every disagreement, rationale, confidence, and final events. It never overwrites originals. Outcomes are agree, resolved to A, resolved to B, new consensus, or remains ambiguous. Remaining ambiguity must export as `insufficient_evidence`; consensus is never forced.

## Versioning, locking, and drift

Every annotation records ontology and handbook versions. Incompatible versions fail validation with instructions to migrate explicitly or re-annotate; they are never silently mixed. Test annotations/adjudications are locked with timestamps and hashes. Normal tooling refuses edits; an exceptional override requires a written reason and records an audit entry. Locking is workflow protection, not cryptographic access control.

Development, validation, and test assignments operate on `source_group_id`. Neighboring, cropped, or edited material from one recording/session/camera must stay in one group to prevent leakage. Exact SHA-256 duplicates are rejected by intake unless explicitly acknowledged. Near-duplicates require manual grouping because perceptual duplicate detection is not implemented.

## Hard cases to seek and report

Coverage should include slow truck overtakes, consecutive overtakes, blocked right lanes, dense flow, queueing, brief changes, curves, partial occlusion, lane-boundary vehicles, shake/zoom, stopped vehicles, merges, varying vehicle classes, daylight/night, and both short and long clear positive cases. Coverage reports actual imbalance; they do not claim balance that is absent.

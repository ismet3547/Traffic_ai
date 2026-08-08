# Traffic AI: explainable left-lane review MVP

Traffic AI analyzes prerecorded highway video and creates evidence packages for a human operator. It detects and tracks vehicles, models lane occupancy and contextual overtaking, and identifies possible unnecessary left-lane occupancy for review.

This is human-review-only decision support, not an enforcement system. It does not identify people, read plates, determine a legal violation, issue fines, or contact police systems. Every review record has `human_review_required: true` and `enforcement_action: none`.

## Architecture

```text
MP4 / future FrameSource
  -> YOLO detector -> ByteTrack
  -> background camera-motion diagnostic -> fixed-camera reference-pose validator
  -> physical permission + central geometry-integrity capabilities
  -> trusted lane assignment/hysteresis
  -> normalized/homography road coordinates -> rolling speed
  -> congestion, neighbors and unit-safe right-lane gaps
  -> bounded history -> contextual overtaking state machine
  -> left-lane policy -> candidate lifecycle -> evidence writer/overlay
```

Detection and tracking run before the motion diagnostic because tracked vehicle boxes are masked from the background-feature estimate. The diagnostic does not alter frames. A future stabilizer can be inserted before road positioning, but must transform frames, lane geometry, and calibration consistently.

```text
traffic_ai/
  app/
    main.py                 CLI composition
    pipeline.py             source-independent analysis loop
    config.py               strict Pydantic configuration
    detection/ tracking/    vendor adapters and protocols
    camera_motion/          diagnostic estimate + static-pose validation
    geometry/               frame compatibility + central integrity gate
    physical_measurements/  centralized fail-closed permission policy
    positioning/            normalized and homography transformers
    lanes/ motion/ speed/   geometry, bounded history, approximate speed
    context/ overtaking/    traffic context and contextual pass assessment
    rules/ candidates/      left-lane policy and explicit lifecycle
    events/ video/          evidence persistence and debug rendering
    benchmark/              offline annotations, matching, metrics and reports
    tools/                  calibration and benchmark developer CLIs
    models/                 framework-neutral typed records
  data/benchmark/           schemas/examples; raw videos remain local
  configs/
    default.yaml
    calibrated_example.yaml # placeholder values only
  tests/                    model-independent unit/integration tests
```

Phase 2 introduced contextual overtaking, congestion, lane transitions, and right-lane opportunity. Phase 3 introduced optional road-plane geometry and approximate speed. Phase 3.1 hardened physical-unit trust and lifecycle closure. Phase 3.2 makes camera/road geometry trust a prerequisite for every geometry-dependent judgment. Phase 4 adds an offline measurement layer and does not alter the production decision policy.

## Geometry Integrity

Lane polygons and a road-plane homography belong to a specific camera pose. Translation, rotation, zoom, a resolution/aspect change, or pitch/roll can make both stale. A system that disables km/h but continues using stale lane polygons is not fail-closed.

`GeometryIntegrityPolicy` produces one per-frame assessment and explicit capabilities for lane assignment, normalized/world relationships, physical speed/gaps, right-lane opportunity, overtaking context, and review-candidate generation. Consumers receive that assessment; none infer that an unavailable pose is safe. When geometry is not trustworthy:

- lane/current-lane values become unavailable and lane-transition hysteresis is cleared;
- relative vehicles, right-lane opportunity, and overtaking geometry become insufficient evidence;
- meter positions/gaps and physical speed are disabled;
- no new candidate can start;
- an active candidate suspends immediately and cancels after `invalidation_grace_seconds` if trust does not recover;
- normalized positions and motion rates may continue only as clearly labeled diagnostics.

The safe default has no lane reference resolution and no runtime pose measurement, so candidates are off. Set the lane reference resolution/pose and enable the experimental pose diagnostic, or use the explicit controlled-deployment escape hatch:

```yaml
lanes:
  reference_width: 1920
  reference_height: 1080
  reference_pose_id: roadside_tripod_2026_08
  scaling_mode: uniform
geometry_integrity:
  external_fixed_camera_guarantee: true
  external_guarantee_id: controlled_mount_change_requires_recalibration
```

An external guarantee means the camera mount is operationally controlled and cannot change without recalibration. It is an operational assumption, not a software measurement; metadata reports `trust_source: external_deployment_guarantee`. See `configs/demo_fixed_camera.yaml`. Exact resolution is accepted; a same-aspect uniform resize is accepted only in `uniform` mode. Aspect changes and unsupported resizing/cropping fail closed.

An external fixed-camera guarantee is used only when runtime camera-pose verification is unavailable. It never overrides measured evidence of camera movement, scale drift, rotation, translation, or projective change.

## Candidate lifecycle

```text
IDLE -> ACCUMULATING -> CANDIDATE_ACTIVE <-> SUSPENDED
                              |
                              v
                         PENDING_CLOSE -> FINALIZED
                              |
                              v
                           CANCELLED
```

- `ACCUMULATING`: occupancy exists but start evidence is not yet sufficient.
- `CANDIDATE_ACTIVE`: start conditions passed and evidence is being recorded.
- `SUSPENDED`: later context is temporarily incompatible; configurable grace applies.
- `PENDING_CLOSE`: a real close trigger occurred and delayed context may still arrive.
- `CANCELLED`: persistent or exculpatory evidence invalidated the episode. It remains auditable but never enters the review queue.
- `FINALIZED`: the episode closed, its settle window elapsed, and the final assessment remained eligible. It is immutable and ready for human review.

Elapsed time after candidate start does not finalize an event. Close triggers are lane exit, permanent track loss, maximum evidence-window duration, or video end. A normal close waits `evidence_settle_seconds`; confirmed overtaking can cancel the candidate during that interval. Video end uses a deterministic forced close because no later frames can arrive.

```yaml
candidate_lifecycle:
  invalidation_grace_seconds: 2.0
  suspension_grace_seconds: 3.0
  evidence_settle_seconds: 2.0
  max_event_duration_seconds: 30.0
  track_loss_close_seconds: 1.5
  restart_cooldown_seconds: 1.5
```

`finalize_after_seconds` is accepted only to load older Phase 3 configurations; it is ignored. Decision history contains bounded state changes rather than one entry per frame. A completed overtake does not permanently exempt a later suspicious episode.

## Calibration trust and independent validation

Default operation is uncalibrated and safe:

```yaml
calibration:
  mode: normalized
physical_measurements:
  require_independent_validation: true
```

Normalized mode retains `image_position` in pixels and dimensionless `normalized_position`. It supports ordering and normalized gaps/motion, but never produces meters or km/h.

A fixed-camera homography uses measured image/road-plane correspondences:

```yaml
calibration:
  mode: homography
  world_units: meters
  image_points: [[410, 720], [880, 720], [690, 420], [590, 420]]
  world_points: [[0, 0], [12, 0], [12, 50], [0, 50]]
  validation_image_points: [[520, 600], [650, 520]]
  validation_world_points: [[2.5, 15], [8.0, 31]]
  reference_width: 1280
  reference_height: 720
  maximum_validation_rmse_world_units: 1.0
  maximum_validation_p95_world_units: 2.0
  minimum_validation_coverage: 0.35
  allow_unverified_physical_measurements: false
  fallback_to_normalized: false
```

These numbers are illustrative only. [configs/calibrated_example.yaml](configs/calibrated_example.yaml) deliberately leaves validation points empty and therefore keeps physical output disabled. Measure control and separate holdout points for the actual camera. Survey stationary road-plane locations, spread them over the analysis area, and do not derive validation points from the fitted matrix.

Four control points that perfectly fit a homography do not prove the physical calibration is accurate. Fit-point residual describes mathematical fit, not real-world trust. The system separately reports:

- matrix validity and numerical conditioning;
- control-point fit reprojection error;
- independent validation reprojection error, when supplied;
- independent world-space RMSE, MAE, maximum, and p95 error in declared units;
- spatial coverage of the usable road region and clustering warnings;
- validation mode, confidence basis, confidence, and reason codes.

Without independent validation, quality is `FIT_POINTS_ONLY`, confidence remains low, and physical output is disabled by default. `allow_unverified_physical_measurements: true` is an explicit experimental override, not verification.

Startup rejects duplicate/collinear or tiny-area control geometry, singular/invertible failures, poor normalized-DLT conditioning, non-finite transforms, and absurd projected bounds. Invalid homography startup falls back only when `fallback_to_normalized: true`, and that state is explicit in metadata/logs.

A homography assumes vehicle contact points lie approximately on the calibrated road plane. Slopes, bridges, lens distortion, and bad bottom-center contact points increase error. The control/validation-point convex hull defines the default support region. Vehicle contact points outside it do not receive world positions, meter gaps, or physical speed; unrestricted extrapolation is unsafe.

## Camera pose and physical-measurement permission

A static homography is compatible only with a stable camera pose. `feature_based` estimates apparent background translation, rotation, and scale using Lucas-Kanade flow and masked vehicle boxes. For OpenCV's partial-affine matrix, scale is the mean Euclidean norm of its two linear columns, which is robust to using one noisy matrix element. It also samples a startup-frame-to-current background homography and reports normalized corner residual after subtracting the best similarity transform as an explainable projective-drift score.

Reference projective matching runs at `camera_motion.reference_analysis_interval_frames`, reuses cached startup features, and is diagnostic and experimental:

- it never warps or stabilizes a frame;
- `stabilization_applied` is always `false`;
- a persistent fixed-camera validator accumulates translation, rotation, and multiplicative scale relative to startup, so repeated small changes cannot evade detection;
- pitch/roll-like projective drift is checked against configurable warning/invalid thresholds.

Static homography measurements are invalidated when camera pose changes unless the pose change is compensated. This release has no compensation. Large drone translation, altitude, pitch, roll, or yaw changes invalidate the fixed matrix; the feature diagnostic does not make drone measurements trustworthy.

`camera_motion.mode: none` means motion is not measured, not that the camera was proven stable. Under secure defaults, unavailable, uncertain, or moved pose disables all geometry judgments and candidates, as well as physical outputs. Configure `feature_based` for runtime validation, or deliberately document an external deployment guarantee.

One `PhysicalMeasurementPolicy` gates every physical value. It considers independent calibration quality, camera pose, transform validity, and track stability. When denied:

- `world_position_m`, `speed_mps`, and `speed_kph` are null;
- meter gap fields are null;
- normalized position, gap, and motion rate may remain separately available;
- status and reason codes explain the lost capability.

No downstream module applies its own confidence threshold to re-enable meters.

## Approximate speed and unit safety

Approximate speed uses robust displacement over a bounded window of permitted road-plane samples. Tracker gaps, teleport-like jumps, impossible configured speeds, transform failure, and pose loss reject or reset estimates. It is not speed-enforcement evidence.

Unit-bearing outputs are explicit:

- `world_position_m` versus `normalized_position`;
- `speed_kph`/`speed_mps` versus `normalized_motion_rate`;
- `right_lane_front_gap_m`/`right_lane_rear_gap_m` versus normalized gap fields.

The internal generic `GapEstimate` always includes an explicit `unit` and `coordinate_mode`. `right_lane_opportunity.mode: auto` uses meter thresholds only while physical permission is active; otherwise it uses normalized thresholds. Calibrated relative ordering/speed can strengthen `TARGET_GAINING_ON_VEHICLE`, `RELATIVE_ORDER_CHANGED`, `TARGET_PASSED_VEHICLE`, and `RETURNED_RIGHT`, but speed is never mandatory for overtaking assessment.

## Calibration preview

The utility saves a GUI-independent preview containing control points, validation points, lane polygons, and the projected road grid:

```powershell
python -m app.tools.visualize_calibration `
  --video input/highway.mp4 `
  --config configs/calibrated_example.yaml `
  --output output/calibration_preview.jpg
```

Its log reports matrix validity, pixel and world validation errors, spatial coverage, support region, frame compatibility, condition metric, confidence basis, reason codes, and physical permission. An unverified fit produces a prominent warning. `--show` is optional.

## Install and run

Python 3.11 or newer, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.main --config configs/default.yaml --input input/highway.mp4
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.main --config configs/default.yaml --input input/highway.mp4
```

The first YOLO model-name use may download weights. Default lane polygons are illustrative and must be adjusted for the camera.

## Output and metadata

```text
output/highway_YYYYMMDD_HHMMSS/
  annotated.mp4
  events.jsonl             # FINALIZED + pending_human_review only
  cancelled_events.jsonl   # cancelled audit records, never review queue
  events/<event_id>/
    metadata.json
    representative.jpg
    event.mp4
```

Schema 3.2 metadata contains bounded lifecycle decisions, close/cancel/final timestamps, geometry capabilities/trust source, lane reference/runtime geometry, calibration pixel/world/coverage diagnostics, support-region status, camera cumulative scale/projective drift and `stabilization_applied`, centralized physical permission, explicit coordinate/unit fields, approximate speed quality, traffic context, and overtaking evidence. Active, suspended, pending-close, and cancelled events cannot enter `events.jsonl`. Terminal delivery is idempotent.

The overlay can show track/lane, coordinate mode, approximate speed or `N/A`, explicitly-unitized right gap, overtaking and lifecycle states, congestion, calibration quality, pose status, and physical-measurement availability. Advanced fields are configurable.

## Benchmarking and validation

The Phase 4 benchmark is separate from detection, tracking, and traffic-rule code. It can run the existing analyzer once, normalize finalized `events.jsonl` records into stable prediction caches, and evaluate those caches repeatedly without loading YOLO:

```text
benchmark manifest
  -> existing app.main inference OR predictions/<video_id>.json
  -> versioned annotation validation
  -> deterministic one-to-one temporal matching
  -> overall, per-video, per-tag, confidence and suppression metrics
  -> suspected FP/FN diagnostics and optional evidence bundles
  -> resolved config snapshot + JSON/Markdown report
```

Start by copying [the benchmark manifest template](data/benchmark/manifests/benchmark_manifest.example.yaml) and [annotation example](data/benchmark/annotations/annotation_example.json). Put local videos under `data/benchmark/videos/` or reference an external path. Raw benchmark videos are ignored by Git by default; do not commit large or unlicensed footage.

Each annotation document is schema `1.0`, belongs to one video and anonymous `annotator_id`, and contains explicit intervals. Every label has one canonical semantic role; an explicitly supplied `role` must match it:

- `unnecessary_left_lane_occupation`: `POSITIVE`, used for candidate precision/recall;
- `legitimate_overtaking`, `congestion_left_lane_use`, `right_lane_unavailable`, `temporary_left_lane_use`, and `geometry_invalid`: `NEGATIVE_CONTROL`, used for separate suppression metrics and never as generic FP-ignore regions;
- `insufficient_evidence`: `IGNORE_REGION`, the only default label allowed to remove a sufficiently covered prediction from FP accounting;
- `camera_motion` and `lane_assignment_uncertain`: `DIAGNOSTIC`, contextual evidence that does not change headline accounting by itself.

Confidence is `high`, `medium`, or `low`. Headline positive and control metrics default to configured high-confidence annotations; exact high/medium/low positive strata are also reported. Confidence and semantic role are independent: a low-confidence positive or control does not automatically become an ignore region. Non-headline positive GT is reported as ignored ground truth, while a prediction overlapping it remains accountable under the conservative headline policy. A manifest may add independent annotation documents through `additional_annotations`; pairwise label agreement, temporal agreement, unmatched counts, mean temporal IoU, and matched-label Cohen's kappa are reported with an explicit caveat.

Ignore-region removal is label-aware and thresholded. `prediction_coverage` is the intersection duration divided by prediction duration and must meet `ignored_regions.minimum_prediction_coverage`; optional temporal-IoU minimums can strengthen that gate. A tiny overlap with an ignored annotation is not sufficient to remove a prediction from false-positive accounting. Every ignored prediction remains in JSON diagnostics with its prediction ID, matched ignore annotation, coverage, IoU, and reason. Negative controls never both hide an FP and count a suppression failure.

`minimum_prediction_confidence` filters only predictions below the configured threshold. Filtered IDs, confidence values, and the threshold remain auditable. Reports reconcile all input records as excluded non-review records, confidence-filtered predictions, or considered predictions; considered predictions then reconcile exactly to TP, FP, or ignored. Positive GT similarly reconciles to matched, FN, or non-headline ignored GT. Invariant violations fail the benchmark.

Manifest tags such as `daylight`, `night`, `rain`, `free_flow`, `dense_traffic`, `fixed_camera`, `camera_motion`, `curved_road`, `heavy_trucks`, `occlusion`, and `low_resolution` produce scenario-specific metrics. Every entry also has a dataset split:

- `development`: threshold and policy tuning;
- `validation`: model/policy selection;
- `test`: reserved final measurement, not tuning.

The benchmark never changes production thresholds. `--split` defaults to `validation`; choosing `test` must be deliberate.

### Run and replay

Run inference and evaluate a local manifest:

```powershell
python -m app.tools.run_benchmark `
  --manifest data/benchmark/manifests/my_manifest.yaml `
  --output benchmark_output/run_001 `
  --split validation
```

Inference writes normalized caches to `benchmark_output/run_001/predictions/<video_id>.json`. Evaluate them again without detector/tracker work:

```powershell
python -m app.tools.run_benchmark `
  --manifest data/benchmark/manifests/my_manifest.yaml `
  --output benchmark_output/replay_001 `
  --predictions-dir benchmark_output/run_001/predictions `
  --skip-inference `
  --split validation
```

The CI-safe synthetic integrity test uses no video and downloads no model weights:

```powershell
python -m app.tools.run_benchmark `
  --manifest data/benchmark/manifests/synthetic_manifest.yaml `
  --output benchmark_output/synthetic `
  --predictions-dir data/benchmark/predictions `
  --skip-inference `
  --no-failure-artifacts `
  --split validation
```

Its committed [example report](data/benchmark/examples/SYNTHETIC_REPORT.md) is explicitly synthetic and is not a real-world performance result.

Validate annotation schemas/references and extract timestamped review frames:

```powershell
python -m app.tools.validate_annotations `
  --manifest data/benchmark/manifests/my_manifest.yaml

python -m app.tools.extract_annotation_frames `
  --video data/benchmark/videos/highway_001.mp4 `
  --output benchmark_output/annotation_frames/highway_001 `
  --every-seconds 2 `
  --contact-sheet
```

Inspect the highest-confidence failures after a run:

```powershell
python -m app.tools.inspect_failures `
  --report benchmark_output/run_001/benchmark_report.json `
  --kind false_positive `
  --limit 20
```

### Matching and metric definitions

Temporal IoU is interval intersection duration divided by interval union duration. Invalid pairs are omitted from the bipartite graph unless they meet the inclusive `minimum_temporal_iou`, optional inclusive `start_tolerance_seconds`, and optional track constraint. A deterministic min-cost maximum-flow assignment first maximizes the number of valid one-to-one matches and then maximizes total temporal IoU. Stable GT/prediction ID ordering resolves equal-quality solutions; the earlier greedy matcher was removed because it could produce avoidable FP+FN pairs.

Negative controls use a separate criterion: both `control_events.minimum_prediction_coverage` and `control_events.minimum_temporal_iou` must pass. A one-frame overlap therefore does not automatically count as a full suppression failure.

- Precision = `TP / (TP + FP)`.
- Recall = `TP / (TP + FN)`.
- F1 = `2 * precision * recall / (precision + recall)`.
- A zero denominator produces `0.0`, never NaN.
- FP/hour and FN/hour divide counts by a validated duration in hours. Available manifest, annotation, prediction-cache, and actual OpenCV video-metadata durations are compared using absolute and relative tolerances; gross disagreement fails before rates are calculated. Actual video metadata is preferred when available. Cache-only single-source rates are labeled `single_source_unverified` with low denominator confidence; configurable acceptance can require multiple consistent sources.
- Start-time and duration errors are prediction values minus matched ground-truth values; absolute summaries are also included.
- Real-time factor is video duration divided by measured end-to-end processing time; values greater than 1 are faster than real time.

Reports include overall, per-video, per-tag, confidence-stratified and policy-specific suppression metrics, full accounting, duration evidence/status, timing errors, performance/hardware metadata when inference was measured, annotation agreement, acceptance criteria if configured, and a suspected failure breakdown. `benchmark_report.json` is machine-readable; `benchmark_report.md` is the human review summary. `resolved_config.yaml`, separate production-config/dataset/evaluation fingerprints, annotation/prediction-cache SHA-256 hashes, Git commit and dirty-worktree state when available, annotation/benchmark schema versions, policy version, detector identifier, and tracker identifier make runs reproducible. Git/version failures are recorded as null and warned, never fabricated.

When enabled, FP/FN bundles are written under `failures/<video_id>/<failure_id>/` with diagnostic metadata and copied or extracted representative media. Categories such as `OVERTAKING_LOGIC_ERROR` and `GEOMETRY_INTEGRITY_ERROR` are heuristic suspects, not asserted root causes. Current production output has rich event-level context but no machine-readable per-frame telemetry stream, so detection misses and tracker ID switches cannot always be diagnosed automatically.

Pass `--baseline previous/benchmark_report.json` to compare precision, recall, F1, FP/hour and processing FPS. Regression deltas are valid only when the evaluated dataset and evaluation protocol are comparable. Dataset fingerprints cover source-video content identity, ordered video IDs/splits, duration identity, and annotation hashes/schema versions; evaluation fingerprints cover explicit protocol semantics plus matching, ignore/control rules, duration policy, confidence threshold, headline confidence, roles, and label set. Production configuration has a separate hash, so policy changes can be compared while the dataset/protocol remains fixed. Fingerprint mismatch suppresses ordinary deltas by default. The developer-only `--allow-incomparable-baseline` flag displays them under a prominent `NON-COMPARABLE BASELINE OVERRIDE` warning. Direction-aware tolerances avoid treating floating-point noise as regression. Optional acceptance criteria report PASS/FAIL only when explicitly configured; the repository invents no government or production-readiness threshold.

### Benchmark Identity and Comparability

`video_id` is a logical key, not proof of video identity. When inference runs against
an available source video, the benchmark streams the file through SHA-256 in bounded
chunks and writes both `source_video_sha256` and `source_video_size_bytes` into the
prediction cache. A later cache-only run can use that immutable evidence as
`cached_full_sha256` when the raw file is unavailable. If the raw file is available,
its current bytes are hashed and checked against the cache before evaluation;
disagreement fails with `PREDICTION_CACHE_SOURCE_MISMATCH`.

Legacy schema-1.0 caches without source identity still load, but their dataset status
is `unverified`. They cannot participate in a strict comparable baseline. Re-run
inference with the current benchmark runner to create a schema-1.1 cache containing
the source hash and byte size. Filename, local directory, and raw-versus-cached hash
acquisition are diagnostic provenance rather than content identity: renaming a
byte-identical source does not change its dataset fingerprint. Full SHA-256 is
intentionally used for official reproducibility; filename, size, and duration alone
are never promoted to verified identity.

The dataset fingerprint canonically combines, per sorted video ID, the split,
source-video SHA-256 and byte size, cryptographic verification status, duration
identity, and each annotation's SHA-256 and schema version. Changing either video or
annotation bytes therefore invalidates dataset comparability. If even one video lacks
verified source identity, `dataset_identity_status` is `unverified` and strict
comparison fails with `DATASET_IDENTITY_UNVERIFIED`.

Evaluation meaning is identified separately by protocol `4.1.1` and explicit matcher,
metric, annotation-ontology, ignore-policy, and control-policy semantic versions. The
current matcher version means maximum valid match cardinality first, maximum total
temporal IoU second, with deterministic ID-based tie behavior. Metric identity covers
zero-denominator behavior, validated rate denominators, ignored-prediction accounting,
and confidence-filtered accounting. The evaluation fingerprint combines those semantic
versions with all evaluation settings and role/label policies. A semantic-version
change produces `EVALUATION_PROTOCOL_MISMATCH`; a settings-only change under the same
protocol produces `EVALUATION_CONFIG_MISMATCH`.

Two benchmark runs are not comparable merely because they use the same video IDs and YAML thresholds. Changing the event-matching algorithm changes the evaluation protocol even if every configuration value remains identical.

Production configuration remains separately hashed so a left-lane-policy change can
be measured on the same verified dataset and evaluation protocol. The Git commit is
reported for traceability, but it is not proof of either dataset or evaluation
compatibility. Historical reports lacking source identity or protocol metadata are
handled as legacy and fail closed with `LEGACY_BASELINE_IDENTITY_INCOMPLETE`. The
developer override may display diagnostic deltas, but `comparison_valid` remains false
and `comparison_mode` is `forced_non_comparable`.

No real-world accuracy claim should be made until a sufficiently diverse, independently annotated test set has been evaluated.

## Real Dataset and Annotation

Phase 4.2 adds a prediction-free, versioned human-ground-truth workflow under `app/dataset`. It is deliberately separate from model evaluation: raw human annotation has no `system_prediction`, `model_score`, or `candidate_reason` fields. **Model predictions must not be shown to annotators before ground truth is finalized.** Agreement between annotators is a property to measure, not an assumption.

The planning target is 30–50 independent traffic clips totaling roughly 1–2 hours. This is neither a hard limit nor a claim that the dataset exists; see `data/benchmark/REAL_DATASET_STATUS.md` for the honest current count. Raw video remains git-ignored and should be collected only with documented permission. Intake records source provenance and rights, SHA-256 identity, byte size, duration, resolution, FPS, and filename. The tools process traffic behavior only: do not annotate faces, plates, drivers, or identity.

The blinded workflow is:

1. Register a permitted clip and assign its original recording/session/camera to a `source_group_id`.
2. Tag scenarios without consulting model predictions and assign source groups—not neighboring clips—to development, validation, or test.
3. Have annotators A and B label validation/test independently into separate files.
4. Validate and lock both passes, then measure temporal event, label, boundary, and confidence agreement.
5. Adjudicate every disagreement without overwriting either original. Unresolved cases remain `insufficient_evidence`.
6. Lock final test adjudication, create a release manifest, and export approved ground truth into the Phase 4 benchmark schema.

```powershell
python -m app.tools.register_benchmark_video --video D:\licensed\highway_001.mp4 --video-id highway_001 --source-group-id session_001 --source-type user_provided --source-reference "owner delivery 2026-08-08" --acquisition-date 2026-08-08 --permission-status verified --benchmark-use-allowed --scenario-tag daylight --scenario-tag fixed_camera
python -m app.tools.annotate_video --video D:\licensed\highway_001.mp4 --output annotation_work/highway_001/annotator_a.json --annotator-id annotator_a --video-id highway_001 --event-json event_a.json --lock
python -m app.tools.validate_annotations --dataset-annotation annotation_work/highway_001/annotator_a.json
python -m app.tools.compare_annotations annotation_work/highway_001/annotator_a.json annotation_work/highway_001/annotator_b.json --official --output agreement_work/highway_001.json
python -m app.tools.adjudicate_annotations annotation_work/highway_001/annotator_a.json annotation_work/highway_001/annotator_b.json --official --adjudicator-id adjudicator_01 --decisions decisions.json --output annotation_work/highway_001/adjudicated.json --lock
python -m app.tools.assign_dataset_splits --candidates split_candidates.json --output split_assignments.json --seed 42
python -m app.tools.generate_dataset_coverage --annotations-dir annotation_work --agreements-dir agreement_work --output-dir release_work
python -m app.tools.build_dataset_release --splits split_assignments.json --annotations-dir annotation_work/independent --adjudications-dir annotation_work/adjudicated --agreements-dir agreement_work --output release_work/dataset_release.json
python -m app.tools.export_adjudicated_benchmark --adjudication annotation_work/highway_001/adjudicated.json --annotations-dir annotation_work/highway_001 --splits split_assignments.json --release release_work/dataset_release.json --output data/benchmark/annotations/highway_001.json
```

The event helper accepts repeatable inline JSON or JSON-file inputs and provides a frame/timestamp-assisted CLI workflow; use the existing frame extractor and a media player for frame stepping and seeking. Label definitions, boundary rules, evidence semantics, confidence levels, hard cases, and adjudication are normative in `docs/annotation_handbook.md`.

Exact duplicate bytes are detected by SHA-256. Cropped, transcoded, or neighboring clips require manual source grouping. Splitting prioritizes isolation over perfect balance. Coverage reports expose actual imbalance. Release manifests use `pilot-0.1` and carry source/annotation hashes, protocol versions, permissions, splits, and adjudication state. Quality gates require verified identity/provenance, double annotation and adjudication for validation/test, and locked test truth. Optional agreement thresholds apply only when configured. Protocol drift fails validation and requires explicit migration or re-annotation.

Run the complete explicitly synthetic lifecycle without real footage:

```powershell
python -m app.tools.run_synthetic_annotation_lifecycle --output benchmark_output/phase42_lifecycle
```

These generated artifacts are test-only and make no real-human-annotation claim.

### Release Integrity

`video_id` is a logical name, not source identity. Phase 4.2.1 binds every release entry to the registry SHA-256 and byte size, validates each annotation against those bytes, and makes adjudication carry the same identity. It also compares the adjudication's stored original-annotation hashes with the exact current annotation revisions. An override and relock therefore makes old adjudication stale until the clip is adjudicated again.

**Perfect annotator agreement does not make ground truth valid if both annotators labeled the wrong source video.** A locked artifact can likewise be internally stable but still refer to the wrong source; lock state never substitutes for provenance validation.

The release boundary independently validates every split entry. Registry `source_group_id` is authoritative, each registered video must have exactly one assignment, unknown/duplicate entries fail, and a group cannot span splits. Byte-identical source SHA-256 values also cannot cross splits even when they were accidentally given different video and group IDs. **A split assignment is not trusted merely because it was produced by the official split tool; release-time leakage checks are mandatory.** Same-byte duplicates in one split are permitted but should normally share a source group and remain subject to manual near-duplicate review.

Validation/test clips require two distinct, locked, current, source-matched, protocol-compatible annotation revisions plus approved adjudication; test adjudication must also be locked. Development may carry a clearly marked single annotation, but source and protocol integrity remain mandatory. The release manifest stores exact source, annotation, adjudication, and exported ground-truth content hashes, along with a typed integrity report and affected IDs.

Official release construction is fail-closed: all artifacts are loaded and checked in memory before output, and JSON persistence uses a same-directory temporary file plus atomic replacement. The benchmark export requires the validated release, current source annotations, registry, and split document, then verifies that regenerated ground truth matches the release hash.

Run the repeatable integrity scenarios:

```powershell
python -m app.tools.run_release_integrity_scenarios --output benchmark_output/phase421_integrity
```

They cover a valid release, perfect agreement on the wrong source SHA, manually leaked source groups, duplicate bytes across splits, and stale adjudication. Only the valid scenario writes a release.

### Agreement Provenance

Agreement protocol `2` binds every report to the registry video ID, source-video SHA-256 and available byte size, both anonymous annotator IDs, and the canonical content SHA-256 of both exact annotation revisions. The deterministic `agreement_id` normalizes A/B order and includes agreement-config identity, so copied or reversed reports cannot reweight a release and the same pair under different rules cannot be confused. The report also has its own content hash and records the ontology and handbook versions used by each input.

Phase 4.2.3 moves adjudication and dataset-release schemas to `1.3`, agreement-report schema to `1.1`, and the agreement protocol from `1` to `2`. Regenerate older agreement, adjudication, and release artifacts from the locked source annotations; legacy or protocol-1 reports are intentionally not migrated or trusted.

The canonical annotation revision hash uses the same full-document semantics as adjudication stale-revision checks. It is independent of the annotation JSON file's filesystem path, but intentionally includes document metadata such as lock state, lock timestamp, annotation hash, and audited override history. Consequently, changing and relocking an annotation invalidates its old agreement report even when the event metrics happen to stay unchanged.

Official validation/test release construction requires exactly one current, provenance-validated report for every eligible video. It reproduces the metrics from the bound annotations, requires the report and adjudication to reference the same revisions, rejects unknown, stale, unsupported, or duplicate reports, and only then calculates quality. Unrelated reports cannot enter the average, and missing reports cannot be hidden by averaging a convenient subset. Development clips do not require agreement, although any supplied development report must still validate.

Dataset agreement uses a macro average per video. A clip with many events therefore cannot silently dominate the release score. When both annotators record zero events, event-detection agreement is explicitly `1.0`; matched-event label, boundary, and confidence agreement are `0.0`, and mean temporal IoU is unavailable. Zero-event clips remain in the macro denominator with these documented semantics.

**A high agreement score is meaningless if it was computed from different annotation revisions than the ground truth being released.**

**Agreement quality is calculated only from provenance-validated reports for the exact release set.**

Run the Phase 4.2.2 provenance scenarios:

```powershell
python -m app.tools.run_agreement_integrity_scenarios --output benchmark_output/phase422_agreement_integrity
```

They exercise a valid current report, unrelated perfect-report injection, duplicate weighting, a stale report after annotation editing, and an agreement/adjudication revision mismatch. Failed cases do not write release manifests.

### Canonical Agreement Protocol

Official validation/test agreement uses one immutable config: minimum temporal IoU `0.30`, boundary tolerance `1.0` second, and required vehicle-reference matching. These are the established handbook/default rules: `0.30` rejects incidental overlap, `1.0s` reflects the current annotation boundary precision, and vehicle-reference matching prevents overlapping events for different target vehicles from being treated as agreement. They are fixed across clips and are not loosened to improve a score.

Protocol version `2`, config version `1`, aggregation version `1`, and the canonical config SHA-256 jointly identify the grading semantics. Each official report stores its mode, versions, full config, and fingerprint. The release independently imposes the canonical config, recomputes agreement from current annotations under that config, requires the report and adjudication to contain the same report identity, and verifies that every required report has one identical protocol/config identity. A protocol or config change creates a different `agreement_id` and invalidates prior reports.

The comparison and adjudication CLIs default to official canonical mode. `--official` makes that intent explicit and rejects threshold overrides. Supplying a custom threshold or `--ignore-vehicle-reference` automatically produces an `exploratory` report; `--exploratory` can also mark research output explicitly. Exploratory reports may be inspected for development work, but cannot enter validation/test release metrics.

```powershell
# Official: overrides are prohibited
python -m app.tools.compare_annotations annotation_a.json annotation_b.json --official --output agreement.json

# Research only: output is marked exploratory
python -m app.tools.compare_annotations annotation_a.json annotation_b.json --exploratory --minimum-temporal-iou 0.20 --output exploratory_agreement.json
```

Dataset output reports overall macro agreement plus the number of clips where both annotators recorded zero events, the number of positive-event clips, and macro event-detection agreement for the positive subset. A positive-event clip is deterministically defined as one where at least one annotator recorded an event. Zero-event clips remain in overall metrics under the documented semantics; they are not silently removed. Existing optional thresholds continue to apply to overall macro-per-video metrics, while positive-event agreement is currently informational.

**An agreement report cannot choose its own grading rules for an official dataset release.**

**All validation/test agreement metrics in a release are computed under one canonical agreement protocol.**

Run the canonical-protocol adversarial scenarios:

```powershell
python -m app.tools.run_agreement_protocol_scenarios --output benchmark_output/phase423_agreement_protocol
```

The scenarios cover canonical reports, a permissive exploratory inflation attempt, mixed configs, an old protocol report, and regeneration under the current protocol.

### Phase 4.3 mini pilot operation

No legally registered real footage is currently present, so Phase 4.3 is operationally ready but honestly blocked. Synthetic videos, annotations, predictions, and metrics are never counted as real pilot progress. The initial status is:

```text
REAL PILOT STATUS: BLOCKED — NO REAL SOURCE VIDEO REGISTERED
registered=0  double_annotated=0  adjudicated=0
benchmark_exported=0  inference_complete=0  benchmark_complete=0
```

The frozen pilot identity and artifact layout live in `data/benchmark/pilot/mini_pilot_manifest.json`. The status tool validates actual registry, annotation, canonical agreement, locked adjudication, release/GT, source-bound prediction, and benchmark artifacts before incrementing a stage:

```powershell
python -m app.tools.pilot_status --manifest data/benchmark/pilot/mini_pilot_manifest.json --output-json data/benchmark/pilot/pilot_status.json --output-markdown data/benchmark/pilot/pilot_status.md
```

Follow `docs/mini_pilot_operator_checklist.md` for the complete register → blind A/B annotation → canonical agreement → adjudication → release → GT export → untouched inference/benchmark → failure-review sequence. After the first 3–5 double-annotated clips, annotation pauses for an agreement review. Model output must remain hidden until adjudicated GT is finalized and locked.

The first real completed run is copied to a never-overwritten `pilot_baseline_0` only when the report is non-synthetic, source identities are verified, the worktree was clean, exact commit/config/detector/tracker/protocol identities are present, and every selected clip has locked released GT:

```powershell
python -m app.tools.freeze_pilot_baseline --manifest data/benchmark/pilot/mini_pilot_manifest.json
```

Only a frozen baseline authorizes post-hoc FP/FN review. It does not authorize tuning on validation clips or a production-accuracy claim. **Mini-pilot sample size is too small for production accuracy claims.**

### Phase 4.3.1 evidence-backed pilot review

Pilot completion is derived from review artifacts; it cannot be declared by editing a status flag. Legacy manifest values named `failure_review_completed`, `first_agreement_review_video_ids`, or `scale_up_recommendation` are accepted only for migration, ignored by the decision path, omitted on rewrite, and reported as `LEGACY_PILOT_COMPLETION_FIELDS_IGNORED`.

Three canonical, atomically written documents form the review chain:

1. `FailureReviewDocument` binds every review to the canonical SHA-256 of the frozen benchmark report plus its dataset fingerprint, evaluation fingerprint, baseline ID, and system commit. Failure IDs are semantic canonical hashes, not list positions. **Every FP and FN in the frozen baseline must be accounted for exactly once before failure review is complete.** Missing, duplicate, unknown, identity-mismatched, tampered, or stale entries fail coverage. A zero-failure baseline auto-completes only this stage with “No FP/FN review required for this baseline”; it does not imply production quality.
2. `FirstAgreementReviewDocument` binds the review to the exact first N (configured 3–5, default 3) current canonical `AgreementReport` identities selected deterministically by `video_id`. Annotation, agreement protocol, or report revisions make it stale. A handbook-revision action must reference stable IDs from `handbook_issues.md`.
3. `ScaleUpDecisionDocument` records the human `GO`, `CONDITIONAL_GO`, or `NO_GO` judgment and rationale while binding it to the current baseline, release, benchmark report, failure-review hash, and agreement-review hash. `CONDITIONAL_GO` requires conditions; `NO_GO` requires blockers. Any evidence change invalidates the decision.

```powershell
python -m app.tools.review_pilot_failures --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json
python -m app.tools.review_pilot_failures --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json --reviews data/benchmark/pilot/work/failure_reviews.json
python -m app.tools.review_initial_agreement --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json --summary data/benchmark/pilot/work/first_agreement_summary.json
python -m app.tools.record_scale_up_decision --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json --decision GO --rationale "REPLACE_WITH_EVIDENCE_BASED_RATIONALE"
python -m app.tools.run_pilot_review_scenarios
```

The status command reports the explicit pilot state and nested baseline, agreement-review, failure-coverage, and decision validity. The terminal states are `COMPLETE_GO`, `COMPLETE_CONDITIONAL_GO`, and `COMPLETE_NO_GO`; none is reachable from a manifest checkbox. Post-hoc tools expose model failures only after the existing locked-GT and immutable-baseline gates.

### Phase 4.3.2 fail-closed terminal states

Current eligibility and historical evidence are separate. A valid historical GO decision does not override a current integrity blocker. A pilot cannot be COMPLETE while any completion-blocking issue is active.

Pilot issues have one centrally assigned severity: `BLOCKER`, `WARNING`, or `INFO`. Unknown future issue codes default to `BLOCKER`. Current source confirmation, permission, benchmark authorization, local source/config presence, frozen protocol identity, dataset/release/provenance, baseline currency, annotation/agreement/adjudication integrity, exact review coverage, review staleness, and decision staleness are completion-blocking. The mini-pilot accuracy caveat is a warning, and ignored legacy manifest fields are informational.

Terminal derivation consumes both the evidence statuses and the classified current blocker set. A current blocker returns a non-terminal state such as `BASELINE_FROZEN` while retaining a still-readable historical decision in status. `pilot_executed` is true only for a terminal state with zero active blockers; the status model independently enforces the same invariant. Changing current production-config content from the identity frozen in baseline 0 produces `PILOT_BASELINE_STALE`.

## Tests and checks

Tests use synthetic geometry, fake detections/tracks, and fake video writers; they do not download YOLO weights:

```powershell
python -m pytest -q
python -m ruff check app tests
python -m ruff format --check app tests
python -m mypy
python -m pyright
python -m compileall -q app tests
```

The committed mypy and pyright configurations cover the benchmark package and its
CLI tools introduced in Phase 4. This keeps benchmark-integrity checks reproducible
without claiming that unrelated legacy production modules are already fully typed.

Coverage includes independent world/pixel validation, spatial coverage and support-region rejection, resolution/aspect compatibility, pose noise and cumulative translation/rotation/scale/projective movement, fail-closed lanes/speed/gaps/candidates, lifecycle suspension/cancellation, tracker jumps/dropout, queue integrity/idempotence, synthetic pipeline flows, annotation roles, adversarial ignore overlaps, optimal matching, duration evidence mismatch, accounting invariants, comparable/non-comparable baselines, failure diagnostics, reports, annotator agreement, cache-only benchmark replay, truthful mini-pilot stage accounting, locked-GT review gating, and immutable pilot baselines.

## Known limitations

- This is an MVP, not production-ready public-sector evidence software. Site validation, monitoring, security/privacy review, audit operations, failure recovery, and measured accuracy studies are still required.
- The road is approximated as a plane; lens distortion is not modeled. Undistort footage first when needed.
- The pose diagnostic is not stabilization and cannot support moving-drone traffic judgments. Large drone altitude/orientation changes invalidate a static homography; production moving-drone use requires tested stabilization plus synchronized transformation/revalidation of lanes and road calibration.
- Detection absence is not proof that a lane or gap is clear. Occlusion and tracker ID switches still affect context.
- Benchmark quality is limited by scenario coverage, annotation consistency, interval definitions, and whether the annotation set is exhaustive. Event-level output cannot by itself prove the root cause of every FP/FN.
- Thresholds and lane geometry require site-specific validation. Weather, signs, roadworks, emergency behavior, and jurisdiction-specific exceptions are not interpreted.
- Outputs are review candidates only. Human judgment remains mandatory; automatic enforcement is intentionally absent.

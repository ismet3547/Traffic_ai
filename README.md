# Traffic AI: explainable left-lane review MVP

Traffic AI analyzes prerecorded highway video and creates evidence packages for a human operator. It detects and tracks vehicles, models lane occupancy and contextual overtaking, and identifies possible unnecessary left-lane occupancy for review.

This is human-review-only decision support, not an enforcement system. It does not identify people, read plates, determine a legal violation, issue fines, or contact police systems. Every review record has `human_review_required: true` and `enforcement_action: none`.

## Architecture

```text
MP4 / future FrameSource
  -> YOLO detector -> ByteTrack -> lane assignment/hysteresis
  -> background camera-motion diagnostic -> fixed-camera pose validator
  -> centralized physical-measurement permission
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
    physical_measurements/  centralized fail-closed permission policy
    positioning/            normalized and homography transformers
    lanes/ motion/ speed/   geometry, bounded history, approximate speed
    context/ overtaking/    traffic context and contextual pass assessment
    rules/ candidates/      left-lane policy and explicit lifecycle
    events/ video/          evidence persistence and debug rendering
    tools/                  non-GUI calibration preview
    models/                 framework-neutral typed records
  configs/
    default.yaml
    calibrated_example.yaml # placeholder values only
  tests/                    model-independent unit/integration tests
```

Phase 2 introduced contextual overtaking, congestion, lane transitions, and right-lane opportunity. Phase 3 introduced optional road-plane geometry and approximate speed. Phase 3.1 hardens trust, units, camera-pose compatibility, lifecycle closure, and queue integrity.

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
  allow_unverified_physical_measurements: false
  fallback_to_normalized: false
```

These numbers are illustrative only. [configs/calibrated_example.yaml](configs/calibrated_example.yaml) deliberately leaves validation points empty and therefore keeps physical output disabled. Measure control and separate holdout points for the actual camera. Survey stationary road-plane locations, spread them over the analysis area, and do not derive validation points from the fitted matrix.

Four control points that perfectly fit a homography do not prove the physical calibration is accurate. Fit-point residual describes mathematical fit, not real-world trust. The system separately reports:

- matrix validity and numerical conditioning;
- control-point fit reprojection error;
- independent validation reprojection error, when supplied;
- validation mode, confidence basis, confidence, and reason codes.

Without independent validation, quality is `FIT_POINTS_ONLY`, confidence remains low, and physical output is disabled by default. `allow_unverified_physical_measurements: true` is an explicit experimental override, not verification.

Startup rejects duplicate/collinear or tiny-area control geometry, singular/invertible failures, poor normalized-DLT conditioning, non-finite transforms, and absurd projected bounds. Invalid homography startup falls back only when `fallback_to_normalized: true`, and that state is explicit in metadata/logs.

A homography assumes vehicle contact points lie approximately on the calibrated road plane. Slopes, bridges, lens distortion, bad bottom-center contact points, and extrapolation outside the surveyed area increase error.

## Camera pose and physical-measurement permission

A static homography is compatible only with a stable camera pose. `feature_based` estimates apparent background translation/rotation using Lucas-Kanade flow and masked vehicle boxes. It is diagnostic and experimental:

- it never warps or stabilizes a frame;
- `stabilization_applied` is always `false`;
- a rolling/persistent fixed-camera pose validator separates small noise from meaningful movement.

Static homography measurements are invalidated when camera pose changes unless the pose change is compensated. This release has no compensation. Large drone translation, altitude, pitch, roll, or yaw changes invalidate the fixed matrix; the feature diagnostic does not make drone measurements trustworthy.

`camera_motion.mode: none` means motion is not measured, not that the camera was proven stable. Under secure defaults, unavailable, uncertain, or moved pose disables physical outputs. Configure `feature_based` for runtime pose validation when using a homography.

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

Its log reports matrix validity, fit and independent validation error, condition metric, confidence basis, reason codes, and physical permission. An unverified fit produces a prominent warning. `--show` is optional.

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

Schema 3.1 metadata contains bounded lifecycle decisions, close/cancel/final timestamps, calibration diagnostics, camera pose and `stabilization_applied`, centralized physical permission, explicit coordinate/unit fields, approximate speed quality, traffic context, and overtaking evidence. Active, suspended, pending-close, and cancelled events cannot enter `events.jsonl`. Terminal delivery is idempotent.

The overlay can show track/lane, coordinate mode, approximate speed or `N/A`, explicitly-unitized right gap, overtaking and lifecycle states, congestion, calibration quality, pose status, and physical-measurement availability. Advanced fields are configurable.

## Tests and checks

Tests use synthetic geometry, fake detections/tracks, and fake video writers; they do not download YOLO weights:

```powershell
python -m pytest -q
python -m ruff check app tests
python -m ruff format --check app tests
python -m compileall -q app tests
```

Coverage includes independent/unverified/bad calibration, numerical rejection, normalized fallback, pose noise and persistent movement, fail-closed speed/gaps, tracker jumps/dropout, lifecycle close/settle/cancel/finalize/restart, delayed overtaking, track/video end, queue integrity/idempotence, and synthetic pipeline flows.

## Known limitations

- This is an MVP, not production-ready public-sector evidence software. Site validation, monitoring, security/privacy review, audit operations, failure recovery, and measured accuracy studies are still required.
- The road is approximated as a plane; lens distortion is not modeled. Undistort footage first when needed.
- The pose diagnostic is not stabilization and cannot support moving-drone physical measurements.
- Detection absence is not proof that a lane or gap is clear. Occlusion and tracker ID switches still affect context.
- Thresholds and lane geometry require site-specific validation. Weather, signs, roadworks, emergency behavior, and jurisdiction-specific exceptions are not interpreted.
- Outputs are review candidates only. Human judgment remains mandatory; automatic enforcement is intentionally absent.

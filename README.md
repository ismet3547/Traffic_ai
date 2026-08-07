# Traffic AI left-lane review MVP

This Python application analyzes a prerecorded highway video and creates human-review candidates when a tracked vehicle stays in the configured leftmost lane longer than a configured threshold. It does **not** determine guilt, issue tickets, identify people, read plates, or contact enforcement systems.

## Architecture

```text
traffic_ai/
  app/
    main.py                       CLI, dependency composition
    config.py                     validated YAML configuration
    pipeline.py                   source-independent processing loop
    detection/
      base.py                     detector protocol
      ultralytics_detector.py     YOLO frame detector
    tracking/
      base.py                     tracker protocol
      bytetrack_tracker.py        persistent ByteTrack IDs
    lanes/
      assignment.py               polygon scaling and lane assignment
    rules/
      left_lane.py                configurable occupancy state machine
    events/
      writer.py                   review image, clip, and JSON artifacts
    video/
      protocols.py                replaceable frame-source/video-sink contracts
      opencv_io.py                MP4 input and output
      annotation.py               debug overlay
    models/
      domain.py                   records shared between stages
  configs/default.yaml            example camera/lane configuration
  input/                           local source videos (ignored by Git)
  output/                          generated run folders (ignored by Git)
  tests/                           model-independent unit tests
  requirements.txt
```

The data flow is:

```text
OpenCV FrameSource
  -> Ultralytics vehicle Detector
  -> ByteTrack VehicleTracker
  -> polygon LaneAssigner
  -> LeftLaneRuleEngine
  -> EventArtifactWriter + annotated VideoSink
```

Detection, tracking, geometry, and traffic policy communicate through small typed records. The rule engine never imports YOLO or ByteTrack. A later RTSP, drone, or message-queue input can implement `FrameSource` without changing lane/rule/event code.

## Candidate semantics

For every persistent track, the rule engine measures video time from its first observation in the configured `left_lane_id`. At the threshold it starts a `left_lane_review_candidate` only when the mean detector confidence also meets the configured minimum.

The MVP uses `overtaking_clearance_mode: none`: it makes no claim about whether the vehicle was overtaking. Every saved record says:

- `review_status: pending_human_review`
- `human_review_required: true`
- `enforcement_action: none`
- `overtaking_assessment: not_implemented`

A candidate ends when the vehicle leaves the polygon, the track is absent past the grace period, or the video ends. `event_start_timestamp_seconds` is the original left-lane entry time; `candidate_created_timestamp_seconds` is when the threshold was reached. All timestamps are relative to the start of the source video.

## Dependencies

- Python 3.11 or newer (3.11/3.12 are the safest choices for broad ML-wheel availability)
- OpenCV for video decoding, encoding, and overlays
- Ultralytics YOLO for vehicle detection
- Supervision ByteTrack for persistent IDs
- NumPy for array interchange
- Pydantic v2 and PyYAML for strict configuration
- pytest for tests

The Supervision version is deliberately constrained below 0.28 because this MVP uses its documented `ByteTrack.update_with_detections` adapter. This keeps the dependency boundary explicit for a later migration to the separate `trackers` package.

## Setup

From the `traffic_ai` directory, create and activate a virtual environment.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run with a model name such as `yolo11n.pt` lets Ultralytics download those weights. A local weights path can be supplied with `--model`.

## Configure lanes

Edit `configs/default.yaml` before evaluating results. The included polygons are illustrative and will not match an arbitrary camera.

With `coordinate_space: normalized`, every point is `[x / frame_width, y / frame_height]` in the range 0–1. Each polygon should cover one visible lane. Vehicle membership uses the bottom-center point of its bounding box, which approximates the tire/road contact location. Exactly one polygon must have `leftmost: true`, and `rules.left_lane.left_lane_id` must reference it.

Important rule settings:

```yaml
rules:
  left_lane:
    left_lane_id: left
    occupancy_threshold_seconds: 8.0
    track_lost_grace_seconds: 1.0
    minimum_mean_confidence: 0.25
    overtaking_clearance_mode: none
```

Use the annotated output video to iteratively align the polygons with the road. The meaning of “leftmost” depends on traffic direction and jurisdiction; configuration must be reviewed for each camera.

## Run

Place a video at `input/highway.mp4`, then run:

```powershell
python -m app.main --config configs/default.yaml --input input/highway.mp4
```

Or specify all common overrides:

```powershell
python -m app.main `
  --config configs/default.yaml `
  --input C:\path\to\highway.mp4 `
  --output-dir output `
  --model yolo11n.pt `
  --device cpu
```

For an NVIDIA CUDA device supported by the installed PyTorch build, use `--device 0`.

Run the tests with:

```powershell
python -m pytest -q
```

## Output

Each execution creates a timestamped directory:

```text
output/highway_YYYYMMDD_HHMMSS/
  annotated.mp4
  events.jsonl
  events/
    left_lane_track_17_0000012500/
      metadata.json
      representative.jpg
      event.mp4
```

`annotated.mp4` shows lane polygons, boxes, IDs, class, current lane, left-lane duration, and `REVIEW` candidate state. A candidate clip contains configured pre-event context and is capped by `clip_max_duration_seconds`. `events.jsonl` is an index of finalized records; the authoritative per-event record is `metadata.json` beside its media.

Example metadata:

```json
{
  "schema_version": "1.0",
  "event_id": "left_lane_track_17_0000012500",
  "event_type": "left_lane_review_candidate",
  "review_status": "pending_human_review",
  "human_review_required": true,
  "enforcement_action": "none",
  "track_id": 17,
  "event_start_timestamp_seconds": 12.5,
  "candidate_created_timestamp_seconds": 20.5,
  "event_end_timestamp_seconds": 25.1,
  "duration_seconds": 12.6,
  "lane_id": "left",
  "confidence_score": 0.87,
  "source_video_name": "highway.mp4",
  "representative_frame": "representative.jpg",
  "event_video_clip": "event.mp4",
  "end_reason": "left_lane_exit",
  "overtaking_assessment": "not_implemented"
}
```

## MVP limitations and next interfaces

- Lane geometry is static. Camera movement or drone footage will require stabilization or per-frame homography before lane assignment.
- Occlusion, shadows, small distant vehicles, and tracker ID switches can affect durations.
- No speed, traffic-density, vehicle-relative positioning, or definite overtaking logic is implemented.
- No license-plate recognition, facial recognition, identity lookup, police API, or automatic enforcement exists.
- Model and lane calibration must be validated on representative local footage before operational use.

Likely next modules are a calibrated perspective mapper, an overtaking evidence policy injected through `OvertakingClearancePolicy`, and an RTSP implementation of `FrameSource`. Those additions do not require rewriting the current traffic-rule state machine.

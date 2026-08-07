# Benchmark data layout

Keep annotation JSON and manifest YAML in Git; keep large or third-party videos out of Git unless their redistribution rights are explicit.

```text
data/benchmark/
  annotations/       # schema 1.0 ground truth
  manifests/         # scenario lists, splits, configs, matching settings
  predictions/       # tiny synthetic cache only; real caches normally go to benchmark_output/
  videos/            # local raw videos, ignored by Git
  examples/           # synthetic report generated from committed fixtures
```

`benchmark_manifest.example.yaml` is a disabled template. Copy it, add local video and annotation paths, and enable only complete entries. Paths are resolved relative to the manifest file.

`synthetic_manifest.yaml` and `predictions/synthetic_video_a.json` are a `SYNTHETIC INTEGRITY TEST` exercising roles, ignore thresholds, confidence filtering, optimal matching, suppression controls, accounting, duration evidence, fingerprints, and reports without a video, detector, tracker, or model download. Their results are not an accuracy claim.

Annotation labels:

- `unnecessary_left_lane_occupation`: canonical positive review-candidate ground truth.
- `legitimate_overtaking`, `congestion_left_lane_use`, `right_lane_unavailable`, `temporary_left_lane_use`, and `geometry_invalid`: negative controls used for suppression metrics, never generic ignore regions.
- `insufficient_evidence`: canonical ignore region. A prediction is ignored only when configured prediction-coverage and IoU thresholds pass.
- `camera_motion` and `lane_assignment_uncertain`: diagnostic context only.

Confidence does not change semantic role. A tiny overlap with an ignored annotation is not sufficient to remove a prediction from false-positive accounting. All ignored and confidence-filtered predictions remain in detailed report accounting.

Use anonymous identifiers such as `annotator_a`; do not put personal information, plate text, faces, or identity data in annotations.

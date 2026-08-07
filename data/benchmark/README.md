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

`synthetic_manifest.yaml` and `predictions/synthetic_video_a.json` exercise matching and reports without a video, detector, tracker, or model download. Their results are deliberately labeled synthetic and are not an accuracy claim.

Annotation labels:

- `unnecessary_left_lane_occupation`: positive review-candidate ground truth.
- `legitimate_overtaking`, `congestion_left_lane_use`, `right_lane_unavailable`, and `temporary_left_lane_use`: negative-control behaviors used for suppression metrics.
- `insufficient_evidence`: the annotator could not confidently decide.
- `geometry_invalid`, `camera_motion`, and `lane_assignment_uncertain`: diagnostic context, not violations.

Use anonymous identifiers such as `annotator_a`; do not put personal information, plate text, faces, or identity data in annotations.

# Benchmark data layout

Phase 4.2 real-dataset metadata includes `intake_registry.json`, `dataset_release.json`, both dataset coverage formats, and `REAL_DATASET_STATUS.md`. These record provenance, hashes, permissions, protocol versions, real counts, splits, and quality gates without committing raw video. Independent annotation work remains separate and only approved adjudication enters validation/test benchmark ground truth. See `docs/annotation_handbook.md`.

## Release Integrity

The registry's source SHA-256, byte size, and `source_group_id` are authoritative. `video_id` alone never binds annotations or adjudication to footage. Release construction rechecks every annotation and adjudication against the registry, confirms adjudication references the current locked annotation hashes, and rejects source-group or identical-byte cross-split leakage. Locked does not mean correctly sourced. Official release output is fail-closed and atomically replaced only after integrity passes.

Perfect annotator agreement does not make ground truth valid if both annotators labeled the wrong source video.

A split assignment is not trusted merely because it was produced by the official split tool; release-time leakage checks are mandatory.

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

## Identity and cache migration

A video ID or filename is not content identity. New inference-generated prediction
caches use schema 1.1 and preserve the source video's streaming SHA-256 plus byte size.
Cache-only evaluation uses this evidence when the raw source is unavailable and checks
it against current raw bytes when the file is present. A mismatch stops evaluation.

Legacy caches without these fields remain readable but make the dataset identity
`unverified`, so strict baseline comparison is disabled. Re-run inference to migrate
such caches. The committed synthetic cache intentionally has no real source video and
therefore remains unverified; it tests metric integrity and makes no official
comparability or real-world performance claim.

Dataset fingerprints combine source content identity with current annotation hashes.
Evaluation fingerprints separately include protocol `4.1.1`, component semantic
versions, and evaluation settings. Production-policy hashes remain separate so policy
changes can be measured without pretending that protocol changes are model changes.

Use anonymous identifiers such as `annotator_a`; do not put personal information, plate text, faces, or identity data in annotations.

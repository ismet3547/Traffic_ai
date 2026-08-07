# Mini Pilot Operator Checklist

Pilot ID: `mini-pilot-001`

Target: 5–10 legally usable real clips, approximately 10–30 minutes total

Preferred clip context: 30–180 seconds, without forcing artificial cuts

Frozen semantics: dataset `pilot-0.1`, ontology `pilot-1`, handbook `1.0`, canonical agreement protocol `2` / config `1`

Do not show model predictions to annotators or adjudicators before the applicable ground truth is finalized and locked. Do not change production thresholds after inspecting individual clips. Do not report synthetic fixtures as pilot evidence.

## Before selecting footage

- Confirm a documented right to use every source for benchmark work.
- Prefer `permission_status=verified`; never put uncertain footage in the official validation pilot.
- Assign one `source_group_id` to all clips from the same recording/session before splitting.
- Seek actual diversity where available: legitimate overtaking, unnecessary left-lane occupation, congestion, right-lane unavailability, a hard negative, truck/bus interaction, partial occlusion, and one geometry/camera-quality challenge.
- Record actual coverage honestly; do not invent missing categories.
- Keep plates, faces, and identities out of annotations and notes.

## 1. Register the first real clip

Set values for your legally usable local file. Add `--redistribution-allowed` only when the license genuinely permits redistribution.

```powershell
$PilotVideo = "D:\permitted_traffic\clip_001.mp4"
$PilotVideoId = "mini_pilot_001_clip_001"
$PilotSourceGroup = "recording_session_001"

python -m app.tools.register_benchmark_video `
  --video $PilotVideo `
  --registry data/benchmark/intake_registry.json `
  --video-id $PilotVideoId `
  --source-group-id $PilotSourceGroup `
  --source-type licensed_source `
  --source-reference "REPLACE_WITH_LICENSE_OR_PERMISSION_REFERENCE" `
  --acquisition-date 2026-08-08 `
  --permission-status verified `
  --benchmark-use-allowed `
  --scenario-tag daylight `
  --scenario-tag fixed_camera `
  --scenario-tag free_flow `
  --vehicle-class passenger_car
```

The command records SHA-256, byte size, duration, resolution, and FPS. Never bypass it. Then add a matching entry to `data/benchmark/pilot/mini_pilot_manifest.json`:

```json
{
  "video_id": "mini_pilot_001_clip_001",
  "real_world_source_confirmed": true,
  "local_video_path": "D:/permitted_traffic/clip_001.mp4",
  "production_config_path": "../../../configs/demo_fixed_camera.yaml",
  "selection_notes": "Why this legally usable clip was selected",
  "annotation_duration_minutes": {}
}
```

Use the actual local path. Keeping raw video outside Git is expected. Run the status command after every material stage:

```powershell
python -m app.tools.pilot_status `
  --manifest data/benchmark/pilot/mini_pilot_manifest.json `
  --output-json data/benchmark/pilot/pilot_status.json `
  --output-markdown data/benchmark/pilot/pilot_status.md
```

## 2. Prepare source-group-safe development/validation splits

Add one object per selected clip to `data/benchmark/pilot/split_candidates.json`, using the registry's exact `video_id`, `source_group_id`, duration, and actual tags. Prefer development plus validation for this tiny pilot; leave test at zero unless a locked test subset is explicitly justified.

```powershell
python -m app.tools.assign_dataset_splits `
  --candidates data/benchmark/pilot/split_candidates.json `
  --output data/benchmark/pilot/split_assignments.json `
  --seed 42 `
  --development 0.60 `
  --validation 0.40 `
  --test 0.00
```

Inspect `split_assignments.json`. Clips sharing a source group must never cross splits.

## 3. Annotator A — blind pass and lock

Prepare event JSON from the handbook without viewing model output or Annotator B's work. An empty list is a valid explicit no-event annotation.

```powershell
python -m app.tools.annotate_video `
  --video $PilotVideo `
  --video-id $PilotVideoId `
  --annotator-id annotator_a `
  --event-json data/benchmark/pilot/work/clip_001_annotator_a_events.json `
  --output data/benchmark/pilot/annotations/clip_001_annotator_a.json `
  --lock

python -m app.tools.validate_annotations `
  --dataset-annotation data/benchmark/pilot/annotations/clip_001_annotator_a.json
```

Optionally record coarse minutes under the clip's `annotation_duration_minutes.annotator_a`. This estimates dataset cost; it is not personal productivity monitoring.

## 4. Annotator B — independent blind pass and lock

Annotator B must not see Annotator A's JSON, agreement output, adjudication, predictions, or candidate video.

```powershell
python -m app.tools.annotate_video `
  --video $PilotVideo `
  --video-id $PilotVideoId `
  --annotator-id annotator_b `
  --event-json data/benchmark/pilot/work/clip_001_annotator_b_events.json `
  --output data/benchmark/pilot/annotations/clip_001_annotator_b.json `
  --lock

python -m app.tools.validate_annotations `
  --dataset-annotation data/benchmark/pilot/annotations/clip_001_annotator_b.json
```

## 5. Generate canonical official agreement

Official mode prohibits per-clip agreement-rule overrides.

```powershell
python -m app.tools.compare_annotations `
  data/benchmark/pilot/annotations/clip_001_annotator_a.json `
  data/benchmark/pilot/annotations/clip_001_annotator_b.json `
  --official `
  --output data/benchmark/pilot/agreements/mini_pilot_001_clip_001.json
```

After the first 3–5 double-annotated clips, stop. Review overall and positive-event agreement plus label, boundary, missing-event, confidence, and vehicle-reference disagreements. Record findings in `first_agreement_review.md` and `handbook_issues.md`; add reviewed IDs to the pilot manifest only after this review is complete.

If the handbook changes, bump its version, record the change, identify affected clips, and re-review them consistently. Never mix incompatible semantics silently.

## 6. Adjudicate every disagreement and lock

Create a decisions JSON with a brief rationale for every disagreement. The adjudicator still must not view model output.

```powershell
python -m app.tools.adjudicate_annotations `
  data/benchmark/pilot/annotations/clip_001_annotator_a.json `
  data/benchmark/pilot/annotations/clip_001_annotator_b.json `
  --official `
  --adjudicator-id adjudicator_01 `
  --decisions data/benchmark/pilot/work/clip_001_decisions.json `
  --output data/benchmark/pilot/adjudications/mini_pilot_001_clip_001.json `
  --lock
```

Summarize recurring disagreement causes in `pilot_annotation_summary.md`: overtaking boundaries, right-lane availability, visibility, target identity, congestion, or another documented category.

## 7. Build the fail-closed dataset release

```powershell
python -m app.tools.build_dataset_release `
  --registry data/benchmark/intake_registry.json `
  --splits data/benchmark/pilot/split_assignments.json `
  --annotations-dir data/benchmark/pilot/annotations `
  --agreements-dir data/benchmark/pilot/agreements `
  --adjudications-dir data/benchmark/pilot/adjudications `
  --output data/benchmark/pilot/dataset_release.json
```

Do not continue if any integrity or canonical-agreement gate fails.

## 8. Export locked benchmark ground truth

Repeat for every released clip:

```powershell
python -m app.tools.export_adjudicated_benchmark `
  --adjudication data/benchmark/pilot/adjudications/mini_pilot_001_clip_001.json `
  --registry data/benchmark/intake_registry.json `
  --splits data/benchmark/pilot/split_assignments.json `
  --annotations-dir data/benchmark/pilot/annotations `
  --release data/benchmark/pilot/dataset_release.json `
  --output data/benchmark/pilot/ground_truth/mini_pilot_001_clip_001.json
```

Only now add and enable the clip in `data/benchmark/pilot/benchmark_manifest.yaml`, pointing to the exact video, production config, and adjudicated ground-truth JSON.

## 9. Run the untouched current system and benchmark

Commit all intended code/config changes first and ensure `git status` is clean. Run all pilot clips without inspecting per-clip output or tuning thresholds:

```powershell
git rev-parse HEAD
git status --short

python -m app.tools.run_benchmark `
  --manifest data/benchmark/pilot/benchmark_manifest.yaml `
  --output benchmark_output/mini-pilot-001/current_run `
  --split all
```

This captures the exact commit, resolved/production config hashes, detector model, ByteTrack identifier, benchmark protocol, source identities, dataset fingerprint, predictions, raw counts, policy suppression, performance, and failure artifacts.

Do not open predictions or benchmark failure output yet.

## 10. Freeze Pilot Baseline 0 and unlock post-hoc review

The freeze command rejects synthetic output, dirty-worktree runs, unverified dataset identity, incomplete clip coverage, or any clip whose adjudicated GT was not finalized and locked. It copies the run and provenance into a new directory and refuses to overwrite it.

```powershell
python -m app.tools.freeze_pilot_baseline `
  --manifest data/benchmark/pilot/mini_pilot_manifest.json
```

Only a successful freeze authorizes post-hoc model review. Never rerun into or edit `pilot_baseline_0`.

## 11. Inspect every false positive and false negative

```powershell
python -m app.tools.inspect_failures `
  --report benchmark_output/mini-pilot-001/pilot_baseline_0/benchmark_report.json `
  --kind all `
  --limit 10000
```

Review every FP and FN. In `pilot_failure_summary.md`, record a brief human note and suspected category such as detector miss, ID switch, lane assignment, right-lane opportunity, overtaking policy, congestion, geometry gate, lifecycle, or GT ambiguity. Rank patterns by safety impact, frequency, likely recurrence, and fixability.

Do not fix one clip opportunistically. Any proposed change must state the repeated failure pattern, general rule, affected development clips, and regression risk. Tune only on development clips; do not reuse validation results as unbiased evidence after tuning.

## 12. Complete summaries and scale-up decision

Update:

- `pilot_annotation_summary.md`
- `pilot_failure_summary.md`
- `pilot_benchmark_summary.md`
- `handbook_issues.md`
- generated `pilot_status.json` and `pilot_status.md`

Always report raw TP/FP/FN alongside precision, recall, F1, and FP/hour. Include legitimate-overtaking suppression, congestion suppression, and geometry fail-closed observations with denominators. Keep the warning: **Mini-pilot sample size is too small for production accuracy claims.**

Choose `GO`, `CONDITIONAL GO`, or `NO-GO` only after evaluating ontology stability, annotator understanding, recurring disagreement manageability, release tooling, end-to-end benchmark operation, failure interpretability, dataset integrity, and whether failures form actionable patterns.

After every FP/FN has a human note, set `failure_review_completed` to `true` in the pilot manifest. Set `scale_up_recommendation` only after documenting the reasons in all three summaries. The status tool does not mark the pilot executed until the immutable baseline, complete failure review, and an explicit scale-up recommendation all exist.

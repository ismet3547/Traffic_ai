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

After the configured first-review count (3–5, default 3) has current canonical reports, stop. The system deterministically selects the first N valid reports by `video_id`; operators cannot cherry-pick clips. Prepare a structured summary JSON and create the provenance-bound review:

```powershell
python -m app.tools.review_initial_agreement `
  --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json `
  --summary data/benchmark/pilot/work/first_agreement_summary.json
```

The summary records recurring disagreement categories, optional `handbook_issue_ids`, an action, and notes. `HANDBOOK_REVISION` requires at least one issue ID. Completion requires every exact `agreement_id` and `agreement_content_sha256`; annotation or protocol revisions make the artifact stale.

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

First enumerate the immutable required set without writing completion evidence:

```powershell
python -m app.tools.review_pilot_failures `
  --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json
```

Review every FP and FN, then supply a structured JSON review list with the exact deterministic failure IDs and their copied semantic identity fields:

```powershell
python -m app.tools.review_pilot_failures `
  --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json `
  --reviews data/benchmark/pilot/work/failure_reviews.json
```

Each entry requires a controlled category, note, severity, systematic-risk assessment, and proposed action. The command refuses partial, duplicate, unknown, identity-mismatched, tampered, or stale evidence and generates `pilot_failure_summary.md` from validated coverage. A zero-FP/zero-FN baseline needs no artificial review artifact and is explicitly reported as “No FP/FN review required for this baseline.”

**Every FP and FN in the frozen baseline must be accounted for exactly once before failure review is complete.**

Do not fix one clip opportunistically. Any proposed change must state the repeated failure pattern, general rule, affected development clips, and regression risk. Tune only on development clips; do not reuse validation results as unbiased evidence after tuning.

## 12. Complete summaries and scale-up decision

Update:

- `pilot_annotation_summary.md`
- `pilot_failure_summary.md`
- `pilot_benchmark_summary.md`
- `handbook_issues.md`
- generated `pilot_status.json` and `pilot_status.md`

Always report raw TP/FP/FN alongside precision, recall, F1, and FP/hour. Include legitimate-overtaking suppression, congestion suppression, and geometry fail-closed observations with denominators. Keep the warning: **Mini-pilot sample size is too small for production accuracy claims.**

Choose `GO`, `CONDITIONAL_GO`, or `NO_GO` only after evaluating ontology stability, annotator understanding, recurring disagreement manageability, release tooling, end-to-end benchmark operation, failure interpretability, dataset integrity, and whether failures form actionable patterns. Record that human judgment as an evidence-bound artifact:

```powershell
python -m app.tools.record_scale_up_decision `
  --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json `
  --decision GO `
  --rationale "REPLACE_WITH_EVIDENCE_BASED_RATIONALE" `
  --known-limitation "Mini-pilot evidence does not establish production accuracy."
```

Use one or more `--condition` values for `CONDITIONAL_GO`; use one or more `--known-blocker` values for `NO_GO`. The command refuses missing/stale baseline, failure-review, agreement-review, release, or benchmark evidence. Any referenced artifact change invalidates the old decision.

**Pilot completion is derived from review artifacts; it cannot be declared by editing a status flag.** Legacy `failure_review_completed`, `first_agreement_review_video_ids`, and `scale_up_recommendation` manifest values are ignored and reported as informational migration notices.

## Current terminal-state eligibility

Current eligibility and historical evidence are separate. A valid historical GO decision does not override a current integrity blocker. A pilot cannot be COMPLETE while any completion-blocking issue is active.

Before treating any terminal state as current, rerun `app.tools.pilot_status`. It rechecks source confirmation, benchmark authorization, permission, local source/config presence and identity, frozen protocol identity, release/provenance, baseline currency, exact review evidence, and scale-up-decision currency. If one becomes invalid, the old evidence remains readable but current state falls back to a non-terminal state and `pilot_executed` becomes false.

Local source presence is not sufficient. The local video bytes must match the source SHA-256 registered during intake. The status check compares the registered byte size first and then computes a streaming SHA-256 when the size matches. `LOCAL_VIDEO_MISSING` identifies absence; `LOCAL_VIDEO_SOURCE_IDENTITY_MISMATCH` identifies an existing file with the wrong bytes.

Replacing a video at the same path invalidates current pilot eligibility until the registered source bytes are restored or the clip is explicitly re-registered. Pilot status does not update the registry. Use the explicit intake/re-registration workflow when the intended source truly changes; otherwise restore the original registered bytes and rerun status.

Status separates:

- `BLOCKER`: prevents terminal completion; unknown new reason codes fail closed into this class.
- `WARNING`: visible but non-blocking, including the mini-pilot sample-size caveat.
- `INFO`: operational notice, including ignored legacy completion fields.

Restoring current eligibility may restore completion when all referenced evidence is still current. Protocol or evidence changes that invalidate identities require the documented migration/re-review process rather than editing status.

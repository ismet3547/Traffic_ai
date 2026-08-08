# First Agreement Review

Status: **NOT STARTED — NO REAL DOUBLE-ANNOTATED CLIPS**

This prose file is informational only. Official completion requires
`first_agreement_review.json`, created by:

```powershell
python -m app.tools.review_initial_agreement `
  --pilot-manifest data/benchmark/pilot/mini_pilot_manifest.json `
  --summary data/benchmark/pilot/work/first_agreement_summary.json
```

The system selects the configured first N valid canonical reports by `video_id`
and binds the review to every exact agreement ID and content SHA-256. A hand-edited
video-ID list cannot satisfy this gate.

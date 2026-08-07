# Real Dataset Status

Last updated: 2026-08-08

- Real pilot clips registered: **0**
- Total real footage duration: **0 seconds**
- Double-annotated real clips: **0**
- Adjudicated real clips: **0**
- Locked real test clips: **0**
- Benchmark-exported real clips: **0**
- Inference-complete real clips: **0**
- Benchmark-complete real clips: **0**

**REAL PILOT STATUS: BLOCKED — NO REAL SOURCE VIDEO REGISTERED**

No real videos or human annotations are committed or claimed. Existing repository examples and Phase 4.2 lifecycle artifacts are synthetic. Phase 4.3 first targets 5–10 legally usable clips and roughly 10–30 minutes so every workflow problem and every FP/FN can be inspected before considering 30–50 clips.

Generate the current status from actual artifacts:

```powershell
python -m app.tools.pilot_status --manifest data/benchmark/pilot/mini_pilot_manifest.json --output-json data/benchmark/pilot/pilot_status.json --output-markdown data/benchmark/pilot/pilot_status.md
```

See `docs/mini_pilot_operator_checklist.md` for the exact first-clip workflow. No TP, FP, FN, precision, recall, F1, or FP/hour values exist yet.

# SYNTHETIC INTEGRITY TEST - NOT REAL-WORLD PERFORMANCE

This example comes only from committed fake annotations and fake predictions. It validates benchmark integrity and arithmetic; no traffic video, detector, tracker, or real driving behavior was evaluated.

| TP | FP | FN | Ignored | Confidence-filtered | Precision | Recall | F1 | FP/video hour |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 1 | 1 | 1 | 0.3333 | 0.6667 | 0.4444 | 72.0000 |

Exact accounting over the 200-second fixture:

- Three positive GT events produce two matches and one FN.
- Eight prediction records produce one confidence-filtered prediction and seven considered predictions.
- The seven considered predictions reconcile to two TP, four FP, and one ignored prediction.
- The four FPs are a duplicate prediction, a genuine unrelated prediction, a tiny-overlap-with-ignore prediction, and a candidate during a legitimate-overtaking control.
- The mostly-contained ignore prediction has 100% prediction coverage and is retained in ignored-prediction diagnostics.
- One of two overtaking controls is a suppression failure; the congestion control is not.
- The manifest and annotation durations agree at 200 seconds, producing `consistent_multiple_sources` and medium denominator confidence.
- Evaluation protocol is `4.1.1`; matcher semantics are `maximum_cardinality_then_maximum_total_temporal_iou_deterministic_v2`.
- Dataset identity is `unverified` because this integrity fixture has no source-video bytes. It is intentionally ineligible for strict regression comparison.

Recreate the full deterministic JSON and Markdown output with the synthetic replay command in the root README.

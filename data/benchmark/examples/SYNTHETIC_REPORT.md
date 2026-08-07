# SYNTHETIC EXAMPLE - NOT REAL-WORLD PERFORMANCE

This example is generated from the committed fake annotations and fake prediction cache. It validates benchmark arithmetic only; no traffic video, detector, tracker, or real driving behavior was evaluated.

| TP | FP | FN | Precision | Recall | F1 | FP/video hour |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0 | 0.6667 | 1.0000 | 0.8000 | 30.0000 |

Fixture definition:

- Two synthetic `unnecessary_left_lane_occupation` intervals are matched.
- One synthetic candidate overlaps a `legitimate_overtaking` control interval and is counted as an FP.
- The fake video duration is 120 seconds, so one FP corresponds to 30 FP/video hour.
- The suspected failure hint is `OVERTAKING_LOGIC_ERROR`; it is a heuristic label, not a proven cause.

Recreate the complete JSON and Markdown reports with the synthetic replay command in the root README.

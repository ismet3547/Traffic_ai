# Mini Pilot Failure Review

Status: **NOT AVAILABLE — PILOT BASELINE 0 IS NOT FROZEN**

No real FP/FN set exists, so no failure review artifact or performance claim is
present. After baseline freeze, `app.tools.review_pilot_failures` derives the exact
required set from the frozen benchmark report and regenerates this summary from a
validated `failure_review.json`.

Every FP and FN in the frozen baseline must be accounted for exactly once before
failure review is complete. Partial, duplicate, unknown, identity-mismatched,
tampered, and stale review evidence does not count.

> Mini-pilot sample size is too small for production accuracy claims.

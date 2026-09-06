# Deploy the remaining-afternoon-wind control

## Decision

The user requested merging PR #17 and deploying its selected model. PR #16
supplies the feature pipeline and was merged first; PR #17 was then retargeted
to main and merged. This deployment uses the existing fitted 94-feature control
from `artifacts/remaining_wind_current_definition/model.joblib`; no retraining,
threshold selection, new data source, or 2026 evaluation is performed.

The target is at least 30 minutes of wind at 18.52 km/h from 225°–315° in
`[max(12:00, issue time), 20:00)`, May–September, Europe/Paris. Both ongoing
and later spells count. Scoring runs from 06:30 through 19:30 inclusive.

## Serving changes

`scripts.remaining_wind export` wraps the fitted pipeline in the existing
schema-v2 inference envelope and retains the research provenance. The model
kind is `remaining_wind`. Its alert threshold is null, because the research
comparison selected a probability estimator without selecting an alert rule.
The previous onset model's threshold and onset companion are not transferred
to the new target.

Live responses retain `traverse_probability`, return `predict_traverse: null`
and empty `onset_within_probabilities`, and identify the target window. An
observed past onset is still available as monitoring information but never
suppresses the remaining-window probability. Less than 30 minutes remaining
produces `outside_prediction_window` without a model score.

The public page describes the new probability and development results. Old
2026 audit metrics and existing archive points are explicitly identified as
previous-model results. No historical predictions or reported evaluations are
rewritten. The old onset evaluators reject this bundle to prevent evaluation
against incompatible labels.

## Reproduce the deployment artifact

First reproduce the fitted control using the commands in the
[research record](../20260905-remaining-wind-model/plan.md), or use its existing
trusted local output. Then:

```bash
uv run --frozen python -m scripts.remaining_wind export \
  --output-dir artifacts/remaining_wind_current_definition
uv run --frozen python -m unittest discover -v
uv run --frozen python realtime_inference.py \
  --model artifacts/remaining_wind_deployment/traverse_model.joblib
```

`deployment_model.json` pins release `model-2026-09-06-remaining-wind` to SHA-256
`8badd03cc4664048b0838acc0a3d21c13cea8ac062fef0d49324675954122d19`.
The generated bundle remains outside Git and is distributed as the release
asset `traverse_model.joblib`. Restart the existing five-minute publisher after
updating its checkout so its in-memory inference code matches the new bundle.

## Validation

- All 54 tests pass, including a focused export/live regression: identical
  scores, an observed onset does not suppress the forecast, no inherited alert
  or onset estimate, scoring at 19:30, and no score at 19:31.
- Exported probabilities exactly match the research pipeline on all 34,393
  fitted rows. This is a serialization check, not a new accuracy measurement.
- Live inference at 16:17 on 2026-09-06 successfully constructed all 94 inputs
  from current Windbird and all four Météo-France stations, and returned
  probability `0.0357118663` for the window ending at 20:00.
- `git diff --check` passes.

The historical development AP comparison and its limitations remain those in
the research record. Early-morning ranking is still weaker than the previous
model at 08:00; this deployment does not claim a new independent test result.

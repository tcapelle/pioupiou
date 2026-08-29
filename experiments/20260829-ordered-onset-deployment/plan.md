# Ordered onset horizons with a completed-season deployment refit

## Question

Can the deployable model use every completed season and also provide coherent
near-term onset estimates from features available at inference time?

## Decisions and evaluation boundary

The primary advance-warning architecture, features, and threshold remain the
ones selected on 2017--2022 folds and 2023 validation. Its serialized pipeline
is refit through the completed 2025 season, as specified in the earlier
[deployment-refit record](../20260825-deployment-refit/plan.md).

The onset companion is a different target, not an improvement to the primary
same-day target. It uses one four-class estimator for mutually exclusive onset
intervals: within 60 minutes, 60--120 minutes, 120--180 minutes, or later/no
onset. Summing the interval estimates produces cumulative 1 h, 2 h, and 3 h
outputs that always satisfy `P(1 h) <= P(2 h) <= P(3 h)`.

The 2024--2025 and partial 2026 results had already been inspected during the
preceding experiments. They are audits, not untouched evidence used to select
this formulation. The ordered formulation was chosen because cumulative
probabilities must be coherent; future observations are needed for a clean
prospective comparison.

## Inference contract

The companion reuses the primary model's 94-feature schema, so it introduces no
new data source or inference-only approximation. Training and retrospective
evaluation are restricted to rows before the first qualifying onset. In live
inference, once the existing sustained-run rule can observe an onset, the API
returns status `onset_observed` and sets all future-onset estimates to zero.

The model detects a qualifying run only after its 30-minute duration is
observable, although the recorded onset is the beginning of that run. This
confirmation delay is explicit and should not be interpreted as advance
forecast performance.

## Results

The primary 2024--2025 evaluation remains unchanged: three-hour event-day AP
`0.1905`, ROC AUC `0.7166`, 34.8% event coverage at least three hours early,
and 14.8% false-alert days. On the later data through 2026-08-17, the deployment
refit reaches AP `0.2674`, AUC `0.6584`, 5/10 event coverage, and 15.6%
false-alert days.

The ordered onset companion has the following row-level audit metrics:

| Slice | Horizon | Prevalence | AP | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| 2024--2025 evaluation | 1 h | 0.67% | 0.134 | 0.904 |
| 2024--2025 evaluation | 2 h | 1.33% | 0.143 | 0.877 |
| 2024--2025 evaluation | 3 h | 2.00% | 0.137 | 0.855 |
| 2026 later audit | 1 h | 0.92% | 0.262 | 0.921 |
| 2026 later audit | 2 h | 1.84% | 0.273 | 0.887 |
| 2026 later audit | 3 h | 2.76% | 0.239 | 0.844 |

Discrimination by ROC AUC is strongest for the shortest horizon on both
slices. AP is not monotonic, so this is not described as accuracy increasing
toward onset. The validation-balanced thresholds are diagnostic operating
points, not recommended alerts: precision on 2024--2025 is only 3.4%, 8.2%,
and 8.5% for 1 h, 2 h, and 3 h. The dashboard therefore plots the estimates and
reports AP without adding thresholded horizon alerts.

The generated 2026 file contains 2,177 rows and zero cumulative-order
violations. Mean estimates are close to but not identical to prevalence; the
outputs have not received a separate probability-calibration fit.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260829-ordered-onset-deployment-refit \
  --wandb-mode disabled
uv run --frozen python -m scripts.predict_dataset \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
uv run --frozen python -m unittest discover -v
```

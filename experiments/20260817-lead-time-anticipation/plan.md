# Lead-time-weighted Traverse anticipation

## Question

Can the same observations produce useful advance warnings rather than high
late-day scores driven by direct evidence that the Traverse is underway?

The prior model labeled every checkpoint on a positive day as positive. This
made post-event rows nearly deterministic and caused pooled timestep metrics to
measure a mixture of forecasting and nowcasting.

## Outcome and objective

Event onset is the first observation in `[12:00, 20:00)` satisfying the wind
speed and direction thresholds on a day that satisfies the complete Traverse
label. This is deliberately the beginning of qualifying wind, not the later
time at which 30 cumulative minutes make the daily label provable.

- Positive rows at or after onset are excluded from training and evaluation.
- Remaining positive rows receive
  `min(minutes_before_onset / 180, 1)` weight.
- Negative rows receive weight 1 and remain available throughout the day.
- The weights are used when fitting logistic regression.
- L2 is selected on expanding-year event-day AP at three hours lead.
- The decision threshold is selected on 2023 by balanced event-day accuracy:
  three-hour alert rate versus the fraction of negative days without any alert.

All observations remain strictly earlier than the issue time. The chronological
split remains 2017–2022 train, 2023 validation, and 2024–2025 held-out test.
The onset-aware dataset has 79,944 rows. No held-out result was used for model,
feature, regularization, or threshold selection.

## Validation decisions

Linear lead weights improved three-hour validation AP from `0.3059` with
uniform weights to `0.3201`. Quadratic weights reached `0.3239`, a small
post-hoc difference that did not justify departing from the specified linear
value function.

The same-day Traverse progress block was removed. It is largely direct event
evidence, has an awkward interpretation after positive rows are censored, and
did not improve the advance-warning validation result robustly. With the block,
validation AP was `0.3224`, three-hour alert rate `70.5%`, and false-alert-day
rate `33.8%`. Without it, AP was `0.3201`, alert rate `72.1%`, and the same
false-alert-day rate. Expanding-year AP favored removal (`0.3348` versus
`0.3302`), and the smaller feature contract is easier to interpret.

Four small histogram-gradient-boosting variants were also checked on validation
only. Their three-hour AP ranged from `0.2537` to `0.2818`, below logistic
regression, so the existing estimator family was retained.

## Held-out result

Both models below use thresholds selected independently on 2023 with the new
event-day threshold policy. The legacy model is scored only on pre-onset rows,
with its original progress features intact.

| Model | 3 h event-day AP | 3 h ROC AUC | Alert ≥3 h | Alert ≥2 h | Alert ≥1 h | False-alert days | Median warning |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy nowcasting objective | 0.2738 | 0.6605 | 67.5% | 70.9% | 72.6% | 41.2% | 426 min |
| Onset-censored, linear lead weight | **0.3409** | **0.7036** | **68.4%** | **70.1%** | **72.6%** | **36.6%** | **438 min** |

The new model improves threshold-independent three-hour discrimination and
reduces false-alert days while preserving early event coverage. It no longer
claims performance from checkpoints after the event has begun.

## Later 2026 dashboard holdout

After the model and threshold were frozen, the subsequently added May–August
2026 archive provided 86 scored days, including 17 positive days. Three-hour
event-day AP is `0.3043`, ROC AUC is `0.6189`, and `70.6%` of events are alerted
at least three hours early. The false-alert-day rate is `52.2%`, higher than the
2024–2025 result, and median warning among alerted events is `529` minutes.

This partial-season result was not used to change the model. It is exposed in
the dashboard together with each day's probability trajectory, decision
threshold, and retrospective event-onset marker so the increased false-alert
rate remains visible rather than being hidden by pooled timestep metrics.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260817-lead-time-final \
  --wandb-mode disabled
uv run --frozen python -m unittest discover -v
```

Generated data, the filtered weather cache, model artifacts, and metrics remain
ignored. The model metadata records dataset, source, runtime, and environment
hashes.

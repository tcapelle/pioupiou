# Train through 2025 and test once on 2026

## Question

Can the evaluation contract be reduced to one historical training period and
one forward test without retaining a permanent validation split or a separate
deployment-refit model?

## Protocol

- Train period: every usable row from 2017 through partial 2025.
- Test period: partial 2026 through August 30.
- Selection: expanding-year out-of-fold predictions for 2020 through 2025.
  Each fold trains only on earlier years.
- L2 selection: mean three-hour event-day average precision across folds.
- Alert threshold: event-day balanced accuracy on the concatenated out-of-fold
  predictions.
- Final fit: one pipeline on all 2017-2025 rows after selection.
- Test isolation: no 2026 row selects a feature, hyperparameter, or threshold.

The target, 94 inference-time features, shallow histogram booster, day-equal
lead weighting, and ordered onset companion remain unchanged.

## Data

| Role | Years | Rows | Days | Events |
|---|---|---:|---:|---:|
| Final training fit | 2017-2025 | 33,237 | 1,279 | 191 |
| Rolling OOF selection | 2020-2025 | 22,413 | 862 | 132 |
| Forward test | partial 2026 | 2,527 | 100 | 11 |

## Result

Rolling selection retains `L2=10` and selects threshold `0.305094`, higher than
the former single-2023 threshold `0.228805`.

| Slice | 3 h event-day AP | 3 h ROC AUC | Alert >=3 h | False-alert days | Median warning |
|---|---:|---:|---:|---:|---:|
| Rolling 2020-2025 OOF | 0.2331 | 0.6243 | 50.0% | 27.3% | 525 min |
| Partial 2026 test | 0.1747 | 0.5598 | 27.3% (3/11) | 20.2% (18/89) | 478 min |

The final fitted probabilities are unchanged from the earlier 2017-2025
deployment refit because the estimator and fit rows are the same. The simpler
protocol changes the threshold: it reduces 2026 false-alert days from 34.8% to
20.2%, while reducing three-hour event coverage from 54.5% to 27.3%. Ranking
remains weak, so the split simplification clarifies rather than fixes the model.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260831-train-through-2025-test-2026 \
  --wandb-mode disabled
uv run --frozen python -m scripts.predict_dataset \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
uv run --frozen python -m unittest discover -v
```

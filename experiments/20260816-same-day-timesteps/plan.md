# Same-day Traverse probability at arbitrary times

## Objective

Return the probability that a qualifying Traverse occurs during the current
local day's fixed `[12:00, 20:00)` window, given everything observed before an
arbitrary forecast time.

The target does not shrink as the day advances. A Traverse that occurred at
14:00 still makes the day's label positive
at 18:00. Later scores are therefore nowcasts of the whole day and may use
strictly earlier evidence that the event is already underway or complete.

## Dataset and features

- 85,116 rows from 3,164 local days.
- 27 training/evaluation checkpoints per complete day, every 30 minutes from
  06:30 through 19:30.
- Smooth sine/cosine clock encoding allows scoring between trained checkpoints.
- PiouPiou and Météo-France values are strictly earlier than the requested
  cutoff, including for non-grid times such as 13:17.
- Same-day progress features record qualifying minutes, longest consecutive
  run, and whether the full event criterion has already been observed.

The model retains median imputation, missingness indicators, standardized
numeric values, and L2 logistic regression. The chronological split remains
2017–2022 train, 2023 validation, and 2024–2025 held-out test. Training
searches `L2 ∈ {1, 10}`; weaker regularization did not converge because
post-event progress can be close to deterministic.

## Held-out scores over the day

The selected model uses `L2=1` (`C=1`) and one validation-selected threshold
of `0.24938`. Pooled across all test timesteps, AP is 0.6096, ROC AUC is 0.8056,
balanced accuracy is 0.7203, and Brier score is 0.1168.

| Time | AP | ROC AUC | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|
| 06:30 | 0.3232 | 0.6774 | 0.6370 | 0.1565 |
| 08:00 | 0.3413 | 0.6941 | 0.6404 | 0.1513 |
| 10:00 | 0.3744 | 0.7095 | 0.6488 | 0.1463 |
| 12:00 | 0.4434 | 0.7443 | 0.6809 | 0.1393 |
| 14:00 | 0.5526 | 0.7749 | 0.6965 | 0.1269 |
| 16:00 | 0.6937 | 0.8349 | 0.7437 | 0.1014 |
| 18:00 | 0.8450 | 0.9141 | 0.8350 | 0.0645 |
| 19:00 | 0.9668 | 0.9857 | 0.9436 | 0.0259 |
| 19:30 | 0.9909 | 0.9967 | 0.9777 | 0.0119 |

The strong late-day scores must not be interpreted as early forecast skill:
by then, the model often has direct evidence that a Traverse occurred. The
pre-noon rows are the cleanest measure of advance forecast performance.

## Arbitrary-time verification

Preparation and scoring were verified at 13:17 on 21 July 2025, a time absent
from the 30-minute training grid. Only observations before 13:17 were used; the
model returned probability `0.21591`.

```bash
uv run python -m scripts.build_timestep_dataset --offline
uv run python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260816-same-day-timesteps \
  --wandb-mode disabled

uv run python -m scripts.prepare_timestep \
  --date 2025-07-21 --time 13:17 \
  --offline-weather \
  --output artifacts/timestep_features_2025-07-21_1317.csv
uv run python -m scripts.predict \
  --model artifacts/traverse_model.joblib \
  --features artifacts/timestep_features_2025-07-21_1317.csv
```

Generated datasets, models, prepared rows, and the complete 27-timestep metric
report remain ignored under `artifacts/`.

## 2026-08-17 feature-block ablation

Before simplifying the repository to this single model, feature blocks were
ablated using only the 2017–2022 training rows and 2023 validation rows. L2 was
held at the selected value of 1. Paired uncertainty intervals resampled whole
validation days 1,000 times, preserving the 27 within-day checkpoints.

| Removed block | Validation AP | Full − ablated AP | Paired 95% interval |
|---|---:|---:|---:|
| none | 0.5923 | — | — |
| season | 0.5868 | +0.0055 | [-0.0082, +0.0192] |
| issue time | 0.5914 | +0.0009 | [-0.0009, +0.0029] |
| historical labels | 0.5911 | +0.0012 | [-0.0160, +0.0168] |
| same-day event progress | 0.4507 | +0.1417 | [+0.0974, +0.1791] |
| PiouPiou wind summaries | 0.5608 | +0.0315 | [+0.0102, +0.0540] |

The 10 historical-label features were removed: their AP effect is negligible
and uncertain, while validation ROC AUC and Brier score were slightly better
without them. Removing them also avoids replaying all prior labels during live
preparation. Event progress and wind summaries have clear positive effects and
remain. The two season and two issue-time encodings remain because they are
cheap, define the model's seasonal/arbitrary-time context, and their point
estimates do not support removal. This is a post-hoc validation analysis; the
held-out years were not used for the trimming decision.

The primary Météo-France block was not re-ablated because the exact filtered
weather cache used by the recorded run is not present in this workspace. It is
retained as part of the already evaluated model; this ablation does not claim a
new causal estimate for that block.

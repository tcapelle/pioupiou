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
- Previous-day and same-season history remains leakage-safe: only dates before
  the forecast date are used.

The model retains median imputation, missingness indicators, standardized
numeric values, and L2 logistic regression. The chronological split remains
2017–2022 train, 2023 validation, and 2024–2025 held-out test. The `same_day`
role searches `L2 ∈ {1, 10}`; weaker regularization did not converge because
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
  --role same_day \
  --wandb-name exp-20260816-same-day-timesteps \
  --wandb-mode disabled

uv run python -m scripts.prepare_timestep \
  --date 2025-07-21 --time 13:17 \
  --offline-weather \
  --output artifacts/timestep_features_2025-07-21_1317.csv
uv run python -m scripts.predict \
  --model artifacts/traverse_model_same_day.joblib \
  --features artifacts/timestep_features_2025-07-21_1317.csv
```

Generated datasets, models, prepared rows, and the complete 27-timestep metric
report remain ignored under `artifacts/`.

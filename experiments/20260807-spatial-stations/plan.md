# 20260807-spatial-stations

Date: 2026-08-07

Question: Do compact, pre-noon observations from stations around Lac du
Bourget improve afternoon Traverse prediction over the current lake sensor plus
CHAMBERY-AIX model?

Status: complete — failed promotion criteria; retain the spatial role as an
opt-in research model.

## Data-quality prerequisite

PP456 stopped providing valid lake coordinates after 2025-08-16 12:00 UTC.
Rows that resume on 2025-08-20 have null coordinates and must not be used as
lake features or labels. Rebuild both models after rejecting PiouPiou rows with
missing or off-site coordinates. This changes the evaluation population, so a
fresh control run is required.

The 1 km fixed-site guard also excludes brief earlier placements, including two
positive validation days in 2023 when the sensor was 1.73 km away. That is an
explicit modeling choice: the target is wind at the lake site, not the sensor
ID wherever it happened to be installed.

## Hypothesis

With the estimator, split, regularization search, threshold policy, and label
held fixed, a compact set of spatial station values will improve test average
precision by at least 0.03 over a freshly trained version of the current model.

The expected signal is the difference between the valley/lake state and the
upstream, ridge, north, and south air masses, rather than more summaries of the
same airport time series.

## Falsifier

The hypothesis is rejected if the spatial model improves test average
precision by less than 0.02, if its paired-day bootstrap interval is centered
at or below zero, or if any apparent gain depends on changing the usable dates,
label, estimator, or decision-threshold procedure.

## Fixed design

- Python 3.11 managed by `uv`; dependencies live in one `pyproject.toml`.
- One daily row and the existing leakage-safe `[06:00, 12:00)` weather window.
- Train through 2022, select the operating threshold on 2023, and evaluate on
  2024 through the last valid PP456 lake observation in August 2025.
- The current scikit-learn preprocessing and L2 logistic regression remain
  fixed so this experiment isolates the value of the new data.
- Optional stations are left-joined. Missing optional observations do not
  remove a day; the lake PiouPiou and CHAMBERY-AIX feeds remain required.
- All values are computed before imputation. Quality code `2`
  remains rejected and observations at or after local noon remain excluded.
- This is a retrospective ablation on an already inspected period, not a new
  unbiased estimate of future production performance.

## Stations

| Role | Station | ID | Distance from PP456 | Elevation | Historical fields used |
|---|---|---:|---:|---:|---|
| lake/south control | CHAMBERY-AIX | 73329001 | 6.73 km | 235 m | existing full weather block |
| west ridge | MONT DU CHAT | 73051001 | 6.65 km | 1496 m | temperature, dew point, humidity, wind |
| west lowland | BELLEY | 01034004 | 16.93 km | 330 m | temperature, dew point, humidity, wind |
| north synoptic | MEYTHET | 74182001 | 30.00 km | 455 m | temperature, dew point, humidity, pressure, wind |
| south valley | MONTMELIAN | 73171002 | 26.47 km | 264 m | temperature, dew point, humidity, wind |

All five official hourly series have records spanning the 2017-2025 modeling
window. The closer BLOYE station is not in the first experiment because its
hourly archive provides temperature and rain but no wind, humidity, or
pressure; adding it would not answer the spatial-flow hypothesis cleanly.

Every cached official row was checked against the configured station location.
All are within the 1 km guard; CHAMBERY-AIX has only a 13 m longitude-rounding
change, and the other four locations are exact.

OpenWindMap is also deferred from the fitted spatial model. Nearby IDs are
valuable for prospective collection, but the historical audit found moved
sensors and short coverage: PP235 changes location, PP635 and PP641 also move,
and the closest north and replacement lake sensors start only in 2026. Those
series must be segmented by coordinates before they can be compared safely.

| OpenWindMap candidate | Current role | Archive finding |
|---|---|---|
| PP235 Parves | west, 12.77 km | records from 2018, but moved before 2021 |
| PP635 Lavours | northwest, 13.41 km | records from 2021, with an early location change |
| PP641 Peyrieu | west, 15.88 km | records from 2022, with a later location change |
| WB1446 Sapenay | north ridge, 13.46 km | begins in 2026 |
| WB2186 | north, 11.40 km | begins in 2026 |
| WB2176 KCB | replacement lake sensor | first record 2026-05-23; no overlap with PP456 |

Corrected build audit:

| Data | Rows | Positives | Date range |
|---|---:|---:|---|
| full table | 3,149 | 601 | 2016-08-09 to 2025-08-15 |
| train | 2,051 | 404 | 2017-01-01 to 2022-12-31 |
| validation | 362 | 61 | 2023-01-01 to 2023-12-31 |
| test | 592 | 117 | 2024-01-01 to 2025-08-15 |

Overall fresh temperature coverage is 96.6% for MONT DU CHAT, 98.9% for
BELLEY, 99.8% for MEYTHET, and 99.5% for MONTMELIAN. Online and offline builds
produce the identical dataset SHA-256
`888055438f67efd8d99df45faecd556827caea748587e9fb9cc0ed12f8bb6729`.

## Compact spatial block

For each optional official station, retain only the core morning observation
count and age plus latest temperature, dew point, humidity, wind speed, wind
direction components, morning temperature change, and mean westerly wind
component. Retain MEYTHET sea-level pressure because it is actually observed
there.

Do not add explicit station-minus-airport gradients: they are exact linear
combinations of values already available to logistic regression and would only
change the geometry of L2 regularization.

No raw images, gridded reanalysis, new model family, large interaction set, or
hyperparameter sweep is in scope.

## Design

This is a small deterministic CPU fit. The smoke pair uses the built-in reduced
split and one regularization value; each run should finish in under one minute.
The full pair uses the fixed four-value regularization search and identical
chronological splits. No GPU is required.

Smoke control (`variant` role):

```bash
uv run python train_traverse_model.py \
  --dataset artifacts/traverse_daily.csv \
  --role variant --smoke --wandb-mode online \
  --wandb-name exp-20260807-spatial-stations-smoke-baseline \
  --wandb-tags smoke,exp/20260807-spatial-stations,baseline
```

Smoke candidate (`spatial` role):

```bash
uv run python train_traverse_model.py \
  --dataset artifacts/traverse_daily.csv \
  --role spatial --smoke --wandb-mode online \
  --wandb-name exp-20260807-spatial-stations-smoke-variant \
  --wandb-tags smoke,exp/20260807-spatial-stations,variant
```

Full control and candidate use the same commands without `--smoke`, named
`exp-20260807-spatial-stations-baseline` and
`exp-20260807-spatial-stations-variant` respectively.

## Smoke

Both CPU smoke runs exited cleanly, finished in W&B, and logged every required
decision and health key with finite values.

| Role | W&B run | Test AP | Recall | Precision at recall ≥ 0.60 |
|---|---|---:|---:|---:|
| control | [w9tbaqc4](https://wandb.ai/capecape/pioupiou-traverse/runs/w9tbaqc4) | 0.3526 | 0.6462 | 0.3514 |
| spatial | [xjasgopb](https://wandb.ai/capecape/pioupiou-traverse/runs/xjasgopb) | 0.3664 | 0.8769 | 0.3630 |

The smoke pair is only an execution/metric gate; it is not the experiment
verdict. Full-run commands remain unchanged.

The first full spatial attempt stopped before W&B initialization because the
`l2=0.01` search folds for 2021 and 2022 reached the 2,000-iteration ceiling.
A diagnostic showed convergence at 2,261 and 2,481 iterations. The only fix was
raising `max_iter` to 3,000; estimator, tolerance, candidates, data, and split
remained fixed. Both smoke roles and both full roles were rerun from the same
fixed source revision. The earlier full control is superseded.

The fixed-revision smoke reruns are the table above. Both are finished, have
all required finite metrics, and share commit `de6e976`. The initial smoke run
IDs `03jhej6j` and `y49svbm2`, plus full control `4m8lms29`, remain in W&B with
`superseded` and `pre-convergence-fix` tags.

## Runs

1. Smoke control: current `variant` feature prefixes on the corrected enriched
   table.
2. Smoke spatial: identical setup plus the compact station prefix.
3. Fresh full control.
4. Fresh full spatial model.

All runs used W&B project `pioupiou-traverse` and tag
`exp/20260807-spatial-stations`. The smoke runs finished and logged the required
metrics before the full pair was launched.

Required W&B keys are `test/average_precision`, `test/balanced_accuracy`,
`test/recall`, `test/precision`, `test/brier_score`,
`test/precision_at_recall_0_60`, `train/log_loss`,
`model/max_abs_coefficient`, and `data/train_rows`.

Active full runs:

| Role | Model role | W&B run | Source commit |
|---|---|---|---|
| control | `variant` | [73fsq5lp](https://wandb.ai/capecape/pioupiou-traverse/runs/73fsq5lp) | `570c832` |
| spatial | `spatial` | [xousyxz6](https://wandb.ai/capecape/pioupiou-traverse/runs/xousyxz6) | `570c832` |

Both runs are finished, have all required finite metrics, and differ in W&B
config only by role, feature list, and input-contract schema. The pinned draft
[W&B comparison report](https://wandb.ai/capecape/pioupiou-traverse/reports/Traverse-spatial-station-ablation-%E2%80%94-2026-08-08--VmlldzoxNzY4ODM5Mg==)
includes both runs and a config comparer.

Historical preparation and scoring for 2025-07-21 also completed successfully
with the spatial artifact. The prepared row matched its model, data, label,
source-schema, and five-station manifest contracts before scoring.

## Decision metrics

Primary:

- test average precision and paired difference from control;
- paired-day bootstrap interval for the average-precision difference.

Secondary:

- balanced accuracy, precision, recall, Brier score, and false positives at the
  threshold selected only on 2023;
- precision at recall at least 0.60;
- per-year station coverage and missingness drift.

## Success criteria

- Test average-precision gain at least 0.03.
- Test recall remains at least 0.60.
- Precision at recall at least 0.60 improves over the fresh control.
- Control and spatial rows, labels, splits, estimator, and threshold-selection
  policy match exactly.
- Tests prove strict pre-noon filtering, coordinate validation, optional-feed
  behavior, and feature-role isolation.

## Result

Verdict: **fail for model promotion**. Keep the `variant` control as the
default model and retain `spatial` only as an opt-in research role.

| Test metric | Control (`variant`) | Spatial | Difference |
|---|---:|---:|---:|
| average precision | 0.4248 | 0.4592 | +0.0344 |
| balanced accuracy | 0.6955 | 0.6259 | -0.0696 |
| precision | 0.3692 | 0.4272 | +0.0580 |
| recall | 0.6752 | 0.3761 | -0.2991 |
| Brier score | 0.1427 | 0.1387 | -0.0040 |
| precision at recall >= 0.60 | 0.4286 | 0.3923 | -0.0363 |
| false positives | 135 | 59 | -76 |
| false negatives | 38 | 73 | +35 |

The primary average-precision gain clears the preregistered +0.03 target. It
is also positive in both test years: +0.0386 in 2024 and +0.0261 in 2025. The
paired-day bootstrap estimate is +0.0344, with 2.5th, 50th, and 97.5th
percentiles of -0.0129, +0.0350, and +0.0874. The distribution is centered
above zero, so the written falsifier is not triggered, but the interval still
includes zero and does not establish a robust out-of-sample gain.

The operational success criteria fail. At the threshold selected only on 2023,
spatial recall falls from 0.6752 to 0.3761, below the required 0.60. Its best
precision while maintaining recall of at least 0.60 also falls from 0.4286 to
0.3923. The new stations therefore contain useful ranking signal, but this
experiment does not support replacing the current model or its artifact.

The tabular ingestion, location guards, and opt-in feature role are retained so
the signal can be studied without changing the default predictor. Any follow-up
feature or model change needs a new preregistered comparison; changing only the
decision threshold would not address the weaker precision-recall trade-off.

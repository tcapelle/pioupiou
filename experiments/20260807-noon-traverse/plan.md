# 20260807-noon-traverse

**Date:** 2026-08-07
**Question:** Can a simple model using only information available before local noon predict a sustained 10-knot westerly wind at Aix-les-Bains that afternoon?

**Status:** Superseded by `20260807-spatial-stations` after correcting the
PP456 location history. Its metrics must not be used for the current dataset.

## Hypothesis

An L2-regularized logistic regression combining morning PiouPiou wind and
nearby Météo-France observations will improve average precision by at least
0.05 over a calendar-only seasonal logistic baseline on the locked 2024–2025
test period. It was untouched for the original evaluation and is reused here
only to verify the scikit-learn migration.

## Falsifier

The hypothesis is false if `test/average_precision` improves by less than 0.05
over the baseline, or if the variant's validation-frozen operating point has
`test/balanced_accuracy < 0.60`.

## Success criteria

- Test average-precision improvement of at least 0.05 over the baseline.
- Test balanced accuracy of at least 0.60 at a threshold selected only on 2023.
- Test recall of at least 0.60, with precision and false alarms reported rather
  than hidden behind overall accuracy.
- No feature uses an observation at or after 12:00 local time.

## Baseline

- **Status:** finished
- **W&B run:** https://wandb.ai/capecape/pioupiou-traverse/runs/4xnotqyg
- **Why this is the right control:** a day-of-year-only logistic model captures
  seasonality without using morning conditions.

## Variant

Use the same logistic implementation, split, regularization search, and
threshold policy, changing only the feature set: calendar plus leakage-free
morning PiouPiou and Météo-France features.

## Metrics

- **Decision:** `test/average_precision`, `test/balanced_accuracy`,
  `test/recall`, `test/precision`, `test/brier_score`
- **Health:** `train/log_loss`, `model/max_abs_coefficient`, `data/train_rows`

## Label contract

- Time zone: `Europe/Paris`.
- Prediction cutoff: 12:00 local; feature timestamps must be strictly earlier.
- Target window: from 12:00 (inclusive) to 20:00 (exclusive) local.
- Positive event: at least 30 cumulative qualifying minutes and at least one
  run of 3 consecutive PiouPiou readings, with adjacent gaps no larger than 10
  minutes, where average wind is at least 18.52 km/h (10 kt) and heading is in
  the 225°–315° westerly sector. Each qualifying reading carries time only
  until the next sample, capped at 5 minutes.
- A day with insufficient afternoon coverage is unknown, not negative.
- Duplicate source timestamps are collapsed before daily aggregation.

All label parameters are command-line configurable because a local expert may
prefer a narrower direction sector or longer persistence requirement.

## Design

- Data: one row per usable local day; train 2017–2022, validation 2023, locked
  test 2024–2025.
- Weather: official hourly CHAMBERY-AIX station 73329001, using observations
  through 11:00 local from the Météo-France public climatology dataset.
- Model: scikit-learn 1.6.1 `Pipeline` with a `ColumnTransformer`, median
  `SimpleImputer`, value-only `StandardScaler`, one `MissingIndicator` per
  input, and unweighted L2 `LogisticRegression(solver="lbfgs")`. All fitted
  preprocessing is learned on training data only.
- Regularization: select from `0.01, 0.1, 1, 10` on expanding-year folds inside
  2017–2022.
- Operating threshold: maximize validation balanced accuracy on 2023, then
  freeze it before evaluating 2024–2025.
- As-of scoring: prepare an unlabeled row from data strictly before noon;
  reject PiouPiou older than 30 minutes or hourly weather older than 90
  minutes rather than imputing an unavailable feed.
- Runtime: CPU, expected under one minute after data preparation.

Commands (validated before launch):

```text
python3 train_traverse_model.py --dataset artifacts/traverse_daily.csv --role baseline --wandb-name exp-20260807-noon-traverse-baseline --wandb-tags exp/20260807-noon-traverse,baseline
python3 train_traverse_model.py --dataset artifacts/traverse_daily.csv --role variant --wandb-name exp-20260807-noon-traverse-variant --wandb-tags exp/20260807-noon-traverse,variant
```

## Smoke

- Passing sklearn baseline smoke: https://wandb.ai/capecape/pioupiou-traverse/runs/yumwnbl6
- Passing sklearn variant smoke: https://wandb.ai/capecape/pioupiou-traverse/runs/kjpjqdc5
- Both exit 0 and contain every required decision and health metric. On the
  smoke test year, average precision is 0.254 for baseline and 0.357 for the
  variant. Their W&B configuration correctly records the smoke-only
  `L2 = 1.0` candidate and 2018–2019 internal selection folds.

## Runs

- Baseline: https://wandb.ai/capecape/pioupiou-traverse/runs/4xnotqyg
- Variant: https://wandb.ai/capecape/pioupiou-traverse/runs/zdiyvhms
- Comparison report: https://wandb.ai/capecape/pioupiou-traverse/reports/Noon-Traverse-sklearn-model-%E2%80%94-verified-result--VmlldzoxNzY4NDk1Mg==
- Dataset artifact: `pioupiou-traverse-daily:v2`
- Exact source artifact: `pioupiou-traverse-source-sklearn-v2:v1`
- Model SHA-256: baseline
  `984b2ad0af7cb0c0e7f36c62bed4fec42b9f99af01212b62e5c0d54ad77757e9`;
  variant
  `c065fa1a94a33d18914c56300435c2936e312547a18d9b5c0e4782c8cae4d70a`.

## Result

**Verdict: PASS — suitable as the first retrospective prototype.** Both the
[baseline `4xnotqyg`](https://wandb.ai/capecape/pioupiou-traverse/runs/4xnotqyg)
and [variant `zdiyvhms`](https://wandb.ai/capecape/pioupiou-traverse/runs/zdiyvhms)
finished successfully.

This sklearn pair is a controlled implementation-migration reproduction, not
a new untouched-test experiment: the 2024–2025 test outcomes had already been
inspected with the frozen predecessor artifacts. The migration retained the
selected L2 values (`0.01` baseline, `0.1` variant), changed zero thresholded
test decisions, and had maximum absolute probability differences of
`6.6e-9` and `6.31e-6`, respectively. The table below therefore confirms the
previous result under the requested library implementation.

| Held-out 2024–2025 metric | Baseline | Variant | Change |
|---|---:|---:|---:|
| Average precision | 0.239234 | 0.421860 | **+0.182626** |
| Balanced accuracy | 0.564651 | 0.696530 | **+0.131879** |
| Recall | 0.625000 | 0.673611 | +0.048611 |
| Precision | 0.238095 | 0.373077 | +0.134982 |
| Brier score (lower is better) | 0.157570 | 0.142461 | **-0.015109** |
| False positives | 288 | 163 | **-125** |

The paired-day bootstrap 95% interval for the average-precision gain is
approximately 0.108–0.255. The variant used the threshold selected on 2023
(`0.253539`; baseline `0.179534`). On the 725 locked test days with prevalence
0.198621, its confusion counts were 97 TP, 163 FP, 418 TN, and 47 FN. Thus
62.7% of its 260 positive alerts were false alarms despite the substantial
gain.

Year-level behavior remained useful but recall softened in 2025:

| Year | Variant AP | Balanced accuracy | Recall | Precision | Brier | FP |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 0.448560 | 0.698877 | 0.708333 | 0.359155 | 0.140345 | 91 |
| 2025 | 0.406818 | 0.694444 | 0.638889 | 0.389831 | 0.144607 | 72 |

The AP gain comfortably exceeds 0.05, balanced accuracy is at least 0.60, and
test recall is at least 0.60. The hypothesis is supported and neither falsifier
condition is triggered.

Fresh W&B configuration comparison shows only the intended `role` and
`features` differences. Dataset path, label, 2,071/364/725 chronological split,
L2 candidates, entrypoint, environment, and training policy match. The baseline
selected `L2 = 0.01` and the variant selected `L2 = 0.1`. Both runs log the same
checksummed dataset and complete source artifact, and each stores its
checksummed model and exact environment. Local artifacts reproduce W&B
summaries, the paired comparison binds the final model hashes, historical and
prepared-row inference agree exactly, and all 24 tests pass.

This establishes value for the combined morning feature set, not the separate
contributions of PiouPiou and Météo-France. It remains retrospective and
single-location evidence; archived weather may include quality-control
revisions unavailable at the original noon. Accept the variant as the
prototype, but evaluate it prospectively and select the operational alert
threshold from an explicit missed-event versus false-alarm cost before live
deployment. A PiouPiou-only versus weather-only ablation is the most useful
next experiment.

# Recent weather changes and spatial wind

## Question and frozen protocol

Can recent heating/cooling and spatial wind evolution improve the existing
three-hour warning objective without changing the estimator or data sources?
This protocol is recorded before inspecting candidate results.

- Keep the existing wind-rule labels and exact 2017–2025 dataset rows.
- Use the 94-feature temperature-only auxiliary baseline.
- Keep the seven-leaf, 200-iteration histogram booster, L2=10, and existing
  equal-per-day anticipation weights. Do not search hyperparameters.
- Train on earlier years and predict each year from 2020 through 2025.
- Compare baseline, thermal changes, wind snapshots, wind snapshots plus
  changes, and thermal plus wind snapshots/changes.
- Primary comparison: pooled out-of-fold three-hour event-day average
  precision, paired whole-day bootstrap intervals, and individual-year AP.
- Also compare event coverage at a common 20% false-alert-day ceiling. These
  thresholds are descriptive points on the historical ROC curves, not
  independently validated operational thresholds.
- Report 09:00, 12:00 and 15:00 AP for a remaining same-day event among rows
  still before onset; these are distinct from the three-hour warning metric.
- Do not evaluate 2026 again, fit a deployment bundle, or publish.

## Features and time boundary

Use the existing quality/location-filtered Météo-France hourly caches. For an
issue at 12:00, use the 11:00 observation; for 12:30, use 12:00. One- and
three-hour changes compare exact hourly timestamps, never adjacent rows in the
half-hour prediction table. Exclude observations before 06:00 local time,
matching the baseline's weather window. Missing slots or invalid measurements
remain missing, with imputation fitted separately inside each training fold.
No new feature changes row membership.

Thermal block (17 columns):

- One- and three-hour temperature changes at the four existing stations.
- Changes over those intervals in Belley–airport, Novalaise–airport, and
  Belley–Mont-du-Chat temperature contrasts, using simultaneous readings.
- Latest hourly global radiation at the airport and its one- and three-hour
  changes, converted from J/cm² to mean W/m².

Wind snapshot block (10 columns):

- Eastward and northward velocity at the airport, Belley and Mont du Chat,
  computed from meteorological speed/direction pairs at the same timestamp.
- Belley–airport and Mont-du-Chat–airport differences in those components.

Wind change block (12 columns):

- One- and three-hour changes in both velocity components at those three sites.

These are feature hypotheses, not direct measurements of east-side heating or
proof of the Traverse mechanism. Observation timestamps are available in the
archives; historical publication delays are not, as in the existing baseline.
The same historical folds have informed earlier research decisions, so a gain
here still needs confirmation on new chronological observations.

## Reproduction

```bash
uv run --frozen python -m unittest tests.test_weather_dynamics -v
uv run --frozen python -m scripts.weather_dynamics_ablation
```

The script uses local caches only. Generated features, out-of-fold predictions,
source hashes and reports are written under ignored
`artifacts/weather_dynamics/`.

## Result

The completed run retained 33,237 training-period rows across 1,279 dates and
191 events. Evaluation contains 862 out-of-fold dates and 132 events. The
baseline reproduced the existing pooled AP exactly (`0.2330779021881845`).

| Variant | Features | Pooled 3 h AP | Paired ΔAP, 95% interval | Event coverage | False-alert days |
|---|---:|---:|---|---:|---:|
| Baseline | 94 | 0.2331 | — | 34.8% | 20.0% |
| Thermal changes | 111 | 0.2403 | +0.0073 [-0.0052, +0.0198] | 34.1% | 19.7% |
| Wind snapshots | 104 | 0.2254 | -0.0076 [-0.0229, +0.0121] | 34.8% | 20.0% |
| Wind snapshots + changes | 116 | 0.2273 | -0.0058 [-0.0222, +0.0111] | 33.3% | 19.6% |
| Thermal + wind snapshots/changes | 133 | 0.2284 | -0.0047 [-0.0224, +0.0138] | 34.1% | 19.6% |

Coverage is the best historical event coverage at a false-alert-day ceiling
of 20%; this is not the live model's threshold. Discrete scores mean the
achieved false-alert rates can differ slightly below that ceiling. The 2,000
paired bootstrap samples resample whole days, condition on the fitted folds,
and do not account for serial weather dependence or past research selection.

Thermal changes improved AP in four of six years, but lost in 2021 and 2022.
Mean yearly AP was 0.2517 versus the baseline's 0.2484. The combined model's
mean yearly AP was 0.2521 despite worse pooled AP; neither aggregation shows
a large or consistent improvement. Fixed issue-time results are also small:

| Variant | 09:00 AP | 12:00 AP | 15:00 AP |
|---|---:|---:|---:|
| Baseline | 0.2085 | 0.2410 | 0.2090 |
| Thermal changes | 0.2077 | 0.2429 | 0.2085 |
| Wind snapshots | 0.2147 | 0.2454 | 0.2164 |
| Wind snapshots + changes | 0.2124 | 0.2455 | 0.2114 |
| Thermal + wind snapshots/changes | 0.2142 | 0.2424 | 0.2126 |

These are remaining-event AP values, not three-hour warning AP. The 09:00,
12:00 and 15:00 slices contain 858/131, 858/132 and 832/102 dates/events,
respectively. At later issue times the population excludes already-started
events.

The new columns are populated: wind snapshots are present on 97.7–100% of
rows, one-hour changes on roughly 90–92%, and three-hour changes on roughly
75–77%. The morning cutoff intentionally leaves early changes missing.
Thus an entirely empty or largely absent feature block does not explain the
result.

## Interpretation and next decision

This experiment does not establish an accuracy gain. The modest thermal AP
increase has an interval crossing zero and does not improve event coverage
at the common false-alert ceiling. More elaborate processing of these
particular observations has not supplied the missing predictive information.
This does not rule out other temporal representations or estimator choices.

The next useful decision is between independent label review and measuring
the missing east–west thermal setup. Label review should distinguish classic
thermal Traverse, other sustained westerlies, and threshold near misses;
it should not relabel examples simply because the model disagrees. A new
eastern-side data source would require checking historical coverage, local
exposure/elevation, and availability at issue time before collection.

No 2026 evaluation or deployment model was produced. The feature calculations
passed the focused cutoff/gap/wind-direction/unit regression test and all 15
existing observation-feature tests.

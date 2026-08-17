# Quartz-inspired pre-noon NWP enrichment

## Question

Does adding Quartz-style gridded weather features improve the existing noon
Traverse classifier without changing its estimator or evaluation protocol?

## Method

- Reference: `variant` (`cal_`, `piou_`, and primary `mf_` blocks).
- Candidate: `nwp` (reference features plus `nwp_`).
- Estimator: unchanged scikit-learn L2 logistic regression pipeline.
- Split: train 2017–2022, validation 2023, test 2024–2025.
- Selection: unchanged expanding-year average-precision L2 search and
  validation balanced-accuracy threshold.
- Comparison: paired bootstrap of held-out local days, 5,000 replicates,
  seed 20260807.

The source is ECMWF IFS through Open-Meteo's historical forecast API, matching
one of the NWP sources supported by Quartz. The request contains the 15
Quartz-style fields available consistently in this archive: temperature,
relative humidity, dew point, precipitation, surface pressure, sea-level
pressure, total/low/mid/high cloud cover, 10 m wind speed and direction, and
daylight state and direct/diffuse radiation.

Every daily feature uses model values with local valid times in
`[06:00, 12:00)`. Afternoon values are never aggregated, even though they are
present in the API response. This is stricter than simply using historical or
reanalysis weather for the target window, which would leak post-cutoff
information into a noon forecast.

## Reproduction

```bash
uv run python -m scripts.build_dataset
uv run python -m scripts.train \
  --dataset artifacts/traverse_daily.csv \
  --role variant \
  --wandb-name quartz-nwp-reference \
  --wandb-mode disabled
uv run python -m scripts.train \
  --dataset artifacts/traverse_daily.csv \
  --role nwp \
  --wandb-name quartz-nwp-candidate \
  --wandb-mode disabled
uv run python -m scripts.evaluate \
  --reference artifacts/traverse_model_variant.joblib \
  --candidate artifacts/traverse_model_nwp.joblib \
  --output artifacts/traverse_comparison_nwp.json \
  --replicates 5000
```

Dataset SHA-256:
`dcdf7f2b8541b06ee96dda579c6fc8cf46bb088b74dc1c2ee0d2eeaf73034065`.

## Results

| Metric | `variant` | `nwp` | Change |
|---|---:|---:|---:|
| Test average precision | 0.4248 | 0.4168 | -0.0080 |
| Test ROC AUC | 0.7337 | 0.7198 | -0.0139 |
| Test balanced accuracy | 0.6955 | 0.6589 | -0.0366 |
| Test precision | 0.3692 | 0.3086 | -0.0606 |
| Test recall | 0.6752 | 0.7094 | +0.0342 |
| Test Brier score | 0.1427 | 0.1499 | +0.0072 (worse) |

The paired average-precision difference (`nwp - variant`) is -0.0079949. The
bootstrap 95% interval is [-0.0451590, +0.0328389], with median -0.0074094.

The candidate had complete pre-noon NWP core fields for 3,005 of the 3,149
usable rows; the 144 missing rows are all before the modeling window (2016).
The trained candidate uses 110 input columns versus 65 for the reference.

## Decision

Do not promote `nwp` over `variant`. The added block improves recall but not the
primary ranking metric, precision, calibration, or balanced accuracy. Retain
the role and its reproducible inference contract for follow-up work with
forecast horizons or a nonlinear model, neither of which is introduced in
this experiment.

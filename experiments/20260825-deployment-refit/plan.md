# Completed-season deployment refit

## Question

Can current Traverse accuracy improve without adding predictors that are
unavailable at inference time?

The published estimator was selected and fitted on 2017--2022, validated on
2023, and evaluated on 2024--2025. Keeping that old fit in production leaves
three completed and more recent seasons unused.

## Frozen decision

The deployment candidate was specified before scoring 2026:

- preserve the 94-feature observation contract, seven-leaf histogram booster,
  `L2=10`, fit weighting, and validation-selected threshold `0.2756703495`;
- preserve the original 2017--2022/2023/2024--2025 pipeline and metrics as the
  chronological model-selection and evaluation evidence; and
- refit only the serialized deployment pipeline on all completed evaluation
  years, 2017--2025.

The partial 2026 season remains later data. It did not select a feature,
hyperparameter, threshold, or training cutoff.

## Result

The rebuilt dataset contains 37,478 pre-onset rows through 2026-08-17. Its 2026
slice contains 10 positive event days and 77 negative days.

| Fit used for 2026 | 3 h AP | 3 h ROC AUC | Alert >=3 h | False-alert days | Median warning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen through 2022 | 0.2255 | 0.6494 | 30.0% (3/10) | 15.6% | 667 min |
| Deployment refit through 2025 | **0.2674** | **0.6584** | **50.0% (5/10)** | **15.6%** | 366 min |

The refit raises three-hour AP by 0.0419 (18.6% relative), alerts two additional
events, and does not add a false-alert day. This is a small ten-event audit, so
it supports deploying the fresher fit but not changing the established model
architecture or target.

The training command now computes metrics with the original frozen evaluation
pipeline, then refits the artifact pipeline on 2017--2025. Artifact metadata
records both the evaluation split and `deployment_fit_years`; later-year
metrics are prefixed `deployment_later_` to keep the distinction explicit.

## Rejected inference-safe experiments

Validation-only experiments did not justify adding complexity:

- extra westerly-component and exact-rule Windbird summaries did not improve
  expanding-year AP or event coverage at the false-alert budget;
- pressure and wind blocks from live-confirmed Grenoble, Lyon, and Annecy
  stations degraded 2023 validation;
- extending Windbird summaries to midnight reduced 2023 three-hour AP from
  0.3274 to 0.2503; and
- a hard-negative refit intended to match the daily maximum-alert objective
  degraded both expanding-year and 2023 ranking.

Generated station caches and experimental tables remain ignored. No rejected
feature was added to the model contract.

## Live availability

All retained inputs remain available through the Windbird archive, local
calendar, and the four Météo-France hourly station packages. Live verification
also found and fixed two adapter mismatches: application credentials use the
`apikey` header, and numeric department `1` is used for the archived `01`
station prefix.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260825-completed-season-refit \
  --wandb-mode disabled
uv run --frozen python -m unittest discover -v
```

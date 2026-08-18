# Hot-season Traverse target with multiple weather stations

## Question

Predict the probability that the arbitrary Traverse wind criterion occurs on a
hot day from May through September. A winter or cool-season westerly is not a
Traverse under this definition.

## Target

A date is eligible from May 1 through September 30. Its label is positive when
both conditions hold:

1. the quality-controlled CHAMBERY-AIX daily maximum temperature is strictly
   greater than 25°C; and
2. `[12:00, 20:00)` lake observations contain wind of at least 18.52 km/h from
   225°–315° for at least 30 cumulative minutes and three consecutive samples,
   with consecutive gaps no larger than 10 minutes.

Cooler eligible days are negative. Out-of-season dates, missing daily maximum
temperature, and target-window coverage below 75% are excluded. The daily
maximum is target construction data and is never supplied as a future model
feature.

The temperature threshold follows Météo-France's climatological definition of
a hot day. Daily and hourly observations come from the official
[daily](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes)
and
[hourly](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires)
archives.

## Weather inputs

| Station | ID | Elevation | Inputs |
| --- | --- | ---: | --- |
| CHAMBERY-AIX | `73329001` | 235 m | Existing full hourly weather block |
| BELLEY | `01034004` | 330 m | Latest, mean, and morning temperature change |
| NOVALAISE | `73191001` | 460 m | Latest, mean, and morning temperature change |
| MONT DU CHAT | `73051001` | 1,496 m | Latest, mean, and morning temperature change |

The dataset also contains BELLEY−CHAMBERY, NOVALAISE−CHAMBERY, and
BELLEY−MONT DU CHAT contrasts for those three temperature summaries. Secondary
feed freshness is based on temperature because NOVALAISE does not publish the
full humidity/wind bundle. CHAMBERY-AIX retains the stricter temperature,
humidity, speed, and direction freshness contract.

## Reproduction

```bash
uv sync
uv run python -m scripts.build_timestep_dataset --offline
uv run python -m scripts.train \
  --wandb-name may-september-hot-traverse-temperature-stations \
  --wandb-mode disabled
uv run python -m unittest discover -v
```

The frozen dataset has 35,828 timestep rows across 1,332 dates. At the daily
level it contains 848 hot days, 319 wind-event days, and 177 positive Traverse
days. The timestep positive rate is 0.1328. Splits remain chronological:
2017–2022 train, 2023 validation, and 2024–2025 test.

## Result

The selected model uses `L2=10` (`C=0.1`) and a validation-selected threshold
of `0.134107`.

| Split | AP | ROC AUC | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| validation (2023) | 0.4057 | 0.8024 | 0.0964 | 0.3191 |
| test (2024–2025) | 0.3982 | 0.7763 | 0.1026 | 0.3381 |

At the selected threshold, held-out recall is 0.7006, precision is 0.2691, and
balanced accuracy is 0.6973. Average precision by issue time is:

| Time | AP |
| --- | ---: |
| 06:30 | 0.2765 |
| 12:00 | 0.4235 |
| 14:00 | 0.4317 |
| 16:00 | 0.5156 |
| 18:00 | 0.6074 |
| 19:30 | 0.7042 |

## Multi-station ablation

A post-hoc ablation retrained the same estimator using only the original
CHAMBERY-AIX weather block. It performed better on validation and was effectively
tied on test:

| Weather design | Validation AP | Test AP |
| --- | ---: | ---: |
| CHAMBERY-AIX only | 0.4461 | 0.3992 |
| Added temperature stations and contrasts | 0.4057 | 0.3982 |

The added station block is therefore not an established accuracy improvement.
It remains in this requested experimental model so the thermal hypothesis can
be inspected, but it should not be described as validated. Do not remove or
reshape it based on the 2024–2025 result; a later decision needs new validation
data or independently reviewed Traverse labels.

## Interpretation

This model estimates a joint outcome: that the day becomes hot and that the
arbitrary wind criterion occurs. Before the daily maximum is observed, current
temperature and thermal contrasts are predictors of the hot-day part of the
target. Later predictions become a nowcast of both the heat and wind criteria.

The new metrics are not directly comparable with the earlier wind-only model:
the target population and positive definition both changed.

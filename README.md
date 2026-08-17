# Same-day Traverse predictor

This repository contains one research model: the probability that a qualifying
Traverse occurs during today's fixed `[12:00, 20:00)` local window, given all
observations available at the requested time.

The model can score any minute from 06:30 through 19:59. It is a forecast before
the event window and increasingly a nowcast afterward; late scores may include
direct evidence that a Traverse has already occurred.

## Model

The estimator is an L2-regularized scikit-learn logistic regression with median
imputation, standardization, and explicit missingness indicators. Its features
are limited to:

- seasonal and issue-time sine/cosine encodings;
- recent PiouPiou wind speed, gust, direction, trend, and freshness summaries;
- progress toward today's Traverse criterion before the requested time; and
- recent observations from the primary CHAMBERY-AIX Météo-France station.

All observations at or after the requested time are excluded. The chronological
split is 2017–2022 training, 2023 validation, and 2024–2025 test. The selected
model uses `L2=1` (`C=1`) and a validation-selected decision threshold of
`0.24938`.

Held-out average precision was 0.3232 at 06:30, 0.4434 at noon, 0.5526 at 14:00,
0.6937 at 16:00, 0.8450 at 18:00, and 0.9909 at 19:30. See the
[experiment record](experiments/20260816-same-day-timesteps/plan.md) for the
complete interpretation and metrics.

## Build, train, and test

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python -m scripts.build_timestep_dataset --offline
uv run python -m scripts.train \
  --wandb-name same-day-traverse \
  --wandb-mode disabled
uv run python -m unittest discover -v
```

Remove `--offline` on the first dataset build to fetch and filter the required
Météo-France archive. Generated datasets, weather caches, model artifacts, and
W&B runs are ignored by Git.

## Predict

Score an existing dataset row:

```bash
uv run python -m scripts.predict --date 2025-07-21 --time 12:00
```

Prepare and score an arbitrary minute:

```bash
uv run python -m scripts.prepare_timestep \
  --date 2025-07-21 \
  --time 13:17 \
  --offline-weather \
  --output artifacts/timestep_features.csv

uv run python -m scripts.predict \
  --features artifacts/timestep_features.csv
```

The JSON response includes the probability, thresholded decision, and largest
signed contributions to the model logit. Contributions describe this model;
they are not causal explanations.

`joblib` uses pickle semantics. Load only trusted model bundles. A downloaded
artifact can be checked before deserialization with `--model-sha256`.

## Future webcam experiments

The standalone Grand Port webcam pipeline is retained for future image-model
research. It inventories timestamped frames, creates a reviewable selection,
and downloads only that frozen selection; it is not part of the current model.

```bash
uv run python -m scripts.image_pipeline inventory \
  --start-date 2024-01-01 --end-date 2025-08-15

uv run python -m scripts.image_pipeline plan \
  --start-date 2024-01-01 --end-date 2025-08-15 \
  --from-time 08:00 --until-time 12:00 \
  --stride 2 --quality mini

uv run python -m scripts.image_pipeline download
```

Inventory metadata, selections, and downloaded images remain ignored under
`artifacts/webcam_grandport/`. Confirm bulk collection and machine-learning
rights with the image rightsholders before downloading a corpus.

## Data and label

PiouPiou observations with missing coordinates or coordinates more than 1 km
from the lake station are rejected. Météo-France observations must match the
configured CHAMBERY-AIX station location and use accepted quality codes 0, 1,
or 9.

A day is positive when observations in `[12:00, 20:00)` show wind of at least
18.52 km/h from 225°–315° for at least 30 cumulative minutes and three
consecutive samples, with gaps no larger than 10 minutes. Days below 75% target
window coverage have an unknown label and are excluded.

This is a leakage-free retrospective prototype, not a production warning
service. Do not tune feature choices or thresholds on the held-out years.

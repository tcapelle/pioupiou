# Same-day Traverse predictor

This repository contains one research model bundle. Its primary output is an
advance warning that a qualifying Traverse will begin later in today's fixed
`[12:00, 20:00)` local
window on a hot day from May through September, given observations available
at the requested time. Dates outside May–September are outside the model
population.

The model can score any minute from 06:30 through 19:59, but it is trained as an
anticipation model rather than a nowcast. On a retrospectively positive day,
event onset is the start of the first qualifying run that sustains 30 minutes.
Rows at or after onset are excluded. Earlier positive rows receive weight
`min(minutes_before_onset / 180, 1)`, so a prediction at least three hours early
has full training value and one five minutes early has very little.

## Model

The advance-risk estimator is a deliberately shallow scikit-learn histogram
gradient booster with median imputation (seven leaves, 200 iterations). Its
features are limited to:

- seasonal and issue-time sine/cosine encodings;
- recent PiouPiou wind speed, gust, direction, trend, and freshness summaries;
- recent observations from the primary CHAMBERY-AIX Météo-France station;
- temperature summaries from BELLEY, NOVALAISE, and MONT DU CHAT; and
- explicit lowland, lake-side, and ridge temperature contrasts.

All observations at or after the requested time are excluded. Model selection
and reporting use the chronological split 2017–2022 training, 2023 validation,
and 2024–2025 test.
L2 regularization is selected by event-day average precision at least three
hours before onset. The threshold balances three-hour event alerts against
false-alert days on validation data. The selected model uses `L2=10` and
threshold `0.27567`. Fit weights preserve the lead-time value within each
day, then normalize every day to equal total weight so days with more available
checkpoints do not dominate training. After those choices and the test metrics
are frozen, the serialized deployment pipeline is refit on every completed
evaluation year, 2017–2025, without changing its architecture or threshold.

The bundle also contains one ordered onset companion. It classifies each
pre-onset row into onset within 1 hour, 1–2 hours, 2–3 hours, or later/no onset,
then sums those mutually exclusive estimates into cumulative 1-, 2-, and
3-hour probabilities. This guarantees `P(1 h) ≤ P(2 h) ≤ P(3 h)`, unlike three
independently fitted binary models. It uses the same 94 inference-time features
and uniform within-day, equal-per-day fit weights.
The estimates have not received a separate probability-calibration fit.

On the chronological 2024–2025 test years, three-hour event-day AP is `0.190`, ROC
AUC is `0.717`, and false-alert days are `14.8%`; `34.8%` of events are alerted
at least three hours early. On the later partial 2026 season, the deployment
refit reaches AP `0.267`, AUC `0.658`, and `50.0%` (5/10) three-hour event
coverage with `15.6%` false-alert days. The otherwise identical model trained
only through 2022 reaches AP `0.226`, alerts 3/10 events, and has the same
false-alert rate. See the
[deployment-refit experiment](experiments/20260825-deployment-refit/plan.md)
and [ordered-onset audit](experiments/20260829-ordered-onset-deployment/plan.md),
as well as the earlier [frozen-model record](experiments/20260818-hot-season-anticipation/plan.md),
for the complete interpretation, plus the earlier
[target](experiments/20260818-hot-season-multistation/plan.md) and
[anticipation](experiments/20260817-lead-time-anticipation/plan.md) records for
the experiment history.

## Build, train, and test

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --wandb-name same-day-traverse \
  --wandb-mode disabled
uv run --frozen python -m unittest discover -v
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

The JSON response includes the probability, thresholded decision, and a
separate `onset_evidence` value: the latest westerly wind component as a
fraction of the 18.52 km/h target, clipped to `[0, 1]`. This physical evidence
index is not a second probability. `GET /predict` also returns ordered onset
estimates under `onset_within_probabilities`. Once a sustained qualifying run
is observable, the response status becomes `onset_observed` and those
future-onset estimates are zeroed.

With `METEOFRANCE_TOKEN` set, serve the current live prediction at
`GET /predict`:

```bash
TRAVERSE_MODEL=artifacts/traverse_model.joblib \
  uv run uvicorn api:app
```

The same server exposes a historical dashboard at
`http://127.0.0.1:8000/dashboard`. It shows wind-rule Traverse candidates by
year and a shared 12:00–21:00 timeline with wind in knots, flow-direction
needles, advance probability, physical onset evidence, the event-onset boundary,
ordered 1-, 2-, and 3-hour onset estimates, CHAMBERY-AIX temperature, and either
cloud cover or solar radiation when cloud cover is unavailable. The 2026 view
also compares event-day AP at three-, two-,
and one-hour lead and summarizes early-alert rate, false-alert days, median
warning time, and the share of events whose onset evidence rises over the final
six hours. Rule matches are candidates, not independently confirmed Traverse
observations. Dashboard data is built from
the local archives on first load and then cached in memory.

## Publish the live GitHub Pages prediction

The static page in `docs/` reads the latest result from
`live-data/current_prediction.json`. Enable GitHub Pages from the `main`
branch's `/docs` directory once. Publish one result from the inference computer:

```bash
uv run --frozen python -m scripts.publish_live
```

The publisher force-replaces the `live-data` branch with one root commit, so
five-minute updates do not accumulate Git history. Keep it running with:

```bash
TRAVERSE_MODEL=artifacts/traverse_model.joblib \
  uv run --frozen python -m scripts.publish_live --watch --interval 300
```

This requires Git push access and `METEOFRANCE_TOKEN`. The page marks results
stale after 15 minutes. Use `--dry-run` to inspect the public JSON without
pushing.

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

Refresh the current Windbird archive month by month (the API rejects overly
large ranges):

```bash
uv run --frozen python -m scripts.fetch_piou_archive --station-id 2176 --year 2026

# Rebuild the feature table; the chronological split still trains on 2017–2022.
uv run --frozen python -m scripts.build_timestep_dataset
uv run --frozen python -m scripts.train \
  --wandb-name 2026-holdout-dashboard \
  --wandb-mode disabled
uv run --frozen python -m scripts.predict_dataset \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
```

The monthly Windbird CSVs are source data and are versioned. Weather caches,
joined features, model bundles, and prediction outputs remain ignored under
`pioudata/.weather_cache/` and `artifacts/` and can be reproduced with the
commands above.

PiouPiou observations with missing coordinates or coordinates more than 1 km
from the lake station are rejected. Météo-France observations must match the
configured CHAMBERY-AIX station location and use accepted quality codes 0, 1,
or 9.

A day is in scope only from May 1 through September 30. It is positive when the
official CHAMBERY-AIX daily maximum temperature is strictly greater than 25°C
and observations in `[12:00, 20:00)` show wind of at least 18.52 km/h from
225°–315° for one sustained run of at least 30 minutes, with qualifying-sample
gaps no larger than 10 minutes. Cooler in-season days are negative. Dates
outside the season and days below 75% target-window coverage have an unknown
label and are excluded from training.

This is a leakage-free retrospective prototype, not a production warning
service. Do not tune feature choices or thresholds on the held-out years.

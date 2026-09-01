# Same-day Traverse predictor

A research model that estimates whether a qualifying Traverse will begin later
today at Lac du Bourget. The public prediction is available at
<https://tcapelle.github.io/pioupiou/>.

A Traverse is defined here as wind from 225°–315° at 18.52 km/h or more,
sustained for at least 30 minutes between 12:00 and 20:00 local time. The model
operates from May through September and can issue predictions from 06:30 until
19:59.

This is a leakage-free retrospective prototype, not a production warning
service.

## How it works

The model combines recent wind from the Grand Port PiouPiou/Windbird station
with Météo-France observations around the lake:

- wind and weather from CHAMBERY-AIX;
- temperature from BELLEY, NOVALAISE, and MONT DU CHAT; and
- temperature contrasts between the lowland, lake-side, and ridge stations.

Only observations strictly before the prediction time are used. A shallow
histogram gradient booster estimates the chance of a Traverse later that day.
A companion model estimates cumulative onset probabilities within one, two,
and three hours.

The model trains on 2017–2025 and uses partial 2026 as a held-out test set. See
[Data.md](Data.md) for the data and label definitions and the
[latest experiment](experiments/20260831-train-through-2025-test-2026/plan.md)
for methodology and results.

<img src="docs/data-sources-map.svg" alt="Observation stations used around Lac du Bourget" width="760">

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
```

Build the dataset, train the model, and run the tests:

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --wandb-name same-day-traverse \
  --wandb-mode disabled
uv run --frozen python -m unittest discover -v
```

Remove `--offline` on the first build to download the required Météo-France
archives. Generated datasets, caches, model artifacts, and W&B runs are ignored
by Git.

## Predict

Score an existing dataset row:

```bash
uv run --frozen python -m scripts.predict \
  --date 2025-07-21 \
  --time 12:00
```

Or prepare features for any minute before scoring them:

```bash
uv run --frozen python -m scripts.prepare_timestep \
  --date 2025-07-21 \
  --time 13:17 \
  --offline-weather \
  --output artifacts/timestep_features.csv

uv run --frozen python -m scripts.predict \
  --features artifacts/timestep_features.csv
```

The JSON response contains the same-day probability, alert decision, physical
onset evidence, and ordered one-, two-, and three-hour onset probabilities.

## Serve locally

Set `METEOFRANCE_TOKEN`, then run:

```bash
TRAVERSE_MODEL=artifacts/traverse_model.joblib \
  uv run --frozen uvicorn api:app
```

- Current prediction: <http://127.0.0.1:8000/predict>
- Historical dashboard: <http://127.0.0.1:8000/dashboard>

`deployment_model.json` pins the live model to an immutable GitHub Release
artifact and SHA-256 hash. The API and publisher download that bundle when the
local model is missing or stale. Because model bundles use `joblib`/pickle,
only load trusted artifacts.

## Publish the live prediction

GitHub Pages serves `main/docs`. A separate `live-data` branch contains the
latest prediction and historical daily files.

Check one publication locally:

```bash
uv run --frozen python -m scripts.publish_live --dry-run
```

Then publish every five minutes:

```bash
uv run --frozen python -m scripts.publish_live --watch --interval 300
```

To rebuild and seed the 2026 date browser:

```bash
uv run --frozen python -m scripts.predict_dataset \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
uv run --frozen python -m scripts.build_web_history
uv run --frozen python -m scripts.publish_live \
  --seed-history artifacts/web_history
```

The publisher replaces the `live-data` branch with a single root commit, so
prediction updates do not accumulate Git history.

## Refresh source data

Fetch the Windbird archive month by month, then rebuild the dataset:

```bash
uv run --frozen python -m scripts.fetch_piou_archive \
  --station-id 2176 \
  --year 2026
uv run --frozen python -m scripts.build_timestep_dataset
```

Wind data: © [OpenWindMap contributors](https://www.openwindmap.org/).
Map shoreline: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
ODbL.

# Same-day Traverse predictor

This repository contains one research model bundle. Its primary output is an
advance warning that a qualifying Traverse will begin later in today's fixed
`[12:00, 20:00)` local
window from May through September, given observations available
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

All observations at or after the requested time are excluded. The model trains
on 2017–2025 and is evaluated on the evolving partial-2026 validation set. Within the training period,
expanding-year folds for 2020–2025 produce out-of-fold predictions used to
select L2 regularization and the alert threshold. The selected model uses
`L2=10` and threshold `0.30509`. Fit weights preserve the lead-time value within each
day, then normalize every day to equal total weight so days with more available
checkpoints do not dominate training. The serialized pipeline is fitted once on
all 2017–2025 rows; 2026 is never used for fitting or selection.

The bundle also contains one ordered onset companion. It classifies each
pre-onset row into onset within 1 hour, 1–2 hours, 2–3 hours, or later/no onset,
then sums those mutually exclusive estimates into cumulative 1-, 2-, and
3-hour probabilities. This guarantees `P(1 h) ≤ P(2 h) ≤ P(3 h)`, unlike three
independently fitted binary models. It uses the same 94 inference-time features
and uniform within-day, equal-per-day fit weights.
The estimates have not received a separate probability-calibration fit.

Across the rolling historical folds, three-hour event-day AP is `0.233`, ROC AUC
is `0.624`, event coverage is `50.0%`, and false-alert days are `27.3%`. On the
partial 2026 validation audit through August 30, AP is `0.175`, AUC is `0.560`, `27.3%`
(3/11) of events are alerted at least three hours early, and `20.2%` (18/89) of
negative days receive an alert. See the
[simplified split experiment](experiments/20260831-train-through-2025-test-2026/plan.md)
and the earlier experiment records for the research history.

## Data sources

The model observes the lake directly and samples the air mass around it,
especially to the west where a Traverse arrives from. The map shows every
station currently used by the feature pipeline; it is not a map of candidate
stations.

<img src="docs/data-sources-map.svg" alt="Map of the five observation stations used around Lac du Bourget" width="760">

The table inventories the quantities retained by the current feature pipeline.
An em dash means that the quantity is not used from that station, even if it
exists in the upstream archive.

| Station and source | Wind | Temperature | Moisture | Pressure | Rain | Cloud and visibility | Sun and radiation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Grand Port, Aix-les-Bains** (`2176`)<br>[OpenWindMap](https://www.openwindmap.org/) PiouPiou/Windbird | Minimum, average, maximum/gust (`km/h`); direction (`°`); morning mean, trend, westerly component, and freshness | — | — | — | — | — | — |
| **CHAMBERY-AIX airport** (`73329001`)<br>Météo-France · [public station view](https://www.meteociel.fr/temps-reel/obs_villes.php?code2=73329001) | 10 m speed (`m/s`); direction (`°`); latest, morning mean/change, and mean westerly component | Air temperature and dew point (`°C`): latest, morning mean/change; dew-point depression | Relative humidity (`%`): latest and morning mean/change | Mean-sea-level and surface pressure (`hPa`): latest and morning mean/change | Hourly precipitation accumulated since 06:00 (`mm`) | Visibility (`m`): latest and morning mean/change; mean cloud cover (`oktas`) and sky-obscured fraction | Sunshine duration (`min`); global, direct, and diffuse radiation totals (`J/cm²`) and means (`W/m²`) |
| **BELLEY** (`01034004`)<br>Météo-France | — | Air temperature (`°C`): latest, morning mean, and morning change | — | — | — | — | — |
| **NOVALAISE** (`73191001`)<br>Météo-France | — | Air temperature (`°C`): latest, morning mean, and morning change | — | — | — | — | — |
| **MONT DU CHAT** (`73051001`, ridge)<br>Météo-France | — | Air temperature (`°C`): latest, morning mean, and morning change | — | — | — | — | — |

The three auxiliary stations are deliberately temperature-only. Their readings
form Belley-minus-airport, Novalaise-minus-airport, and
Belley-minus-Mont-du-Chat contrasts, representing the lowland, lake-side, and
ridge temperature structure west of the lake. CHAMBERY-AIX is therefore the
same airport station often identified by the WMO code `07491`; it is not an
additional sixth source.

For retrospective builds, the repository keeps monthly OpenWindMap CSVs and
downloads the required Météo-France hourly open-data archives by
department. Live inference reads the same four Météo-France sites from the
public observation API. At every issue time, only observations strictly before
that time are exposed to the model. Weather summaries start at 06:00 local time
and their latest core reading must be no more than 90 minutes old; Grand Port
wind summaries start at 06:00 and must be no more than 30 minutes old.

Wind data attribution: © contributors of the
[OpenWindMap wind network](https://www.openwindmap.org/). Map shoreline:
© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL.

## Build, train, and test

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --wandb-name same-day-traverse \
  --wandb-mode disabled
uv run --frozen python eval_model.py
uv run --frozen python -m unittest discover -v
```

Remove `--offline` on the first dataset build to fetch and filter the required
Météo-France archive. Generated datasets, weather caches, model artifacts, and
W&B runs are ignored by Git.

Model performance should be measured with the canonical evaluation entrypoint
described below rather than with one-off scoring code.

## Evaluate model performance

`eval_model.py` is the default entrypoint for analysing every current and future
model. It loads an already fitted model, refuses to score 2026 if that year was
part of the fit, and evaluates the model without fitting or selecting any
parameter. Run it after training:

```bash
uv run --frozen python eval_model.py
```

The default inputs are `artifacts/traverse_model.joblib` and
`artifacts/traverse_timestep.csv`. The complete report is printed and written to
`artifacts/eval_2026.json`. Alternate artifacts can be evaluated explicitly:

```bash
uv run --frozen python eval_model.py \
  --model artifacts/candidate_model.joblib \
  --dataset artifacts/traverse_timestep.csv \
  --output artifacts/candidate_eval_2026.json
```

The report provides several complementary views of predictive performance:

| Report block | Interpretation |
| --- | --- |
| `event_day_confusion.at_least_3h` | Primary operational result: detected Traverse events, false-negative events, false-alert days, and correctly quiet days at least three hours ahead. |
| `event_day_confusion.any_lead` | The same event-day counts when an alert at any pre-onset checkpoint is accepted. |
| `metrics` | Event-day AP, ROC AUC, coverage, false-alert rate, and warning time, plus diagnostic row-level classification and probability metrics. |
| `onset_horizon_metrics` | Row-level diagnostics for onset within 1, 2, and 3 hours. These probabilities are not separately calibrated. |
| `events` | One auditable record per positive day, including onset, maximum three-hour score, detection, and achieved warning lead. |

Event-day results are the primary comparison because checkpoints within a day
are correlated and operationally describe a single forecasting decision.
Row-level TP/FP/FN/TN counts are useful diagnostics, but must not replace the
event-day result when deciding whether one model is better.

The current snapshot contains 100 validation days through **2026-08-30**: 11
Traverse event days and 89 negative days. The deployed model detects 3/11 events
at least three hours ahead, misses 8/11, raises a false alert on 18/89 negative
days, and remains correctly quiet on 71/89. Its three-hour event-day AP is
`0.175` and ROC AUC is `0.560`.

### The 2026 validation set is still evolving

The 2026 set is a **forward validation/audit set, not a permanently frozen test
set**. New Windbird and Météo-France observations are appended as they become
available. Summer 2026 is almost over, but September remains part of the model's
May–September Traverse season, so the number of days, events, misses, and false
alerts can still change. Re-running an unchanged model on a later snapshot can
therefore produce different metrics.

For a fair comparison between future models:

1. Evaluate every candidate with `eval_model.py` on the exact same dataset
   snapshot.
2. Compare the `dataset_sha256`, covered-through date, row count, and event-day
   count before comparing metrics.
3. Record both the model and dataset SHA-256 values with reported results.
4. If fresh 2026 data has arrived, either re-evaluate every candidate or preserve
   the older dataset as a named frozen comparison snapshot.
5. Do not train on 2026 or select a threshold from it and then describe its
   scores as held-out performance. Once used to choose among models, it is
   validation evidence; genuinely untouched evidence must come from later data.

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

The public page at <https://tcapelle.github.io/pioupiou/> reads the latest result
from `live-data/current_prediction.json`. GitHub Pages is configured to serve
`main/docs`, while the inference computer updates only the data branch.

`deployment_model.json` pins the live model to an immutable GitHub Release URL
and SHA-256. Before every API or publisher prediction, the inference process
checks `artifacts/traverse_model.joblib` and atomically downloads the pinned
bundle when the local file is missing or stale. Pulling and restarting the
merged code therefore selects the model declared by that commit rather than an
untracked artifact left on the inference computer.

### Run the publisher in a terminal

Open a fresh terminal in this repository and confirm the local prerequisites:

```bash
cd /path/to/pioupiou
uv sync --frozen
test -f artifacts/traverse_model.joblib
test -n "$METEOFRANCE_TOKEN"
git ls-remote origin HEAD
```

The two `test` commands produce no output when successful. Set
`METEOFRANCE_TOKEN` in that terminal first if the second command fails. Inspect
one result locally without pushing:

```bash
uv run --frozen python -m scripts.publish_live --dry-run
```

Then start the publisher and leave the terminal open:

```bash
uv run --frozen python -m scripts.publish_live --watch --interval 300
```

It publishes immediately, then every five minutes. Stop it with `Ctrl-C`; rerun
the same command after restarting the terminal or computer. The publisher
force-replaces the `live-data` branch with one root commit, so updates do not
accumulate Git history. The page marks a result stale after 15 minutes.

### Rebuild and seed the date browser

The public date browser contains retrospectively reconstructed 2026 predictions,
the PiouPiou observation available strictly before each issue time, and a small
audit summary derived from the prediction metadata. These are labeled separately
from contemporaneously published live results. Rebuild and seed them with:

```bash
uv run --frozen python -m scripts.predict_dataset \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
uv run --frozen python -m scripts.build_web_history
uv run --frozen python -m scripts.publish_live \
  --seed-history artifacts/web_history
```

After the one-time seed, the normal `--watch --interval 300` command preserves
the daily files, appends or replaces the current five-minute point, rebuilds the
date index, and amends the branch's single root commit. Restart an already
running publisher after updating the repository so it uses this archive-aware
behavior.

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

# Rebuild the feature table; training uses 2017–2025 and testing uses 2026.
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

A day is in scope only from May 1 through September 30. It is positive when
observations in `[12:00, 20:00)` show wind of at least 18.52 km/h from
225°–315° for one sustained run of at least 30 minutes, with qualifying-sample
gaps no larger than 10 minutes. Temperature remains a predictor but is not part
of the target. Dates outside the season and days below 75% target-window
coverage have an unknown label and are excluded from training.

This is a leakage-free retrospective prototype, not a production warning
service. Do not tune feature choices or thresholds on the held-out years.

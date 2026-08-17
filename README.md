# Traverse predictor

This repository contains reproducible research models for several related
questions. The original daily model asks:

> At 12:00 local time in Aix-les-Bains, will a sustained westerly wind above
> 10 knots occur before 20:00?

It is a leakage-free, backtested prototype rather than a production weather
warning service. The model is deliberately simple: a scikit-learn
`ColumnTransformer` and L2-regularized `LogisticRegression`, using only
information timestamped before noon. The default uses PiouPiou and official
station observations; an opt-in candidate also uses pre-noon ECMWF model
fields from Open-Meteo.

## Same-day probability at arbitrary times

The new `same_day` role answers a time-independent question: given everything
observed so far, what is the probability that a qualifying Traverse occurs at
some point during today's fixed `[12:00, 20:00)` window? It trains on a dense
30-minute grid from 06:30 through 19:30, uses smooth clock features, and can
prepare and score any minute inside that window—for example 13:17. Inputs
stamped at or after the requested time are always excluded.

Held-out average precision rises naturally as evidence accumulates: 0.3232 at
06:30, 0.4434 at noon, 0.5526 at 14:00, 0.6937 at 16:00, 0.8450 at 18:00, and
0.9909 at 19:30. Late scores include direct evidence that the event already
occurred and should be interpreted as same-day nowcasting, not advance forecast
skill. See the [timestep experiment record](experiments/20260816-same-day-timesteps/plan.md).

## Current experiment

The Quartz-inspired NWP enrichment is implemented and evaluated without
changing the estimator. A new `nwp` role adds 45 daily features derived from
15 ECMWF IFS variables: temperature, humidity, dew point, precipitation,
surface and sea-level pressure, total/low/mid/high cloud cover, wind speed and
direction, daylight state, and direct/diffuse radiation. Values are summarized
only over `[06:00, 12:00)` local time; reconstructed afternoon weather is
deliberately excluded because it would leak information unavailable at the
noon decision.

On the unchanged 592-day 2024–2025 test slice, the NWP candidate reached
0.4168 average precision versus 0.4248 for `variant` (difference -0.0080;
paired bootstrap 95% interval [-0.0452, +0.0328]). Recall increased from
0.6752 to 0.7094, but precision fell from 0.3692 to 0.3086 and balanced
accuracy fell from 0.6955 to 0.6589. It is therefore retained as a research
role and is not promoted over the default. See the
[experiment record](experiments/20260816-quartz-nwp/plan.md).

## Previous spatial-station experiment

PP456 stopped reporting valid lake coordinates on 16 August 2025, but its
archive later resumed with null coordinates. The builder now rejects every
PiouPiou observation with missing coordinates or more than 1 km from the lake
site before creating either features or labels. The corrected enriched table
contains 3,149 usable days and 601 positive days; the chronological test slice
contains 592 days and 117 positives from 2024 through 15 August 2025.

The controlled station ablation is complete. Adding the four optional stations
improved test average precision from 0.4248 to 0.4592 (+0.0344), with a paired
bootstrap 95% interval of [-0.0129, +0.0874]. However, recall at the threshold
selected on 2023 fell from 0.6752 to 0.3761, and precision while maintaining at
least 0.60 recall fell from 0.4286 to 0.3923. The spatial model therefore
**failed promotion criteria**: the `variant` lake/airport model remains the
default, while `spatial` is retained as an opt-in research role.

The full design, run IDs, metrics, and decision are in the
[experiment record](experiments/20260807-spatial-stations/plan.md) and the
[W&B comparison report](https://wandb.ai/capecape/pioupiou-traverse/reports/Traverse-spatial-station-ablation-%E2%80%94-2026-08-08--VmlldzoxNzY4ODM5Mg==).
This is a retrospective ablation on an already inspected test period, not an
unbiased estimate of performance on future seasons.

## Webcam image dataset

The Grand Port panoramic webcam is a promising spatial source for the next
Traverse experiment. Its public player exposes a timestamped archive from
September 2018 at a roughly 20-30-minute cadence. The metadata audit found all
five candidate pre-noon frames for 2,317 of 2,381 archive-overlapping label
days (97.31%), including 588 of 592 dates in the 2024-2025 test period.

`scripts.image_pipeline` is a standalone, standard-library-only collector. It
keeps discovery, selection, and transfer separate so that no images need to be
downloaded before the exact dataset has been reviewed.

Inventory a period without downloading images:

```bash
uv run python -m scripts.image_pipeline inventory \
  --start-date 2024-01-01 --end-date 2025-08-15
```

Create an exact, editable selection. The time window is half-open and
`--stride 2` keeps every second capture within each day:

```bash
uv run python -m scripts.image_pipeline plan \
  --start-date 2024-01-01 --end-date 2025-08-15 \
  --from-time 08:00 --until-time 12:00 \
  --stride 2 --quality mini
```

Inspect or edit `artifacts/webcam_grandport/selection.csv`. After confirming
bulk collection and machine-learning rights with the image rightsholders,
download exactly those frozen rows:

```bash
uv run python -m scripts.image_pipeline download
```

Monthly API responses are cached as JSON and existing image files are skipped
when a download is restarted. Generated metadata and images stay under
`artifacts/webcam_grandport/` and are ignored by Git.

## What counts as a Traverse

The default label is positive when all of the following hold:

- local window `[12:00, 20:00)` in `Europe/Paris`;
- PiouPiou average wind speed is at least 18.52 km/h (10 knots);
- wind heading is in the broad west sector 225°–315°;
- qualifying samples cover at least 30 cumulative minutes; and
- at least three qualifying readings are consecutive, with gaps no larger
  than 10 minutes.

A day needs at least 75% observed afternoon coverage. Otherwise its outcome is
unknown rather than silently treated as "no Traverse". The thresholds are CLI
options in `pioupiou.data.daily`.

This is a wind-pattern proxy. It cannot prove that an event had thermal rather
than synoptic or storm-driven origins. A local Lake Bourget description says
the classical Traverse rotates from southwest to northwest, occurs late in the
day, and can last one to four hours; that motivated the broad sector and
persistence guard: <https://www.lac-du-bourget.com/a-la-decouverte-de-la-traverse-le-mysterieux-vent-du-lac-du-bourget/>.

## Data enrichment

`scripts.build_dataset` turns the 1,186,599 raw wind rows into one
leakage-free row per usable local day. It:

1. reads only `pioudata/YYYY-MM.csv` files and ignores the overlapping aggregate
   CSVs;
2. removes 556 duplicate timestamps, rejects 55,004 missing/off-site location
   rows, and converts UTC to `Europe/Paris`;
3. derives pre-noon wind summaries and the afternoon label;
4. discovers current department 01, 73, and 74 resources through data.gouv.fr;
5. streams each large Météo-France archive once and caches only the five chosen
   stations; and
6. adds the existing airport weather block plus compact ridge, west, north, and
   south observations through 11:00 local; and
7. caches yearly ECMWF IFS responses from Open-Meteo and creates the optional
   Quartz-inspired pre-noon `nwp_` block.

The station manifest is deliberately short:

| Role | Station | ID | Distance from PP456 |
|---|---|---:|---:|
| airport control | CHAMBERY-AIX | 73329001 | 6.73 km |
| west ridge | MONT DU CHAT | 73051001 | 6.65 km |
| west lowland | BELLEY | 01034004 | 16.93 km |
| north synoptic | MEYTHET | 74182001 | 30.00 km |
| south valley | MONTMELIAN | 73171002 | 26.47 km |

CHAMBERY-AIX is 6.73 km from the PiouPiou and is at nearly the same elevation.
Each official observation must remain within 1 km of its configured station
site, so a reused or moved station ID cannot silently enter the table.
Values with Météo-France quality code `2` (doubtful) are treated as missing;
codes `0`, `1`, and `9` are accepted. Missing features are imputed from the
training years only and get an explicit missingness indicator. Imputation,
standardization, missingness indicators, fitting, prediction, and metrics all
use scikit-learn APIs; models are serialized with `joblib`.
Filtered weather caches are checksummed, and the Piou source files are
Git-tracked. A prediction is refused when the latest PiouPiou reading is more
than 30 minutes old or the latest hourly weather
reading with valid temperature, humidity, wind speed, and wind direction is
more than 90 minutes old. Prepared rows are also bound to the exact model,
feature contract, and station IDs before they can be scored.

The first build transfers roughly 390 MB of compressed public archives. The
filtered station cache is about 14 MB; later builds use that verified cache.

## Reproduce

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python -m scripts.build_dataset
uv run python -m scripts.build_timestep_dataset --offline

uv run python -m scripts.train \
  --dataset artifacts/traverse_daily.csv \
  --role variant \
  --wandb-name exp-20260807-spatial-stations-control \
  --wandb-tags exp/20260807-spatial-stations,control

uv run python -m scripts.train \
  --dataset artifacts/traverse_daily.csv \
  --role spatial \
  --wandb-name exp-20260807-spatial-stations-spatial \
  --wandb-tags exp/20260807-spatial-stations,spatial

uv run python -m scripts.train \
  --dataset artifacts/traverse_daily.csv \
  --role nwp \
  --wandb-name exp-20260816-quartz-nwp \
  --wandb-tags exp/20260816-quartz-nwp,candidate

uv run python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --role same_day \
  --wandb-name exp-20260816-same-day-timesteps \
  --wandb-tags exp/20260816-same-day-timesteps,candidate

uv run python -m scripts.evaluate \
  --reference artifacts/traverse_model_variant.joblib \
  --candidate artifacts/traverse_model_spatial.joblib
uv run python -m scripts.evaluate \
  --reference artifacts/traverse_model_variant.joblib \
  --candidate artifacts/traverse_model_nwp.joblib \
  --output artifacts/traverse_comparison_nwp.json \
  --replicates 5000
uv run python -m unittest discover -v
```

If `WANDB_API_KEY` is absent, training records an offline run. Pass
`--wandb-mode disabled` to skip tracking intentionally.

Score a historical row from the dense dataset:

```bash
uv run python -m scripts.predict --date 2025-07-21 --time 12:00
```

Prepare an unlabeled row at any minute, then score it:

```bash
# Prepare and score a same-day probability at any minute in the model window.
uv run python -m scripts.prepare_timestep \
  --date 2025-07-21 --time 13:17 \
  --offline-weather \
  --output artifacts/timestep_features_2025-07-21_1317.csv
uv run python -m scripts.predict \
  --model artifacts/traverse_model_same_day.joblib \
  --features artifacts/timestep_features_2025-07-21_1317.csv

# The older noon-only roles still use their original preparer.
uv run python -m scripts.prepare_noon \
  --date 2025-07-21 \
  --model artifacts/traverse_model_nwp.joblib \
  --piou-input-dir pioudata \
  --offline-weather \
  --output artifacts/noon_features_2025-07-21.csv
uv run python -m scripts.predict \
  --model artifacts/traverse_model_nwp.joblib \
  --features artifacts/noon_features_2025-07-21.csv
```

Without `--piou-input-dir`, preparation fetches the morning directly from the
public PiouPiou archive API. Without `--offline-weather`, it checks the current
data.gouv.fr Météo-France resource metadata; `--refresh-weather` forces a new
station-cache download. The command never needs an afternoon label and refuses
to produce a row before that date's noon cutoff.

The JSON response includes the probability, the thresholded decision, and the
largest signed contributions to the model logit. Contributions describe this
model; they are not causal explanations.

`joblib` uses pickle semantics. Load only model bundles produced by this
project from a trusted location. For a downloaded artifact, pass a trusted
out-of-band digest with
`uv run python -m scripts.predict --model-sha256 ...`; the digest is checked
against one immutable byte snapshot before deserialization and scoring.

## Code layout

- `pioupiou/data/`: source loading and daily/timestep dataset construction.
- `pioupiou/feature_eng/`: leakage-safe Open-Meteo feature aggregation.
- `pioupiou/inference/`: preprocessing, model fitting, metrics, serialization,
  and scoring utilities.
- `scripts/`: dataset builds, image collection, training, evaluation, feature
  preparation, and prediction entry points.
- `experiments/20260807-noon-traverse/plan.md`: hypothesis, falsifier, run IDs,
  and review record.
- `experiments/20260807-spatial-stations/plan.md`: corrected multi-station
  ablation, W&B runs, metrics, and promotion decision.
- `experiments/20260816-quartz-nwp/plan.md`: Quartz-inspired NWP methodology,
  evaluation, and non-promotion decision.
- `experiments/20260816-same-day-timesteps/plan.md`: dense-timestep same-day
  probability model and full-day score curve.
- `tests/`: unit tests for time boundaries, label persistence, weather units,
  preprocessing, model persistence, feed contracts, and metrics.

Generated datasets, caches, model bundles, and W&B local run files are ignored
by git. They remain in `artifacts/`, `pioudata/.weather_cache/`, and `wandb/`.

## Public sources and licences

- PiouPiou/OpenWindMap attribution is preserved from each source CSV preamble:
  `(c) contributors of the OpenWindMap wind network` and
  <http://developers.pioupiou.fr/data-licensing>.
- Live morning wind is available through the documented
  [PiouPiou archive API](https://developers.pioupiou.fr/api/archive/).
- Météo-France hourly climatology resources are discovered through the
  [official data.gouv.fr dataset](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires)
  and are published under Licence Ouverte 2.0.
- The official Météo-France field and unit dictionary is
  <https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/HOR/H_descriptif_champs.csv>.
- The NWP candidate follows the variable methodology in
  <https://github.com/openclimatefix/open-source-quartz-solar-forecast> and
  uses Open-Meteo's documented historical forecast archive:
  <https://open-meteo.com/en/docs/historical-forecast-api>.
- Grand Port webcam inventories and selections preserve media IDs, timestamps,
  and source URLs. Public archive/download availability is not treated as a
  reusable dataset licence; confirm bulk collection and ML rights with Skaping
  and/or Grand Lac before downloading the full archive.

## Important limitation for live use

The public climatology bulk files are not a guaranteed real-time feed. The
as-of-noon command therefore fails closed if the current weather observation is
missing or stale. A reliable automatic daily alert still requires credentials
and an adapter for Météo-France's operational observation API with the same
measurement semantics. Historical climatology values may also include quality
control or revisions that were not available at the original noon cutoff. Do
not silently replace an unavailable weather feed with medians in production.

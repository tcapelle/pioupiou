# Noon Traverse predictor

This repository now contains a reproducible research model that answers:

> At 12:00 local time in Aix-les-Bains, will a sustained westerly wind above
> 10 knots occur before 20:00?

It is a leakage-free, backtested prototype rather than a production weather
warning service. The model is deliberately simple: a scikit-learn
`ColumnTransformer` and L2-regularized `LogisticRegression`, using only
PiouPiou and weather observations timestamped before noon.

## Current experiment

The previous single-station result is superseded. PP456 stopped reporting valid
lake coordinates on 16 August 2025, but its archive later resumed with null
coordinates. Those off-site or unverified rows entered the old evaluation.
The builder now rejects every PiouPiou observation with missing coordinates or
more than 1 km from the lake site before creating either features or labels.

The corrected enriched table contains 3,149 usable days and 601 positive days.
The chronological test slice contains 592 days and 117 positives from 2024
through 15 August 2025. A fresh controlled comparison of the airport-only
weather model against the spatial-station model is specified in
`experiments/20260807-spatial-stations/plan.md`. Until that comparison is run,
the old model metrics and artifacts should not be treated as current results.

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
options in `build_traverse_dataset.py`.

This is a wind-pattern proxy. It cannot prove that an event had thermal rather
than synoptic or storm-driven origins. A local Lake Bourget description says
the classical Traverse rotates from southwest to northwest, occurs late in the
day, and can last one to four hours; that motivated the broad sector and
persistence guard: <https://www.lac-du-bourget.com/a-la-decouverte-de-la-traverse-le-mysterieux-vent-du-lac-du-bourget/>.

## Data enrichment

`build_traverse_dataset.py` turns the 1,186,599 raw wind rows into one
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
   south observations through 11:00 local.

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
uv run python build_traverse_dataset.py

uv run python train_traverse_model.py \
  --dataset artifacts/traverse_daily.csv \
  --role variant \
  --wandb-name exp-20260807-spatial-stations-control \
  --wandb-tags exp/20260807-spatial-stations,control

uv run python train_traverse_model.py \
  --dataset artifacts/traverse_daily.csv \
  --role spatial \
  --wandb-name exp-20260807-spatial-stations-spatial \
  --wandb-tags exp/20260807-spatial-stations,spatial

uv run python compare_traverse_models.py \
  --reference artifacts/traverse_model_variant.joblib \
  --candidate artifacts/traverse_model_spatial.joblib
uv run python -m unittest discover -v
```

If `WANDB_API_KEY` is absent, training records an offline run. Pass
`--wandb-mode disabled` to skip tracking intentionally.

Score an already prepared historical noon row:

```bash
uv run python predict_traverse.py --date 2025-07-21
```

Prepare an unlabeled row exactly as it exists at noon, then score it:

```bash
# Reproducible historical example using the checked local archives.
uv run python prepare_noon_features.py \
  --date 2025-07-21 \
  --piou-input-dir pioudata \
  --offline-weather \
  --output artifacts/noon_features_2025-07-21.csv
uv run python predict_traverse.py \
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
`uv run python predict_traverse.py --model-sha256 ...`; the digest is checked
against one immutable byte snapshot before deserialization and scoring.

## Files

- `build_traverse_dataset.py`: enrichment, label, and daily feature builder.
- `prepare_noon_features.py`: unlabeled as-of-noon row preparation.
- `traverse_model.py`: scikit-learn preprocessing, logistic fit, metrics, and
  `joblib` serialization.
- `train_traverse_model.py`: chronological training/evaluation and W&B logging.
- `compare_traverse_models.py`: reproducible paired-day bootstrap comparison.
- `predict_traverse.py`: inference for a prepared feature row.
- `experiments/20260807-noon-traverse/plan.md`: hypothesis, falsifier, run IDs,
  and review record.
- `experiments/20260807-spatial-stations/plan.md`: corrected multi-station
  ablation plan.
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

## Important limitation for live use

The public climatology bulk files are not a guaranteed real-time feed. The
as-of-noon command therefore fails closed if the current weather observation is
missing or stale. A reliable automatic daily alert still requires credentials
and an adapter for Météo-France's operational observation API with the same
measurement semantics. Historical climatology values may also include quality
control or revisions that were not available at the original noon cutoff. Do
not silently replace an unavailable weather feed with medians in production.

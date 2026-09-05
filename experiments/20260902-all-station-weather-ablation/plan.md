# All-variable weather-station ablation

## Question

Does retaining every available measurement at BELLEY, NOVALAISE, and MONT DU
CHAT improve advance prediction over the existing temperature-only auxiliary
station contract?

This is a predictive feature ablation, not a causal claim. A useful addition
must improve chronological predictions, not merely explain the training data.

## Available fields

The four archives do not have a common measurement schema:

| Station | Added usable quantities |
|---|---|
| BELLEY | Dew point, humidity, rain, wind speed, wind direction |
| NOVALAISE | Intermittent rain |
| MONT DU CHAT | Dew point, humidity, rain, wind speed, wind direction |
| CHAMBERY-AIX | No change; its full weather block was already retained |

Pressure, visibility, cloud, sunshine, and radiation are absent at the three
auxiliary sites. The dataset emits those columns consistently, but training
removes fields that are empty throughout 2017–2025. Auxiliary freshness remains
temperature-based, preserving the exact 37,138 rows and 1,432 dates used by the
94-feature baseline.

## Protocol

- Baseline: the existing 94-feature contract, including auxiliary temperature
  and temperature contrasts.
- Candidates: auxiliary wind only; auxiliary moisture and rain only; all 29
  newly non-empty fields.
- Fit period: 2017–2025.
- Selection: expanding-year predictions for 2020–2025. Each variant selects
  L2 from `{1, 10}` by mean three-hour event-day AP and selects its alert
  threshold from the concatenated rolling predictions.
- Forward test: partial 2026, inspected only after selection.
- Uncertainty: 2,000 paired bootstrap resamples of whole days. The paired AP
  interval measures candidate minus baseline on identical dates.

## Result

| Variant | Features | Rolling 2020–2025 3 h AP | AP difference, paired 95% interval | Partial 2026 3 h AP | AP difference, paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Temperature-only auxiliaries | 94 | 0.2331 | — | 0.1747 | — |
| Add auxiliary wind | 106 | 0.2370 | +0.0039 [-0.0145, +0.0299] | 0.2630 | +0.0884 [+0.0134, +0.2177] |
| Add auxiliary moisture and rain | 111 | 0.2169 | -0.0162 [-0.0368, +0.0038] | 0.1613 | -0.0134 [-0.0407, +0.0013] |
| Add all available auxiliary weather | 123 | 0.2184 | -0.0147 [-0.0406, +0.0132] | 0.2497 | +0.0751 [+0.0044, +0.2057] |

Every variant selected `L2=10`. At its rolling-selected threshold, the
all-variable model raised rolling event coverage from `50.0%` to `65.2%`, but
also raised rolling false-alert days from `27.3%` to `39.7%`. On partial 2026,
coverage rose from 3/11 to 5/11 and false-alert days rose from 18/89 to 26/89.

## Interpretation

Adding all variables is not validated: its historical point estimate is worse
and its paired interval includes both a modest loss and a small gain. The 2026
gain is encouraging but too small and recent a sample to override that result.
Moisture and rain show no predictive benefit here. Auxiliary wind is the only
addition worth carrying as a focused candidate because it is historically tied
and better on partial 2026, but its higher false-alert rate still prevents a
clean replacement claim.

The feature pipeline retains all fields for continued research, while the
pinned live 94-feature model remains unchanged until new chronological data or
independently reviewed Traverse labels can resolve the uncertainty.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset \
  --offline \
  --output artifacts/traverse_timestep_all_stations.csv

uv run --frozen python -m scripts.weather_station_ablation \
  --dataset artifacts/traverse_timestep_all_stations.csv \
  --output artifacts/weather_station_ablation.json \
  --bootstrap-samples 2000

uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep_all_stations.csv \
  --metadata artifacts/traverse_timestep_all_stations.metadata.json \
  --output-dir artifacts/all_stations \
  --wandb-name exp-20260902-all-weather-stations \
  --wandb-mode disabled
```

The generated JSON, Markdown report, dataset, and model bundle remain ignored
under `artifacts/`.

# Real-time inference

## Conclusion

All 94 inputs of the current same-day model can be constructed at prediction
time from two observation feeds and the local clock:

| Feature family | Count | Source | Status |
| --- | ---: | --- | --- |
| `cal_` | 4 | Requested local date and time | Available locally |
| `piou_` | 29 | OpenWindMap Windbird 2176 | Live and archive feeds verified |
| `mf_` | 61 | Four Météo-France stations and thermal contrasts | Historical feeds verified; live token required |

The root-level `realtime_inference.py` command implements this path with the
existing feature engineering and trained model. It expects a temporary
Météo-France bearer token in `METEOFRANCE_TOKEN` and does not persist API
responses.

```bash
uv run python realtime_inference.py \
  --model artifacts/traverse_model.joblib
```

The same prediction is available over HTTP at `GET /predict`:

```bash
TRAVERSE_MODEL=artifacts/traverse_model.joblib \
  uv run uvicorn api:app
```

`scripts.prepare_timestep` remains the retrospective path backed by monthly
PiouPiou CSV files and the historical Météo-France cache.

The response separates the boosted model's advance probability from
`onset_evidence`, the observed westerly wind component normalized by the target
speed. The latter is an interpretable event-monitoring index, not a forecast
probability.

## Time and leakage contract

- The model scores any local minute from `06:30` through `19:59` in
  `Europe/Paris`, from May 1 through September 30. Outside that season the API
  returns probability zero without scoring the model.
- Only observations strictly before the requested issue time may enter a
  feature. An API response received later must still be filtered by the
  observation's validity timestamp.
- PiouPiou summaries begin at `06:00` local time and use rolling windows up to
  three hours.
- Météo-France summaries begin at `06:00` local time.
- PiouPiou's newest usable observation must be no more than 30 minutes old.
- CHAMBERY-AIX requires a newest observation with finite temperature, humidity,
  wind speed, and wind direction no more than 90 minutes old. The three
  secondary stations require finite temperature with the same age limit.
- The prediction window remains `[12:00, 20:00)` local time. The model was
  trained only on positive rows before the onset of the first sustained
  30-minute qualifying run; once such wind has begun, live observations should
  be treated as event monitoring. The target also requires the eventual
  CHAMBERY-AIX daily maximum to exceed 25°C.

These boundaries are part of the trained model contract. A live implementation
must not substitute receipt time for observation time or include observations
at the issue timestamp.

## OpenWindMap / Windbird 2176

Station page: <https://www.openwindmap.org/windbird-2176>

API endpoints:

- latest observation and metadata:
  <https://api.pioupiou.fr/v1/live-with-meta/2176>
- recent observations:
  <https://api.pioupiou.fr/v1/archive/2176?start=last-day&stop=now>

The archive endpoint, rather than the latest-observation endpoint alone, is
required to construct rolling summaries. Fetch the last day and retain
observations from `06:00` local time up to, but not including, the issue time.

The feed was checked on 2026-08-17. It was live and returned 287 observations
over the preceding 24 hours, approximately one every five minutes. Its location
was `(45.701761, 5.883394)`, within roughly 10 metres of the historical model
location `(45.701731, 5.883505)`. The station metadata identifies it as the KCB
station installed at Lac du Bourget in April 2026.

The archive supplies every raw value needed by the model:

| API field | Use |
| --- | --- |
| `time` | Observation time, freshness, counts, and trend |
| `latitude`, `longitude` | Reject data when the sensor is not at the lake |
| `wind_speed_avg` | Latest speed, rolling mean/max, and trend |
| `wind_speed_max` | Latest and rolling maximum gust, gust factor |
| `wind_heading` | Direction encoding and west fraction/component |
| `wind_speed_min` | Parsed as part of the observation contract, but not used by the current model |
| `pressure` | Not used; it was `null` when checked |

OpenWindMap reports wind speeds in km/h and headings in degrees, matching the
historical PiouPiou representation used during training. No unit conversion is
needed.

### Derived PiouPiou features

The 29 `piou_` model inputs are all derived from the fields above:

- latest/freshness: `piou_last_age_minutes`, `piou_last_wind_avg_kmh`,
  `piou_last_wind_max_kmh`, `piou_last_heading_sin`, and
  `piou_last_heading_cos`;
- since-06:00 count: `piou_observation_count_morning`;
- for each of `30m`, `1h`, and `3h`: mean and maximum average wind, maximum
  gust, west fraction, mean heading sine/cosine, and mean west component;
- three-hour dynamics: `piou_wind_avg_trend_3h_kmh_per_hour` and
  `piou_gust_factor_3h`.

The exact rolling feature names are:

```text
piou_mean_wind_avg_{30m,1h,3h}_kmh
piou_max_wind_avg_{30m,1h,3h}_kmh
piou_max_wind_gust_{30m,1h,3h}_kmh
piou_west_fraction_{30m,1h,3h}
piou_heading_sin_mean_{30m,1h,3h}
piou_heading_cos_mean_{30m,1h,3h}
piou_west_component_mean_{30m,1h,3h}_kmh
```

## Météo-France observations

The model uses four hourly stations:

| Station | ID | Department | Location | Elevation |
| --- | --- | --- | --- | ---: |
| CHAMBERY-AIX | `73329001` | 73 | `(45.641333, 5.877833)` | 235 m |
| BELLEY | `01034004` | 01 | `(45.769333, 5.688000)` | 330 m |
| NOVALAISE | `73191001` | 73 | `(45.597333, 5.776833)` | 460 m |
| MONT DU CHAT | `73051001` | 73 | `(45.660500, 5.821500)` | 1,496 m |

Météo-France documents its observation API as real time, with a 24-hour
retention window. Hourly observations are normally published about ten minutes
after the round hour. The appropriate feed is the 24-hour hourly package for
departments 01 and 73, filtered to the configured station IDs:

```text
https://public-api.meteofrance.fr/public/DPPaquetObs/v1/paquet/horaire?id-departement=01&format=json
https://public-api.meteofrance.fr/public/DPPaquetObs/v1/paquet/horaire?id-departement=73&format=json
```

The request requires a Météo-France OAuth2 bearer token. The subscribed API's
current Swagger documentation is the authority for the base path and token
flow. Relevant public documentation:

- [API Paquet Observations](https://confluence-meteofrance.atlassian.net/wiki/spaces/OpenDataMeteoFrance/pages/854851588/API%2BPaquet%2BObservations)
- [Ground-observation field specification](https://donneespubliques.meteofrance.fr/client/document/descriptiftechnique_observations_donneespubliques_v2_20250315_403.pdf)

The hourly package is preferable to the six-minute feed because the current
feature set needs dew point and cloud cover, which are part of the documented
hourly schema. Its 24-hour depth covers all observations needed from `06:00`
local time through the issue time.

### Raw-field and unit mapping

The model's historical Météo-France data and the real-time API use different
names and, for several quantities, different units. Normalize before calling
the existing feature engineering:

| Real-time field | Historical field | Conversion | Derived feature group |
| --- | --- | --- | --- |
| `validity_time` | `AAAAMMJJHH` | Parse as UTC | Counts, freshness, latest and morning deltas |
| `t` | `T` | K to °C: `t - 273.15` | Temperature latest/mean/delta |
| `td` | `TD` | K to °C: `td - 273.15` | Dew point latest/mean/delta and dew-point depression |
| `u` | `U` | None; percent | Relative humidity latest/mean/delta |
| `rr1` | `RR1` | None; mm | Morning precipitation total |
| `pmer` | `PMER` | Pa to hPa: `pmer / 100` | Sea-level pressure latest/mean/delta |
| `pres` | `PSTAT` | Pa to hPa: `pres / 100` | Surface pressure latest/mean/delta |
| `vv` | `VV` | None; metres | Visibility latest/mean/delta |
| `n` | `N` | None; oktas | Mean cloud cover and sky-obscured fraction |
| `insolh` | `INS` | None; minutes | Morning sunshine total |
| `ray_glo01` | `GLO` | J/m² to J/cm²: `ray_glo01 / 10000` | Morning global radiation and mean W/m² |
| `ff` | `FF` | None; m/s | Wind speed latest/mean/delta and west component |
| `dd` | `DD` | None; degrees | Latest direction sine/cosine and west component |

The conversion of `ray_glo01` is particularly important. The existing feature
engineering expects hourly radiation in J/cm² and multiplies by `10000 / 3600`
to derive W/m². Feeding the real-time J/m² value without first dividing by
10,000 would make the radiation features wrong by four orders of magnitude.

### Derived Météo-France features

CHAMBERY-AIX retains the 34 original `mf_` inputs:

```text
mf_observation_count_morning
mf_core_observation_count_morning
mf_last_age_minutes

mf_temperature_c_latest
mf_temperature_c_mean
mf_temperature_c_delta_morning
mf_dewpoint_c_latest
mf_dewpoint_c_mean
mf_dewpoint_c_delta_morning
mf_dewpoint_depression_c_latest
mf_relative_humidity_pct_latest
mf_relative_humidity_pct_mean
mf_relative_humidity_pct_delta_morning
mf_pressure_msl_hpa_latest
mf_pressure_msl_hpa_mean
mf_pressure_msl_hpa_delta_morning
mf_surface_pressure_hpa_latest
mf_surface_pressure_hpa_mean
mf_surface_pressure_hpa_delta_morning
mf_visibility_m_latest
mf_visibility_m_mean
mf_visibility_m_delta_morning
mf_wind_speed_10m_ms_latest
mf_wind_speed_10m_ms_mean
mf_wind_speed_10m_ms_delta_morning

mf_precipitation_morning_mm
mf_sunshine_morning_minutes
mf_global_radiation_morning_j_cm2
mf_global_radiation_mean_w_m2
mf_cloud_cover_mean_oktas
mf_sky_obscured_fraction
mf_wind_direction_latest_sin
mf_wind_direction_latest_cos
mf_west_component_mean_ms
```

Each secondary station contributes observation count, temperature-valid count,
age, latest temperature, mean temperature, and morning temperature change,
using prefixes `mf_belley_`, `mf_novalaise_`, and `mf_mont_du_chat_`. Nine
`mf_contrast_` inputs encode latest, mean, and morning-change differences for:

```text
BELLEY - CHAMBERY-AIX
NOVALAISE - CHAMBERY-AIX
BELLEY - MONT DU CHAT
```

Météo-France notes that station-specific unavailable values are returned as
`null`. The model median-imputes individual missing features and includes
missingness indicators, so occasional optional gaps are supported. They do not
override the freshness requirement: a recent observation with finite `t`, `u`,
`ff`, and `dd` is required to score.

The historical training rows were filtered using Météo-France quality codes 0,
1, and 9. The real-time API is documented as raw observations and does not
provide the same quality fields. This is a train/serve difference that should
be measured during shadow operation.

## Calendar features

No external data is needed for the four calendar features:

```text
cal_doy_sin
cal_doy_cos
cal_issue_time_sin
cal_issue_time_cos
```

They are calculated from the requested date and minute in `Europe/Paris`.

## Inference sequence

1. Freeze an issue timestamp in `Europe/Paris` and reject times outside the
   model's `06:30`–`19:59` scoring interval.
2. Download Windbird 2176's recent archive and the Météo-France 24-hour hourly
   package for department 73.
3. Select the configured station/location, normalize units, deduplicate by
   observation timestamp, and discard observations at or after the issue time.
4. Reuse `piou_features`, `weather_features`, and the calendar feature
   functions to construct the row.
5. Enforce the 30-minute PiouPiou and 90-minute Météo-France freshness limits.
6. Bind the row to the feature names stored in the model artifact and score it
   with the existing prediction pipeline.
7. Print the prediction time, latest timestamp and age of each source, model
   checksum, probability, and thresholded result.

No observations need to be persisted to make one prediction, but retaining the
raw responses and produced feature rows is valuable for reproducibility and
sensor-drift analysis. Generated responses and inference logs must remain out
of version control.

## Validation before relying on live scores

### Feed acceptance

Run the real-time path in shadow mode before treating its output as usable:

1. Confirm that station `73329001` is present in every expected hourly package.
2. Measure latency and missingness for each required Météo-France raw field,
   especially the core `t`, `u`, `ff`, and `dd` fields.
3. Verify unit conversions with plausible ranges and a small hand-calculated
   feature row.
4. For captured API responses, compare live-adapter features with features
   produced through the historical representation of the same observations.
5. Exercise stale-feed and cutoff-boundary cases explicitly.

### Replacement Windbird

Windbird 2176 occupies the historical location and exposes the expected data
contract, but it is a replacement instrument installed in 2026. The trained
model learned from the older PiouPiou sensor. Matching coordinates and units do
not establish matching calibration, response, or sampling behaviour.

Collect shadow data before changing the reported model results. Compare at
least:

- observation cadence and outage patterns;
- wind-speed and gust distributions by local hour and season;
- heading distributions and west-wind frequency;
- gust factor and three-hour trend distributions;
- frequency and duration of the model's Traverse criterion;
- prediction calibration once outcomes are available.

If a material shift appears, define a chronological retraining/evaluation
period in advance. Do not tune transformations or thresholds against the
existing held-out 2024–2025 results, and do not rewrite those reported results
without a reproducible run.

## Remaining work

- Automate renewal of the temporary Météo-France bearer token if this becomes
  a scheduled process.
- Shadow the path and complete the sensor-shift checks above.

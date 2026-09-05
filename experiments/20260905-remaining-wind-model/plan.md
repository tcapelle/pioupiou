# Remaining afternoon wind model

## Objective

Forecast a useful wind spell still ahead, updating throughout the morning and
afternoon, including 14:00 and 16:00. Reward useful early predictions while
retaining value for shorter warnings later in the day.

The target window is `[max(12:00, issue_time), evening_end)`. A completed
earlier spell does not make a later target positive. Continuing wind and
second spells remain eligible if enough qualifying future wind is observed.
Target coverage is checked within the remaining window. Labels stay unknown
when coverage is insufficient.

Wind strength, duration, direction and the evening cutoff must be confirmed
before training a model with a changed useful-wind definition. The proposed definition awaiting confirmation is sample-
average wind at least 10 knots for one qualifying 30-minute run, any direction,
ending at 21:00. This experiment retains the existing sampled persistence
semantics; it does not silently introduce a rolling-average target.

## Protocol fixed before model results

- Use the 2017–2025 dates in the existing model-ready dataset as the research
  population, preserving season and site. Rebuild all available checkpoints
  from 06:30 through 20:00 every 30 minutes, including those omitted after
  first onset by the old dataset. A chosen target can exclude checkpoints with
  insufficient time remaining for a full useful spell.
- Use existing local wind and weather archives only. Observations used as
  predictors must be strictly before issue time; future wind only defines the
  target and is never a model feature.
- Preserve the original meaning of wind features: `piou_west_fraction` and
  westerly components remain westerly even if the target accepts any direction.
- Remove retrospective onset censoring and lead-dependent training weights.
  Give each date equal total fit weight, uniformly among its retained rows.
- Keep the existing shallow histogram booster at 200 iterations, seven leaves,
  learning rate 0.05, minimum leaf size 100 and L2=10. No hyperparameter search.
- Fit on earlier years and predict each year 2020–2025. Imputation is learned
  within each fold.
- Compare the new 94-feature model with the same model plus the 39 previously
  specified weather dynamics features. Include the prior historical event
  rate by checkpoint as a simple baseline.
- Retrain the old objective model on earlier years only and score its
  probabilities against the same new labels on the same evaluation rows. This
  is a transfer of the old forecast to the new task; its probabilities were
  trained for a different target and its old training population excluded
  post-onset rows. Do not compare new-task AP directly to the old 0.2331 metric.
- Primary summary: AP averaged equally across 08:00, 10:00, 12:00, 14:00,
  16:00 and 18:00. Report each checkpoint separately, alongside Brier score,
  log loss, observed event rate and average predicted probability.
- Also report each checkpoint among rows whose latest observed wind does not
  yet satisfy the target speed/direction rule. This distinguishes advance
  prediction from easier continuation forecasts. One qualifying current
  reading is not evidence that a full sustained spell has already occurred.
- Use 1,000 paired whole-date bootstrap samples for differences in mean
  checkpoint AP. Preserve a day's rows together across checkpoints. These
  intervals condition on the fitted folds and do not account for weather
  persistence across days or repeated historical research selection.
- Select the dynamics variant only if its paired interval versus the simpler
  new-target model is entirely above zero; otherwise retain the simpler model.
- Fit the selected candidate on 2017–2025 and save a research bundle, feature
  provenance and out-of-fold predictions. Do not reevaluate 2026 or publish.
  An improvement on these reused development folds needs future confirmation.

## Implementation and reproduction

The feature build is independent of the pending useful-wind definition:

```bash
uv run --frozen python -m unittest tests.test_remaining_wind -v
uv run --frozen python -m scripts.remaining_wind features
```

Training requires explicit label arguments, for example **only if the proposed
definition is confirmed**:

```bash
uv run --frozen python -m scripts.remaining_wind train \
  --knots 10 --duration 30 --direction any --end-hour 21
```

Artifacts remain ignored under `artifacts/remaining_wind/`. The new research
bundle has an explicit `remaining_wind_research` model kind and is not a
drop-in replacement for the old Traverse API. The `predict` subcommand scores
saved feature rows; historical fitted predictions are not evaluation results.

## Status

The remaining-window regression test passes: past completed wind is excluded,
a later continuing spell can qualify, and incomplete future coverage remains
unknown. Feature preparation can proceed while the useful-wind definition is
being confirmed.

The completed feature build contains 35,677 rows on the same 1,279 historical
dates. All 94 baseline feature values match on all 33,237 original rows; the
2,440 additional checkpoints include post-onset rows and the new 20:00 slot.
All 17 remaining-wind, weather-dynamics and observation-feature tests pass.

While the broader definition is pending, run a control candidate with the
**existing physical definition** (10 knots, west, 30 minutes, 20:00 cutoff).
This changes only the remaining-window objective and weighting/population;
it does not assume consent to broadening directions or extending the evening.
Its results and bundle are saved separately in
`artifacts/remaining_wind_current_definition/`.

## Completed control result

The control fit has 34,393 rows on 1,279 dates. Historical evaluation has
23,197 rows on 862 dates. All variants are scored on the same remaining-window
labels, so these scores are not comparable to the earlier three-hour event-day
AP statistic.

| Issue time | Old objective AP | Remaining-wind AP | Old Brier | Remaining-wind Brier |
|---|---:|---:|---:|---:|
| 08:00 | 0.2166 | 0.2037 | 0.1337 | 0.1301 |
| 10:00 | 0.2052 | 0.2092 | 0.1346 | 0.1283 |
| 12:00 | 0.2410 | 0.2535 | 0.1310 | 0.1260 |
| 14:00 | 0.2242 | 0.2449 | 0.1164 | 0.1148 |
| 16:00 | 0.2046 | 0.2757 | 0.1041 | 0.0999 |
| 18:00 | 0.1367 | 0.3515 | 0.0908 | 0.0755 |

Mean checkpoint AP rises from **0.2047 to 0.2564**, a gain of 0.0517 with
paired whole-date 95% interval **[+0.0296, +0.0732]**. The historical-rate
baseline scores 0.1327. Mean checkpoint AP improves in every validation year
from 2020 through 2025. Brier score and log loss improve at all six checkpoints.

The gain is mainly in afternoon updates. At 08:00, ranking is slightly worse
despite improved probability error. This does not solve the harder problem
of recognizing late wind very early in the morning.

There is also improvement when the latest wind reading does not satisfy the
target speed/direction rule:

| Issue time | Old AP, not currently qualifying | New AP, not currently qualifying |
|---|---:|---:|
| 08:00 | 0.2212 | 0.2108 |
| 10:00 | 0.2046 | 0.2082 |
| 12:00 | 0.2254 | 0.2366 |
| 14:00 | 0.2214 | 0.2326 |
| 16:00 | 0.1417 | 0.1709 |
| 18:00 | 0.0810 | 0.1218 |

Thus the afternoon gain is not entirely explained by continuing already-windy
conditions, though continuation accounts for an important part of it. This
subset uses a single latest reading, not retrospective knowledge of an onset.

Adding weather dynamics scores 0.2533 mean AP. Its difference from the simpler
new-target model has interval [-0.0094, +0.0042], so the prespecified selection
rule retains the **94-feature remaining-wind model**. Its fitted bundle is
`artifacts/remaining_wind_current_definition/model.joblib`.

The new label, restored post-onset examples, and revised weighting change
together. This experiment cannot attribute the gain to any one of them or
claim a breakthrough in meteorological signal. It establishes a stronger
historical candidate for the updated decision task under the current physical
wind definition. These reused development years are not an independent final
test. No live model replacement or 2026 evaluation was performed.

The saved model was reloaded and successfully scored a historical feature row
through the research prediction command. That fitted prediction is a loading
and inference check, not an accuracy result.

### Reproduce the control

After the feature build above:

```bash
mkdir -p artifacts/remaining_wind_current_definition
cp artifacts/remaining_wind/features.csv artifacts/remaining_wind/features.metadata.json artifacts/remaining_wind_current_definition/
uv run --frozen python -m scripts.remaining_wind train \
  --output-dir artifacts/remaining_wind_current_definition \
  --knots 10 --duration 30 --direction west --end-hour 20
uv run --frozen python -m scripts.remaining_wind predict \
  --output-dir artifacts/remaining_wind_current_definition \
  --date 2017-05-01 --time 16:00
```

Use `oof_predictions.csv` for accuracy analysis. `report.json` contains the
complete fold, calibration and subset metrics plus model/data provenance.

# Sustained hot-season Traverse anticipation

## Question

Can the expanded archive support genuine advance warnings under a Traverse
definition that rejects fragmented or unrelated late-day wind?

The first hot-season definition summed every qualifying interval between noon
and 20:00, then separately required one three-sample run. Manual review found a
counterexample on 2026-07-02: scattered west-northwest fragments summed beyond
30 minutes, while the strongest late wind was northerly. Its longest continuous
qualifying run was only 25 minutes, so it was not a Traverse.

## Revised target and objective

A date is eligible only from May through September and when the CHAMBERY-AIX
daily maximum temperature is available. It is positive when that temperature
is strictly above 25 C and the Windbird records one sustained qualifying run in
`[12:00, 20:00)` lasting at least 30 held minutes. Every observation in that run
must have average wind at least 18.52 km/h from 225--315 degrees; qualifying
samples may be at most 10 minutes apart and each sample holds for at most five
minutes. Eligible cool or non-qualifying days are negative; dates with less
than 75% target-window coverage are excluded.

No arbitrary early-onset cutoff is added. Grand Lac describes the Traverse as
a strong westerly wind following heat, and a local account describes the
classic thermal mechanism as occurring late in the day. Timing alone therefore
does not distinguish a Traverse from unrelated evening wind; sustained west
direction does. See the [Grand Lac environmental analysis](https://grand-lac.fr/fileadmin/ARBORESCENCE/Au_quotidien/Amenagement_et_developpement_du_territoire/Urbanisme/Doc_PLUi_ex_Calb/PIL/2._Analyse_environnementale_PIL_appro.pdf)
and [local Traverse description](https://www.lac-du-bourget.com/a-la-decouverte-de-la-traverse-le-mysterieux-vent-du-lac-du-bourget/).

Event onset is the start of the first run that eventually reaches 30 minutes.
Positive rows at or after that onset are excluded. Earlier positive rows receive
`min(minutes_before_onset / 180, 1)` fit weight; negative rows receive weight 1.
The weights are then normalized within each day so every date contributes equal
total fit weight. All predictor observations remain strictly earlier than their
issue time. The retrospective daily maximum temperature is used only to
construct the target.

The revised dataset has 37,453 rows from 2016-08-09 through 2026-08-16. The
chronological split remains 2017--2022 train, 2023 validation, and 2024--2025
test. The partial 2026 season is a later reporting slice. No 2024--2026 outcome
selected regularization or threshold.

## Estimator improvement

The original linear model's raw probability usually fell toward onset because
it estimates whether an event remains ahead today: the positive population
necessarily shrinks through the afternoon. A time-balanced imminence target
made trajectories rise, but reduced validation ranking and generated too many
false alerts. It was rejected.

A shallow histogram gradient booster was then compared using only expanding
folds inside 2017--2022 and the 2023 validation year. The frozen architecture
uses seven leaves, 200 iterations, learning rate 0.05, at least 100 samples per
leaf, median imputation, and the same equal-day lead weights. Expanding-year
training folds selected `L2=10` by three-hour event-day average precision:

| L2 | Expanding-year 3 h AP |
|---:|---:|
| 1 | 0.2072 |
| 10 | **0.2357** |

The decision threshold `0.27567` was selected on 2023 to maximize event-day
balanced accuracy: the mean of the three-hour alert rate and true-negative day
rate. The model retains all 94 leakage-free features from the four weather
stations and Windbird.

The dashboard separately reports a leakage-free physical onset-evidence index:
the latest observed westerly wind component divided by the 18.52 km/h Traverse
threshold and clipped to `[0, 1]`. It is not blended into alert probability and
is not described as a forecast. It gives the requested increasing pre-event
trace without corrupting the advance-risk objective.

The cached secondary-station archive also contains humidity and wind fields not
used by the 94-feature contract. A final exploratory validation-only audit,
performed after the primary model had already been evaluated, expanded the
inputs with those fields. All regional fields raised expanding-fold AP to
`0.2527` but reduced 2023 AP to `0.2840` and produced `30.9%` validation
false-alert days. A wind-only subset reached `0.2791` on expanding folds but
`0.3176` on 2023, and could recover only `50.0%` of validation events while
holding false-alert days below 15%. Because it did not improve the frozen
validation operating point, the extra feature contract was rejected. Its
held-out scores were not inspected or used to rescue it.

## Results

| Slice | Events | 3 h event-day AP | 3 h ROC AUC | Alert >=3 h | Alert >=2 h | Alert >=1 h | False-alert days | Median warning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 validation | 12 | 0.3274 | 0.7848 | 58.3% | 58.3% | 58.3% | 11.5% | 474 min |
| 2024--2025 test | 23 | 0.1905 | 0.7166 | 34.8% | 39.1% | 39.1% | 14.8% | 508 min |
| 2026 later slice | 10 | 0.2309 | 0.6553 | 30.0% | 30.0% | 30.0% | 14.5% | 667 min |

Threshold-independent event-day AP increases as more recent evidence becomes
available: `0.3274 -> 0.3478 -> 0.3478` at three, two, and one hours on 2023;
`0.1905 -> 0.1939 -> 0.1948` on 2024--2025; and `0.2309 -> 0.2309 -> 0.2313`
on 2026. The gain is modest, so the dashboard shows the exact values rather
than implying that all missed events become predictable near onset.

The target correction removes 2026-07-02 and four other fragmented days from
the 2026 positive set. The frozen boosted model alerts three of the ten
remaining sustained events at least three hours early. Relative to the linear
baseline, 2026 three-hour AP rises from `0.1627` to `0.2309`, AUC from `0.5211`
to `0.6553`, and false-alert days fall from `28.9%` to `14.5%`; event coverage
remains 3/10. On 2024--2025, AP and AUC also improve and false alerts fall, but
three-hour coverage decreases from `43.5%` to `34.8%`. These tradeoffs are kept
visible rather than selecting a more permissive threshold on held-out data.

The onset-evidence trace rises during the final six hours on 11/12 (91.7%) 2023
events, 19/23 (82.6%) 2024--2025 events, and 10/10 2026 events. This measures
observed physical build-up; it does not turn the late wind into an early claim.

The dashboard plots each 2026 advance-probability trajectory, the physical
onset-evidence trace, the frozen validation-selected threshold, and sustained-
run onset. It displays wind and weather from 12:00 through 21:00, while the
target remains fixed at `[12:00, 20:00)`.

## Reproduction

```bash
uv run --frozen python -m scripts.build_timestep_dataset --offline
uv run --frozen python -m scripts.train \
  --dataset artifacts/traverse_timestep.csv \
  --wandb-name exp-20260818-day-normalized-sustained-anticipation \
  --wandb-mode disabled
uv run --frozen python -m scripts.predict_dataset \
  --model artifacts/traverse_model.joblib \
  --dataset artifacts/traverse_timestep.csv \
  --year 2026 \
  --output artifacts/traverse_predictions_2026.csv
uv run --frozen python -m unittest discover -v
```

Generated datasets, weather caches, models, predictions, and run outputs remain
ignored. Dataset and model metadata record hashes for reproducibility.

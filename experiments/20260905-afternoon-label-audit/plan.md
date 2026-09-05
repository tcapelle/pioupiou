# Afternoon wind: label and objective audit

## User objective

Know as early as possible whether it will be windy in the afternoon, while
continuing to update the forecast at later times such as 14:00 and 16:00.
Late starts on summer days are explicitly in scope. This requires defining
useful wind conditions and measuring forecast quality throughout the day.
It does not require an independently diagnosed thermal Traverse unless that
specific wind regime is what the user needs.

## Current definition and implementation

The target is evaluated at Grand Port during `[12:00, 20:00)` Europe/Paris,
May through September. A positive day needs one run of sample-average wind
at least 18.52 km/h (10 knots), from 225–315 degrees, accumulating at least
30 minutes. Every below-threshold or out-of-sector sample resets the run.
Qualifying-sample gaps must be at most 10 minutes; each observation represents
at most five minutes until the next sample or the end of the window. This is
a sampled-wind persistence rule, not a 30-minute rolling average.

At least 75% target-window coverage is required. Unknown days are dropped.
Temperature has no role in the current label, despite legacy hot-day fields
in the metadata. Only the Grand Port station defines the outcome.

Recomputing all 1,279 model-ready historical dates from 2017–2025 exactly
reproduced all saved labels: 191 positive days. No stored-label discrepancy
was found. The audit evaluates definition sensitivity, not confirmed mistakes
or revised training labels. The direction and useful wind threshold remain
questions for the user.

## Definition sensitivity on identical dates

All variants below keep the afternoon window, coverage requirement and
30-minute persistence rule fixed. These choices were not selected for better
model scores. No 2026 evaluation was performed.

| Minimum sample-average wind | Westerly only | Any direction |
|---|---:|---:|
| 8 knots | 358 positive days | 578 positive days |
| 10 knots | 191 positive days | 350 positive days |
| 12 knots | 103 positive days | 195 positive days |

At 10 knots, removing the direction restriction adds 159 afternoons. For
example, 2018-09-24 is currently negative despite nearly 478 minutes in one
run above 10 knots when all directions count. Its time-weighted average over
the observed afternoon is 17.69 knots; target-window coverage is nearly 100%.
The negative label expresses the westerly restriction, not a calm afternoon.

Among current negative days:

- 124 contain at least 30 cumulative minutes of westerly wind above 10 knots.
- 41 contain at least 60 cumulative minutes, spread across shorter runs.
- 84 contain a westerly run lasting at least 20 but less than 30 minutes.

These counts overlap. A brief low-speed or out-of-sector reading can split
an otherwise useful session. For example, 2017-08-19 has 148.75 cumulative
qualifying westerly minutes but a longest run of only 16.15 minutes. It remains
negative even without the direction restriction at 10 knots. This is a useful
case for a human judgement of whether the day was usable, not proof that the
label should be changed.

The 75% coverage rule permits up to two unobserved hours in the eight-hour
window. That weakens the interpretation of negative labels, although only
four negative dates in this model-ready sample have coverage below 90%.

## Late events and the evening cutoff

Of the 191 currently labeled events, 159 (83.2%) start at or after 14:00,
122 (63.9%) at or after 16:00, and 70 (36.6%) at or after 18:00. These are
first qualifying run starts under the existing afternoon rule, not an
independent diagnosis of the meteorological onset.

Keeping the existing westerly 10-knot, 30-minute persistence criterion:

| Target window | Eligible dates | Positive dates | Added vs 12:00–20:00 |
|---|---:|---:|---:|
| 12:00–20:00 | 1,279 | 191 | — |
| 12:00–21:00 | 1,279 | 241 | 50 |
| 12:00–22:00 | 1,279 | 268 | 77 |

All the original dates pass the 75% coverage requirement for both extended
windows, and no existing positives disappear. The additions can include runs
starting before 20:00 that need later observations to reach 30 minutes; they
are not all events beginning after 20:00. The useful evening cutoff is a user
choice, not a parameter to choose by model accuracy.

The outcome should also move forward with the issue time. Applying the same
rule only to the remaining window gives 175 positives after 14:00, 153 after
16:00, and 114 after 18:00, all ending at 20:00. There is one unknown date for
the 16:00–20:00 coverage check; the other slices retain all 1,279 dates.
These counts include qualifying wind that is already underway and continues
for at least 30 observed minutes inside the remaining window, and any later
qualifying run. They are not merely first-onset counts. Consequently, a
positive full-afternoon label cannot be reused unchanged at every later time.

## The early-warning objective is also mismatched

The main evaluation score takes each event day's maximum predicted probability
at least three hours before its retrospective onset. A 06:30 alert and a
15:00 alert receive the same three-hour coverage credit when wind starts at
18:00. Both can supply that day's maximum ranking score. There is no extra
credit in this metric for the much earlier decision.

Training weights are flat once a row is three hours or more before onset.
All positive rows at or after onset are removed, and every day is normalized
to equal total weight. Thus morning rows on shorter positive days receive
more fit weight than morning rows on full negative days. At 09:00, actual
event prevalence is 14.96% in 2017–2025, while its fit-weighted prevalence is
22.35%. These weights encode an alert preference; they do not directly train
an ordinary calibrated probability of a windy afternoon at that issue time.

Existing historical out-of-fold forecasts, scored against the current labels:

| Issue time | Dates | Events | Event rate | Mean prediction | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 06:30 | 857 | 131 | 15.29% | 19.55% | 0.2162 | 0.1349 |
| 08:00 | 857 | 131 | 15.29% | 18.95% | 0.2166 | 0.1337 |
| 09:00 | 858 | 131 | 15.27% | 19.43% | 0.2085 | 0.1349 |
| 10:00 | 857 | 130 | 15.17% | 19.28% | 0.2052 | 0.1346 |
| 12:00 | 858 | 132 | 15.38% | 19.00% | 0.2410 | 0.1310 |
| 14:00 | 835 | 107 | 12.81% | 15.31% | 0.2073 | 0.1118 |
| 16:00 | 818 | 88 | 10.76% | 9.32% | 0.1487 | 0.0954 |
| 18:00 | 780 | 52 | 6.67% | 3.12% | 0.0810 | 0.0634 |

The current model already supports inference at 14:00 and 16:00. The issue is
the target and evaluation, not an absence of afternoon prediction endpoints.
These afternoon scores include only pre-onset event rows; they do not evaluate
remaining useful wind after an earlier event has begun or finished.

Average probability exceeds event frequency in the morning and falls below
it at 16:00 and 18:00. At 18:00, the fit-weighted training event fraction is
2.11%, versus an actual 6.06% at that checkpoint. Short-lead downweighting is
at odds with giving useful late updates. These are aggregate calibration
discrepancies, not a complete reliability analysis or proof that the weights
are their only cause. Populations differ with predictor availability and
post-onset exclusion; differences between times are descriptive.

## Recommended next contract, pending useful-wind choices

Use a continuously updated forecast of useful wind in the remaining
afternoon/evening, including at 14:00, 16:00 and 18:00:

`P(useful wind occurs during [max(12:00, t), evening_end) | observations before t)`

Before noon this is the same fixed-afternoon label at every checkpoint.
After noon, only useful wind still ahead contributes to the label. A past
event alone cannot make a later prediction positive. Continuing wind or a
second qualifying spell should remain eligible rather than removing the day
after its first onset. The exact useful-duration rule still needs agreement.

The user must determine whether direction matters and what wind strength and
duration make an afternoon useful. If short dips are acceptable, compare the
current persistence criterion with a clearly specified rolling-average or
tolerated-dip criterion using human-reviewed examples, not whichever target
produces the highest model score.

Evaluate chronological forecasts separately at 06:30, 08:00, 10:00, 12:00,
14:00, 16:00 and 18:00, including ranking, calibration, and event coverage
at comparable false-alert rates. For a notification policy, report first-alert
times relative to actual qualifying spells and cumulative false-alert days;
select the threshold using only earlier years. Clock-specific scores keep late
updates valuable, while warning-time summaries measure how early useful
decisions become possible. The current daily maximum-score calculation assumes
one constant label per day, so it cannot be reused for the changing remaining-
window label without revising its meaning.

A simple baseline would use the same shallow estimator and all issue-time
checkpoints, with ordinary day-balanced weights and no retrospective first-
onset censoring or short-lead penalty. Build the remaining-window labels from
raw observations rather than just reweighting the existing censored dataset.
This is a proposed next experiment, not a reported accuracy improvement or an
implemented target change.

## Reproduction and artifacts

```bash
uv run --frozen python -m scripts.weather_dynamics_ablation
uv run --frozen python -m scripts.audit_windy_labels
```

The first command supplies the saved baseline out-of-fold predictions and is
unnecessary if those artifacts already exist. The audit itself performs no
training or downloads. It checks raw labels against the saved dataset and
prediction identities against the same historical rows.

Generated `artifacts/windy_label_audit/daily_labels.csv` and `audit.json`
contain per-day comparisons, forecast/weight diagnostics, and source hashes.

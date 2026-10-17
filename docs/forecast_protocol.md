# Economic dataset and Model A protocol — Steps 2 and 3

## Status and scope

This is a **retrospective monthly dataset**, constructed from the existing local
`fredgraph.csv` snapshot and cross-checked against local `GS10.csv`. It is not a
historical real-time information set. Retrieval dates, historical vintages and
release timestamps are not supplied in these inputs. Local consistency checks
do not independently authenticate the observations or establish their vintages.
The original files are preserved byte for byte. No observations are downloaded,
imputed or fabricated. Step 2 prepares the economic dataset. Step 3 implements
Model A and a single-forecast smoke test; a full backtest is not implemented.

## Observation and target definitions

Each row is indexed by a first-of-month `predictor_month` representing calendar
month t. This date is an observation-period label, not an actual forecast origin
or publication date. `target_month` is exactly six calendar months later, even
across year boundaries; the target is looked up by that date rather than by row
position or a fixed number of days.

| Column | Definition | Role / units |
|---|---|---|
| `predictor_month` | Source observation month t | Monthly key, YYYY-MM-01 |
| `target_month` | t + 6 calendar months | Label observation month, YYYY-MM-01 |
| `USREC_t` | Source USREC in month t | Reference only; **not an input feature** |
| `GS10_t` | Source GS10 in month t | Percent |
| `TB3MS_t` | Source TB3MS in month t | Percent |
| `spread_t` | GS10_t minus TB3MS_t | Percentage points; sole Model A feature |
| `target_t` | Source USREC in `target_month` | Binary future recession-status label |

The outcome is recession **status in the sixth future month**, not whether a
recession begins at any time during the intervening six months. FRED defines
USREC as 1 from the month following an NBER peak through the trough month, and
0 otherwise ([FRED USREC notes](https://fred.stlouisfed.org/series/USREC)).
The pipeline uses these supplied labels without redefining recession dates.

## Input validation and exclusions

- Require the documented CSV headers, UTF-8 encoding (optional BOM), valid
  first-of-month dates, ascending order, and a complete monthly calendar within
  each file's first and last observation. Reject duplicates, including identical
  repeated rows; never silently deduplicate or interpolate calendar gaps.
- Interpret blank cells, `.`, `NA` and `N/A` as missing. Reject other nonnumeric
  values, NaN, infinities, and USREC values other than 0 or 1. Signed finite
  yields are permitted; no unsupported economic bounds are imposed.
- Require exact decimal equality and identical non-missing monthly coverage
  for GS10 in the two files. Mismatches stop preparation; the second file never
  fills gaps in the primary file. Agreement does not prove independent origin.
- Exclude a predictor row if either yield is missing. Otherwise exclude it if
  its target month is absent or its future USREC value is missing. Record these
  mutually exclusive reasons, in that order, in the manifest. Missing current
  USREC alone does not invalidate a row because it is not a predictor.
- Keep all eligible months in the snapshots; do not restrict the economic
  dataset to a proposed lyrics period. Use decimal subtraction without scaling
  or rounding the yields. No standardization or feature selection is performed.

`docs/data_manifest.json` records source hashes, schemas, per-series coverage,
missingness, validation outcomes, exclusions, output coverage and output hash.
It is deterministic for the same input bytes, paths and implementation. Unknown
retrieval/vintage/release metadata remain null rather than being guessed.
Re-running replaces only the designated processed CSV and manifest. Validation
failure occurs before those artifacts are written; any pre-existing artifacts
remain from their previous successful run and must not be mistaken for new output.

## Timing limitations and requirements before forecasting

A monthly average yield for month t cannot be assumed known at the start of t.
These files provide neither historical publication times nor revision histories.
Step 3 assumes a forecast cutoff on the **15th calendar day of month t+1**, after
predictor month t ends. There is no business-day adjustment. This is a
retrospective timing convention, not evidence that the source values had been
published by that day. The outcome remains USREC in **t+6**, anchored to predictor
month t, not six months after the day of issuance in t+1.

NBER dates turning points retrospectively and waits for sufficient evidence.
For example, the December 2007 peak was announced on December 1, 2008
([NBER announcement chronology](https://www.nber.org/research/business-cycle-dating/business-cycle-dating-committee-announcements)).
Consequently, an eventual USREC value was not necessarily known in the month
it describes. A six-month training embargo addresses the prediction horizon
but **does not establish historical label availability**. Zero labels, as well
as one labels, must not be assumed final and contemporaneously available.

Before a historical real-time backtest, obtain or document predictor release
and vintage evidence and a defensible label-availability rule based on available
vintages/announcement evidence. Do not invent a fixed NBER announcement lag.
Until then, any later analysis using these snapshots must remain explicitly
retrospective. `USREC_t` must never accidentally enter the spread-only
baseline. Neither final labels nor future source rows may be used as features.

## Implemented Model A and training eligibility

Model A uses `statsmodels.discrete.discrete_model.Probit` with an explicit
intercept and only `spread_t`:

\[
P(\mathrm{USREC}_{t+6}=1) = \Phi(\beta_0 + \beta_1\,\mathrm{spread}_t).
\]

It is unregularized maximum likelihood, using Newton optimization initialized
at zero coefficients. Spread is **not standardized**. It remains in percentage
points; no full-sample scaling or other learned preprocessing occurs.
The intercept is explicitly included as required by the
[statsmodels Probit API](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Probit.html).

For predictor month t, training uses observations i satisfying both:

- i is April 1953 or later (the configured starting month);
- target month i+6 is **no later than t-1**, inclusive.

At least **240 actual eligible months** are required. Eligibility is calculated
using dates, not by removing a fixed number of rows. Missing monthly observations
are never fabricated to reach 240. Missing eligible labels cause an explicit
error; they are not silently dropped. Data at t, later months, and labels after
t-1 cannot enter the fit. Prediction takes only t's spread and does not require
its future target value. `fit_probit` fits an already-selected sample;
`select_training` enforces eligibility and `forecast_one` composes both steps.

For the supplied snapshot:

| Check | Verified value |
|---|---|
| First eligible predictor month | October 1973 |
| Assumed forecast cutoff | November 15, 1973 |
| Forecast target month | April 1974 |
| Training predictor months | April 1953–March 1973 |
| Training target months | October 1953–September 1973 |
| Training sample | 240: 203 non-recession labels, 37 recession labels |
| September 1973 forecast eligibility | 239 months; rejected |

This training rule provides a chronological target boundary; it does not remove
the hindsight in final FRED observations or retrospective NBER classifications.
The cutoff is **not** a historical NBER announcement-availability rule.

## Configuration and failure policy

`config/baseline.json` is the shared configuration. All settings and the actual
training coverage are copied into each successful report. Unknown configuration
keys and invalid values fail rather than being ignored.

| Setting | Default | Meaning / supported changes |
|---|---|---|
| `model` | `probit` | Fixed; no alternative estimator |
| `horizon_months` | 6 | Fixed for this study and the Step 2 dataset |
| `training_start` | `1953-04-01` | Earliest permitted predictor month |
| `min_training_months` | 240 | Minimum actual eligible rows; changing it changes the study protocol |
| `forecast_cutoff_month_offset` | 1 | Fixed: issuance in the following month |
| `forecast_cutoff_day` | 15 | Calendar day; supported values 1–28 |
| `training_target_lag_months` | 1 | Latest eligible label month is t minus this lag; must be at least 1 |
| `standardize` | false | Fixed; standardization is not implemented |
| `optimizer` | `newton` | Fixed; no automatic optimizer switching |
| `maxiter` | 100 | Positive iteration limit |
| `tolerance` | 1e-10 | Newton parameter convergence tolerance |
| `max_average_score` | 1e-6 | Maximum absolute per-observation likelihood gradient accepted after fitting |
| `max_condition_number` | 1e12 | Maximum accepted design/information matrix condition number |

The numerical gradient check is an optimizer diagnostic, not a forecast scoring
metric. The program rejects missing/non-finite inputs, nonbinary labels, duplicate
or misaligned dates, insufficient samples, a constant predictor, and single-class
training. Complete and quasi-complete separation are checked via class-specific
spread ranges, appropriate to this one-predictor model, and statsmodels warnings.
It also rejects optimizer exceptions, nonconvergence, numerical warnings,
non-finite estimates, excessive gradients and ill-conditioned information matrices.
The separation check must be redesigned if additional predictors are introduced.

Failure raises a `BaselineError` with an explicit code and available diagnostics.
The CLI exits nonzero and prints a JSON failure to stderr; it does not overwrite
a previous successful report. A stale report must not be interpreted as output
from a failed run. There is no logistic substitution, automatic regularization,
probability clipping, row dropping, or other fallback estimator.

Successful reports record coefficients, model-based standard errors, training
dates, sample count, class balance, convergence/iteration status, warnings,
log likelihood, gradient and conditioning diagnostics. These standard errors
are fit diagnostics, not a claim of valid time-series inference. Numerical
convergence alone does not establish model adequacy or predictive performance.

## Reproduction and smoke verification

From the repository root, after creating `.venv` as documented in the README:

```bash
.venv/bin/python -m pip install -e . -c requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m music_recession.baseline --predictor-month 1973-10-01
```

`requirements-baseline.txt` records the runtime dependency versions used for this
verification on Python 3.9.6. The equivalent installed command is
`.venv/bin/baseline-smoke`. Defaults read `data/processed/economic_monthly.csv`
and `config/baseline.json` and write `reports/baseline_smoke.json`. CLI overrides
are `--data`, `--config`, `--predictor-month` and `--output`; all relative paths
are resolved from the working directory. The report records data/config hashes
and the numerical-library versions. Minor numerical differences across library
versions or platforms are possible.

For October 1973, spread is -0.43 percentage points. The verified fit yields
intercept -0.5260581048 and spread coefficient -0.6003347955, converges in six
iterations without warnings, and returns probability **0.3943827036** for
April 1974. This single prediction is an execution check, **not a measured
forecasting result**. No aggregate Brier Score or full backtest is calculated.
The Step 2 data manifest remains the record of the economic preparation stage;
the Step 3 smoke report records the model execution separately.

## Next phase, not implemented here

The full expanding-window backtest and evaluation remain future work. When music
features become available, **rerun both models on compatible training windows
and identical evaluation months**; do not compare this full-history economic
fit against a music model trained on an incompatible subset.

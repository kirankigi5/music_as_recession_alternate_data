# Economic dataset and Model A protocol — Steps 2–4

## Status and scope

This is a **retrospective monthly dataset**, constructed from the existing local
`fredgraph.csv` snapshot and cross-checked against local `GS10.csv`. It is not a
historical real-time information set. Retrieval dates, historical vintages and
release timestamps are not supplied in these inputs. Local consistency checks
do not independently authenticate the observations or establish their vintages.
The original files are preserved byte for byte. No observations are downloaded,
imputed or fabricated. Step 2 prepares the economic dataset. Step 3 implements
Model A and a single-forecast smoke test. Step 4 implements the retrospective
expanding-window evaluation described below.

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

## Step 4: retrospective expanding-window evaluation

The evaluation predictor months are fixed at **October 1973–March 2026**.
The associated target months are **April 1974–September 2026**. Inclusive calendar
arithmetic gives `(2026 - 1973) * 12 + (3 - 10) + 1 = 630` forecasts; the input
contains all 630 predictor months and future labels. All 630 fits converge with
no model warnings or exclusions. Training expands from 240 rows to 869 rows,
adding one row each month in this complete snapshot. The final training predictor
month is August 2025, whose target month is February 2026. The final assumed
forecast cutoff is April 15, 2026.

`src/music_recession/backtest.py` calls the existing `select_training` and
`fit_probit` functions for every forecast, then uses `predict_probability`.
The Step 3 implementation and `config/baseline.json` are unchanged. Every fit
starts from zero coefficients, uses an intercept and the unstandardized spread,
and retains the same optimizer and acceptance checks. There are no warm starts,
retries, random splits, shuffled observations, tuning or alternate estimators.
The forecast's actual label is used only after fitting for evaluation.

This is **retrospective pseudo-out-of-sample evaluation**, because each fitted
model excludes its future evaluation target and respects the stated chronological
boundary. It is **not a historically executable real-time forecasting record**:
final FRED values and retrospective NBER labels still carry historical hindsight.
The 15th-of-next-month convention does not resolve publication/vintage uncertainty.

### Metrics and the reference forecast

For each forecast, the reference probability is the recession fraction in that
same eligible training sample. It does not use the evaluation label, overall
dataset prevalence, or a fixed prevalence estimated from future observations.
Model and reference are evaluated on exactly the same successful, labeled months.

- **Brier Score (primary):** mean of `(probability - actual_label)^2`.
- **Reference Brier Score:** the same calculation for the expanding training
  prevalence forecast on the paired evaluation months.
- **Brier Skill Score:** `1 - model_BS / reference_BS`. Positive values mean
  smaller squared probability error than the reference. A zero reference score
  makes skill undefined and is recorded as null.
- **ROC-AUC:** the pairwise ranking interpretation, implemented with average
  ranks for ties. It is null if only one outcome class occurs.
- **Log loss:** negative mean natural log of the observed-outcome likelihood
  (`p` for a recession, `1-p` otherwise). Only an exactly zero likelihood is
  floored to float64 epsilon, approximately 2.22e-16. Positive likelihoods,
  including values below epsilon, are unchanged. The saved probabilities and
  all other metrics are never clipped. **No clipping was needed in this run.**
- **Calibration:** ten bins fixed before evaluation: `[0,0.1)`, `[0.1,0.2)`, …,
  `[0.9,1]`. Each records its sample count, mean predicted risk and observed
  recession frequency. Empty bins remain with count zero and null rates.
  There is no fitted recalibration or threshold optimization.

Accuracy at a 0.5 threshold is not used as the primary measure. The evaluation
has 70 recession months out of 630 (11.11%), and monthly observations are not
independent recession episodes. No significance claim or confidence interval is
inferred from these point estimates; block-bootstrap uncertainty remains future work.

### Record schema and failure accounting

`reports/baseline_backtest/forecasts.csv` uses schema version 1 and one row for
**every scheduled predictor month**. It records `model_id`, predictor/cutoff/target
dates, spread, probability, actual label, reference probability, training counts
and boundaries, training prevalence, coefficients, convergence and numerical
diagnostics. `coefficients_json` supports future model-specific coefficient sets;
the fixed intercept/spread columns are convenient Model A fields. JSON-valued
fields are CSV-quoted. `training_months_sha256` hashes the ordered eligible
predictor-month list (newline separated, no trailing newline) to help audit
training-window compatibility in the future Model B comparison.

Statuses are `ok` (forecast and label available), `missing_actual` (forecast
available but label absent), and `failed` (missing predictor row, ineligible
training window, or model failure). Non-OK records include reasons and available
diagnostics and are also saved in `failures.json`. Blank CSV cells mean unavailable,
not zero. Non-finite failure diagnostics are represented as explicit strings such
as `"inf"` in JSON; they do not become valid probabilities.

Metrics use paired `ok` months only and explicitly report scheduled, successful,
failed, missing-label and evaluated counts, all excluded months, and whether
coverage is complete. Missing source months are not filled: windows count actual
eligible observations. Malformed source data aborts input validation rather than
silently dropping rows. CLI exit code 0 indicates full evaluation coverage; code 2
indicates incomplete coverage with the reports still saved; code 1 indicates an
aborted run. After an aborted run, older reports must not be treated as new output.

### Artifacts and reproduction

From the repository root:

```bash
.venv/bin/python -m pip install -e . -c requirements-baseline.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m music_recession.backtest
```

The installed `baseline-backtest` entry point is equivalent; no separate wrapper
script is required. Defaults read the validated economic CSV and existing model
configuration. `--data`, `--config`, `--start`, `--end` and `--output-dir` provide
explicit overrides for diagnostics or future compatible-sample comparisons.
The reported project run uses the prescribed defaults without tuning.

The output directory contains:

- `forecasts.csv`: all forecast records, including failures if any;
- `metrics.json`: metrics, complete calibration table, coverage audit, model
  settings, input/code hashes, and numerical-library versions;
- `calibration.csv`: the same ten-bin table in tabular form;
- `failures.json`: all non-evaluated records (empty for this run);
- `probability_timeline.png/.svg`, `calibration.png/.svg`, and
  `brier_comparison.png/.svg`: locally saved raster and vector figures.

The timeline places each probability at its **target month**, with actual USREC
recession target months shaded. It does not suggest those forecasts were issued
in the target month: issuance is on day 15 of predictor month + 1. Calibration
figures show bin counts, and Brier charts compare identical evaluation months.
Metrics and forecast records are deterministic in the pinned environment;
small floating-point/platform or figure-rendering differences are possible elsewhere.

### Verified results and interpretation

| Metric | Value |
|---|---:|
| Model A Brier Score | 0.0949210112 |
| Expanding-prevalence Brier Score | 0.1018706841 |
| Brier Skill Score | 0.0682205383 |
| ROC-AUC | 0.7202040816 |
| Log loss | 0.3381485845 |
| Mean predicted recession probability | 14.55% |
| Observed recession frequency | 11.11% |

Model A reduces Brier error by **6.82% relative to expanding prevalence** in this
retrospective sample. This is a modest aggregate improvement, not a significance
claim, causal result, or guarantee about real-time forecasting.

Calibration is uneven. The 0–10% bin has 345 months, mean prediction 3.66% and
observed frequency 5.22%. The 20–30% bin has 82 months, mean prediction 24.06%
and observed frequency 7.32%. High-probability bins are sparse: the 90–100% bin
contains only five months, with one recession label. Do not overinterpret its
20% observed rate as a stable population calibration estimate.

The timeline shows elevated risk around 1980–81, but also large errors in exact
six-month timing: September 1980's target has a 97.13% forecast and label 0.
For the 18 recession target months January 2008–June 2009, mean predicted risk
is only 8.97%; the earlier target-month peak is September 2007 (40.57%). For
March–April 2020, probabilities are 33.20% and 28.77%. Risk rises again for
2023–25 targets, peaking at 57.55% in November 2023, although all those target
months are labeled 0 in the supplied snapshot. These are descriptive findings,
not evidence of causal effects or a reason to retune the horizon after evaluation.

## Recommended next phase

Audit lyrics-source feasibility, permissions, matching coverage and potential
year-end selection bias before implementing Model B. When music features become
available, **rerun both models on compatible training windows and identical
evaluation months**. Retain this baseline as the reference implementation and
add uncertainty analysis before making claims about improvement. No lyric
processing or Model B is implemented in Step 4.

# Can Popular Music Predict U.S. Recessions?

**Evaluating music-derived emotion indicators as alternative data for six-month-ahead U.S. recession forecasting**

**NYU Courant | Predictive Analytics | Fall 2026 | Team 5**  
**Team:** Keefer Wu · Nathan Starkman · Kiran Reddy Bhumireddy

> **Project status:** Model A reproduces all **630 forecasts** from October 1973–March 2026. The paired Model A/B runner, cached GoEmotions adapter, monthly aggregation and multivariate Probit are implemented and tested with synthetic fixtures. The actual music experiment is **blocked: zero records satisfy the current source-use evidence policy**. No real lyrics were scored and no Model B forecasting result is claimed. All analysis remains retrospective, with historical release/vintage limitations.

## Overview

Can emotions expressed in popular music help forecast the U.S. economy? This project investigates whether **popularity-weighted lyrical emotion indicators** improve predictions of U.S. recession status **six months ahead**, beyond a standard economic baseline based on the Treasury yield curve.

We will compare two **Probit** models on the same chronological test periods:

- **Baseline (Model A):** 10-year minus 3-month Treasury yield spread.
- **Extended (Model B):** The same yield spread **plus six monthly lyrical emotion features**.

The goal is to measure **incremental predictive value**, not to claim that music causes recessions. A null or negative result is a valid research outcome.

## Research Question

> Do popularity-weighted emotions in popular-song lyrics improve out-of-sample forecasts of U.S. recession status six months ahead, compared with a yield-curve-only Probit model?

For a forecast made using information available at month $t$, the binary prediction target is:

$$
Y_t = \mathrm{USREC}_{t+6},
$$

where $Y_t=1$ means the U.S. is classified as being **in a recession six calendar months later**. This is a **recession-status** target, not a prediction of whether a new recession begins at any time within the next six months.

**Primary evaluation metric:** Out-of-sample **Brier Score** (lower is better).  
**Additional metrics:** ROC-AUC and PR-AUC; a three-month horizon may be explored as a sensitivity analysis.

## Data Sources

| Data | Source | Frequency / role | Status |
|---|---|---|---|
| U.S. recession indicator (`USREC`) | [FRED — USREC](https://fred.stlouisfed.org/series/USREC) | Monthly binary recession label; future target | Local snapshot validated, Dec 1854–Sep 2026 |
| 10-year Treasury yield (`GS10`) | [FRED — GS10](https://fred.stlouisfed.org/series/GS10) | Monthly long-term yield | Local snapshots agree, Apr 1953–Sep 2026 |
| 3-month Treasury bill rate (`TB3MS`) | [FRED — TB3MS](https://fred.stlouisfed.org/series/TB3MS) | Monthly short-term yield | Local snapshot validated, Jan 1934–Sep 2026 |
| Billboard Hot 100 | [Historical weekly Billboard archive](https://github.com/mhollingshead/billboard-hot-100) | Chart date, song, artist, rank | Historical source identified; matching pending |
| Historical lyrics, candidate 1 | [Foramitti et al. (2025) OSF repository](https://osf.io/2k7ut/) | Candidate research files associated with Billboard lyrics, 1973–2023 | File contents and reuse permissions to verify |
| Historical lyrics, candidate 2 | [Kaggle: Top 100 Songs & Lyrics](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019) | Candidate year-end song and lyrics lookup | Reported dataset size is **not** verified usable coverage |
| GoEmotions annotations | [Official Google Research repository](https://github.com/google-research/google-research/tree/master/goemotions) | Emotion taxonomy and original Reddit annotations | Methodological resource |
| GoEmotions BERT checkpoint | [Hugging Face model](https://huggingface.co/monologg/bert-base-cased-goemotions-original) | Pretrained text emotion scoring | Proposed model; lyric validation pending |
| Audio-features fallback | [TidyTuesday: Billboard and Spotify](https://github.com/rfordatascience/tidytuesday/tree/main/data/2021/2021-09-14) | Candidate precomputed audio features | Backup; coverage and permissions to verify |

**Target historical period:** **1959–2023**, conditional on a viable matched lyrics sample; **1973–2023** is a potential narrower period aligned with Foramitti et al.'s work. These are *targets*, not confirmed complete study samples.

### Important lyrics limitation

Lyrics are often copyrighted and can be difficult to obtain in bulk. **Year-end hit lists are potential lyrics-lookup sources, not valid replacements for historical weekly charts.** Selecting songs based on their eventual year-end rankings would use future chart behavior when constructing earlier forecasts. We will evaluate matched-song coverage against the actual weekly chart population, including variation across months and historical periods.

## Data Integration Pipeline

The data-preparation workflow follows **CRISP-DM**, particularly **data integration (data fusion), cleaning, feature construction, and aggregation**.

```mermaid
flowchart TD
    B[Weekly Billboard Hot 100<br/>chart date · title · artist · rank]
    L[Historical lyric sources<br/>title · artist · text]
    B --> J1[Join 1: normalized title + artist<br/>record matching and coverage audit]
    L --> J1
    J1 --> NLP[GoEmotions BERT<br/>six selected emotion scores per song]
    NLP --> AGG[Popularity-weighted<br/>monthly emotion indicators]
    B --> AGG
    F[FRED GS10 and TB3MS<br/>monthly yield spread] --> J2[Join 2: calendar month t]
    AGG --> J2
    R[FRED USREC<br/>recession labels] --> SHIFT[Shift target by six months<br/>USREC at t + 6]
    SHIFT --> J2
    J2 --> M[One row per forecast month<br/>yield spread + six emotions + future label]
    M --> TEST[Time-ordered Probit comparison<br/>Brier Score · ROC-AUC · PR-AUC]
```

### Join 1: Weekly Billboard charts ↔ song lyrics

- **Proposed join key:** Canonicalized song title and artist name, with manual checks for ambiguous matches, remixes, alternate spellings, and duplicates.
- A song's lyrics can be scored once, then reused for each dated chart appearance of that song.
- Unmatched weekly chart entries remain in a **coverage audit**; they are not silently treated as emotionally neutral.
- Lyrics coverage, permitted use, and possible year-end selection bias determine whether this approach is feasible.

### NLP: Lyrics → emotion scores

GoEmotions provides **27 emotion categories plus neutral**. We propose selecting **six native labels**, chosen before inspecting recession outcomes:

**Joy · Sadness · Fear · Anger · Nervousness · Optimism**

The implemented adapter uses a pinned pretrained **RoBERTa GoEmotions classifier**. It pools nonoverlapping 480-token chunks by content-token count; see [the execution protocol](docs/model_comparison.md). Actual lyric inference is blocked until eligible records exist.

**Domain-shift check:** GoEmotions was trained on **Reddit comments**, not lyrics. We plan a small manual review of predictions on a sample of songs before trusting the emotion indicators. The precise sampling and agreement protocol will be documented during implementation.

### Aggregate: Weekly charts → monthly indicators

The implemented preparation uses chart rankings to give more popular chart entries more influence. Its rank weight is:

$$
w_i = 101 - r_i
$$

For emotion $k$ and forecast month $t$:

$$
E_{t,k} =
\frac{\sum_{i\in \mathcal C_t^{\mathrm{matched}}}w_i e_{i,k}}
     {\sum_{i\in \mathcal C_t^{\mathrm{matched}}}w_i},
$$

where $\mathcal C_t^{\mathrm{matched}}$ contains only **eligible chart observations known by the forecast cutoff** for month $t$. Each observation belongs to its chart-issue month, and repeated weeks contribute separate rank weights. Coverage retains unmatched and excluded observations in its denominator.

### Join 2: Monthly music features ↔ FRED

Compute the yield spread:

$$
S_t = \mathrm{GS10}_t-\mathrm{TB3MS}_t.
$$

Join monthly music indicators $E_{t,k}$ and spread $S_t$ **by calendar month**, then construct $Y_t=\mathrm{USREC}_{t+6}$. The modeling table has **one row per forecast month**.

For a truly real-time forecasting study, both publication timing and historical revisions of macroeconomic data must be considered. NBER recession dates are retrospective labels and should not be treated as instantly known during historical training.

## Modeling Approach

**Model A — Yield-curve Probit:**

$$
\Pr(Y_t=1\mid S_t) = \Phi(\beta_0+\beta_1 S_t).
$$

**Model B — Yield-curve + music Probit:**

$$
\Pr(Y_t=1\mid S_t,\mathbf E_t) =
\Phi\left(\beta_0+\beta_1S_t+\sum_{k=1}^{6}\gamma_kE_{t,k}\right).
$$

Here $\Phi$ is the standard normal CDF. The six music terms are the only added inputs in the main comparison. The implemented Probit is unregularized; separation and optimization failures are reported rather than silently changing the model.

### Evaluation protocol

1. **Use chronological expanding-window evaluation**, not random train/test shuffling.
2. Train each forecast using only predictors available at that date and outcomes that would have matured by the training cutoff (including the six-month horizon).
3. Evaluate **both models on identical test months**.
4. Compare the **Brier Score** as the primary metric:

   $$BS = \frac{1}{N}\sum_{t=1}^{N}(\hat{p}_t-Y_t)^2$$

5. Report ROC-AUC and PR-AUC as supporting discrimination metrics, with caution where test periods contain few or no recession observations.
6. Estimate uncertainty using **block-bootstrap confidence intervals**, while acknowledging that the small number of distinct recession episodes can make these estimates unstable.

The historical sample has relatively few independent recession episodes; strong claims about generalizable improvements would therefore be unwarranted. **Model A's retrospective backtest is available below. Model B comparisons and uncertainty intervals remain future work.**

## Project Plan

| Phase | Planned work | Decision / deliverable |
|---|---|---|
| **1. Sources + baseline** | Validate datasets, rights, monthly coverage; build FRED target and baseline Probit | **Go/no-go on lyrics at end of Phase 1** |
| **2. Emotion features** | Match lyrics; run GoEmotions; review predictions; aggregate music scores | Monthly music feature table |
| **3. Backtesting** | Join monthly tables; walk-forward comparison; assess uncertainty and leakage | Out-of-sample evaluation |
| **4. Final paper** | Synthesize related work, methods, findings, limitations | Reproducible report and conclusions |

**Fallback:** If usable lyric coverage is inadequate, assess historical **audio features** (such as valence and energy) from an appropriately licensed precomputed dataset. This would be a documented change to the music-feature measurement, **not** a claim that lyric-based results were obtained.

Three workstreams run in parallel: **economic baseline**, **music/NLP pipeline**, and **data integration/evaluation**, with shared validation and writing.

## Related Work

1. **Estrella & Mishkin (1998)** — [Recession forecasting with financial indicators and Probit](https://www.nber.org/papers/w5379). Foundation for the economic baseline.
2. **Zullow (1991)** — [Popular-song lyric pessimism and economic changes](https://www.sciencedirect.com/science/article/abs/pii/016748709190029S). Early motivation for testing music as an economic signal.
3. **de Lucio & Palomeque (2023)** — [Music preferences and business cycles](https://doi.org/10.1007/s10824-022-09454-7). Links music behavior and macroeconomic conditions; our focus is future forecasting.
4. **Musthyala et al. (2024)** — [AI framework for predicting Grammy winners](https://ieeexplore.ieee.org/document/10607237). Example of combining music-related features for a predictive task.
5. **Foramitti et al. (2025)** — [Billboard lyrics, stress, negativity, simplicity, and societal crises, 1973–2023](https://www.nature.com/articles/s41598-025-28327-5). Most directly related historical Billboard lyric-analysis study.

**NLP method reference:** [Demszky et al. (2020), *GoEmotions: A Dataset of Fine-Grained Emotions*](https://aclanthology.org/2020.acl-main.372/).

## Reproducibility and Repository Status

The economic preparation step is implemented in `src/music_recession/economic.py`
and uses only Python's standard library. Model A is implemented in
`src/music_recession/baseline.py` and uses statsmodels Probit, NumPy and SciPy.
Python 3.9 or newer is required; the verified environment is Python 3.9.6 with
statsmodels 0.14.6, NumPy 2.0.2, SciPy 1.13.1 and transitive dependency pandas
2.3.3. Backtest plots use matplotlib 3.9.4. Runtime version constraints are recorded in `requirements-baseline.txt`;
availability of those exact versions on newer Python versions may differ.
Tests use `unittest`; setuptools is a packaging build dependency.

From the repository root, create an isolated environment and install the package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip>=21.3'
.venv/bin/python -m pip install -e . -c requirements-baseline.txt
```

Generate the economic dataset and manifest, then run all tests:

```bash
.venv/bin/python -m music_recession.economic
.venv/bin/python -m unittest discover -s tests -v
```

The equivalent installed command is `.venv/bin/prepare-economic`. Use `--help`
to inspect optional input and output paths. Defaults are relative to the current
working directory, so run the commands above from the repository root.

For an offline, economic-preparation-only setup, skip package installation and
run only the economic tests. Model A and its tests require the numerical packages:

```bash
python3 -m venv .venv
PYTHONPATH=src .venv/bin/python -m music_recession.economic
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_economic.py' -v
```

The pipeline reads the original `fredgraph.csv` and `GS10.csv` without modifying
them. It validates monthly calendars, source values and cross-file GS10 equality,
computes `spread_t = GS10_t - TB3MS_t`, and looks up `USREC` exactly six calendar
months ahead. Rows lacking a predictor or future label are excluded and counted.
Malformed input, duplicate/missing dates and cross-file mismatches stop the run.

Outputs:

- [`data/processed/economic_monthly.csv`](data/processed/economic_monthly.csv):
  **876 rows × 7 columns**, predictor months **April 1953–March 2026**, target
  months **October 1953–September 2026**. Columns: `predictor_month`,
  `target_month`, `USREC_t`, `GS10_t`, `TB3MS_t`, `spread_t`, `target_t`.
  `USREC_t` is retained for reference only and must not enter the baseline model.
- [`docs/data_manifest.json`](docs/data_manifest.json): input/output SHA-256
  hashes, schemas, coverage, validation findings and exclusions. The current
  snapshots yield 882 matching GS10 observations; 1,180 source rows lack valid
  predictors and six final predictor months lack future labels.
- [`docs/forecast_protocol.md`](docs/forecast_protocol.md): target definition,
  validation policy, source provenance limitations and forecasting prerequisites.

Outputs are deterministic for the same inputs, paths and implementation. A
successful rerun replaces the processed CSV and manifest. Validation failures
leave previous artifacts untouched; inspect the command's exit status before
using them. Tests use temporary fixtures and independently verify all 876 output
rows against the original local files. Snapshot-specific regression expectations
must be deliberately reviewed if the source snapshots change.

**Retrospective limitation:** observation months are not publication timestamps.
These snapshots provide no historical release or vintage evidence. NBER recession
announcements can occur after the months being labeled; six elapsed months alone
do not establish training-label availability. No real-time forecasting claim is
made. See the protocol for the implemented Probit timing convention and the
limits of retrospective chronological evaluation.

### Model A: single-forecast verification

```bash
.venv/bin/python -m music_recession.baseline --predictor-month 1973-10-01
```

The equivalent installed command is `.venv/bin/baseline-smoke`. It writes
[`reports/baseline_smoke.json`](reports/baseline_smoke.json), including coefficients,
training coverage, class counts, convergence diagnostics, configuration, input
hashes and software versions. See `--help` for input/config/output overrides.

[`config/baseline.json`](config/baseline.json) specifies the protocol: issue the
forecast on the 15th of t+1, predict USREC in t+6, and train only on rows whose
target month is at most t-1. Start training history in April 1953 and require at
least 240 eligible observations. This makes **October 1973** the first eligible
predictor month: training predictors run April 1953–March 1973, with targets
October 1953–September 1973. September 1973 has only 239 eligible observations.

The model includes an intercept and only the spread in percentage points, without
standardization. Invalid data, single-class samples, constant spread, separation
and optimization failures produce explicit errors; no alternative model is
silently substituted. The full settings and failure policy are documented in
[`docs/forecast_protocol.md`](docs/forecast_protocol.md).

The October 1973 smoke fit uses 240 months (37 recession labels, 203 non-recession
labels). It converges in six iterations without warnings: intercept **-0.526058**,
spread coefficient **-0.600335**. With October's spread of **-0.43**, the probability
for April 1974 is **0.394383**. The assumed issuance date is November 15, 1973.
**This is one execution check, not a measured forecasting result.** Final FRED
observations and NBER classifications still contain historical hindsight.

### Model A: expanding-window backtest (Step 4)

```bash
.venv/bin/python -m music_recession.backtest
.venv/bin/python -m unittest discover -s tests -v
```

The equivalent installed command is `.venv/bin/baseline-backtest`. It reuses the
Step 3 fitting and eligibility functions without altering the model or numerical
settings. For each predictor month t, training begins in April 1953 and includes
only rows with target months at most t-1. The assumed cutoff remains the 15th
calendar day of t+1. A new fit is made for each month; there are no random splits,
warm starts, hyperparameter searches or fallback estimators.

**630 of 630 forecasts succeeded**, with no model warnings. Predictor months run
October 1973–March 2026; target months run April 1974–September 2026. Training grows
from 240 to 869 observations. The calendar count and every training window were
checked independently against the raw economic snapshot.

| Metric | Retrospective result |
|---|---:|
| **Model A Brier Score** | **0.094921** |
| Expanding-training-prevalence Brier Score | 0.101871 |
| Brier Skill Score | 0.068221 |
| ROC-AUC | 0.720204 |
| Log loss | 0.338149 |
| Mean predicted recession probability | 14.55% |
| Observed recession frequency | 11.11% (70 / 630) |

The reference predicts the recession frequency in each forecast's eligible
training sample. On identical evaluation months, Model A reduces Brier error
by **6.82%** relative to this benchmark. Brier Score remains the primary measure;
this point estimate does not establish statistical significance or causation.

Outputs in [`reports/baseline_backtest/`](reports/baseline_backtest/):

- [`forecasts.csv`](reports/baseline_backtest/forecasts.csv): all scheduled months,
  probabilities, actual labels, reference probabilities, training coverage,
  coefficients, convergence and diagnostics in a stable versioned schema.
- [`metrics.json`](reports/baseline_backtest/metrics.json) and
  [`calibration.csv`](reports/baseline_backtest/calibration.csv): paired metrics,
  ten fixed calibration bins with counts, coverage, input/code hashes and versions.
- [`failures.json`](reports/baseline_backtest/failures.json): all non-evaluated
  records and reasons; empty for this run. Failed months are never silently removed.
- [Probability timeline](reports/baseline_backtest/probability_timeline.png),
  [calibration plot](reports/baseline_backtest/calibration.png), and
  [Brier comparison](reports/baseline_backtest/brier_comparison.png), each also
  saved as SVG for export.

The timeline is indexed by **target month**, with actual recession target months
shaded. It shows high risk around 1980–81, timing errors around the 2008–09 recession,
and elevated 2023–25 probabilities despite non-recession labels in the supplied
snapshot. Calibration is uneven: the 20–30% bin averages 24.06% predicted risk
but only 7.32% observed recessions (82 months); high-probability bins have few
observations. Read the protocol for fuller interpretation and limitations.

No probability clipping is used for Brier Score, ROC-AUC or saved forecasts.
Log loss floors only an exactly zero observed-outcome likelihood to float64
epsilon; none needed this treatment in the actual run. Incomplete coverage is
reported explicitly and returns CLI exit code 2; malformed input aborts with code
1. Successful full coverage returns 0. Defaults are fixed to the requested sample;
`--help` describes overrides for diagnostics and later compatible-sample work.

These results remain **retrospective pseudo-out-of-sample**, not evidence of
historically executable real-time forecasts. Final FRED data and retrospective
NBER labels contain hindsight, monthly errors are dependent, and no uncertainty
intervals have yet been calculated. Lyrics feasibility and coverage auditing is
the recommended next step. The later music comparison must rerun both models on
compatible training windows and identical evaluation months.

**Data rights:** Do not commit copyrighted full lyrics or redistribution-restricted datasets to the repository. Only share data, derived features, or metadata when the relevant terms permit it. Source datasets should be obtained from their original publishers or repositories.

---

*Prepared for the NYU Courant Predictive Analytics course, Fall 2026. Economic preparation, the Probit baseline and retrospective backtesting are implemented; music-source availability and historical real-time availability remain subject to verification.*

## Local lyrics validation (1973–2023)

The new offline audit preserves the master inventory and historical reports, and
writes only metadata, exact source locators, content hashes, and coverage statistics.
Candidate availability is **provisional**, not verified recording identity or proof
of historical real-time availability. Raw lyric files remain under ignored
`data/raw/lyrics/`; no lyrics are exported to reports.

```sh
.venv/bin/python -m music_recession.lyrics_audit --output reports/lyrics_validation/new_run
.venv/bin/python -m unittest discover -s tests -q
```

Choose a fresh output directory. See [validation protocol](docs/lyrics_validation.md),
[source registry](config/lyrics_sources.json), and
[review checklist](docs/music_review_checklist.md). Existing `scripts/tally_osf_lyrics.py`
and `scripts/apply_osf_aliases.py` overwrite the original inventory and predate these
checks; use the new audit for reproducible validation. The audit runs no BERT or Model B.

## Current Model A/B comparison runner

The paired comparison is implemented with eligibility-gated local GoEmotions
scoring, source/content-keyed caches, monthly Billboard weighting and expanding
Probit fits. See [execution and evidence requirements](docs/model_comparison.md).

```sh
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m music_recession.music_model --output-dir reports/model_comparison/new_run
```

Use a fresh output directory. With the current empty review registry, the command
returns exit 2 (`blocked_no_policy_eligible_lyrics`), saving coverage and forecast
status reports without model downloads or lyric inference. The 925 newly recovered
candidates are not automatically admitted. The original Model A baseline remains
unchanged and its broad-period score is not a paired Model B comparison.

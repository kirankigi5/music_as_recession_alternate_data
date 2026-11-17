# PR #2: conditional paired inference

This branch integrates Keefer Wu's original commit
`3e8ddc6109c76747a1ef1bf1b086ad6c227044c2` with main's PR #1 merge
`98d837f78a74bc395a195dd606256c4b82208cb0`. It adds analysis of the **unchanged
saved forecasts**, not a new modeling experiment. The original separate local
uncertainty implementation is not imported. PR #1's [validation record](closeout_validation.md)
and [review dispositions](review_response.md) remain historical records of that release.

## Data and estimand

For each `status=ok` row, the loss differential is
`d_t = (p_B,t − y_t)^2 − (p_A,t − y_t)^2`. Positive mean differential means
Model B has greater observed Brier loss. Both probabilities and the binary
outcome come from the same saved row. Dates must be strictly increasing,
first-of-month, with target exactly six calendar months after predictor month.
The target is **status in that single month**, not onset or a cumulative outcome
over six months. The assumed issuance is the 15th of t+1; this remains a
retrospective pseudo-out-of-sample exercise.

Calendar gaps are retained. The 183/102/45 evaluated dates span 362/316/263 predictor
months, respectively. Primary dates run November 1993–December 2023, 75% dates
November 1993–February 2020, and 80% dates May 1994–March 2016.
The 75% and 80% sets are subsets of the
primary dates, not independent replications. Recession-positive months come
from three/two/one episodes, respectively (2001, GFC and COVID at primary).
Thresholds change the evaluation and training samples; cross-threshold AUCs
cannot isolate a coverage effect.

## Methods and conventions

### Circular calendar block bootstrap

We retain Keefer's construction: 10,000 draws, NumPy seed 20261026, block lengths
6/12/24 **calendar months**, with 95% percentile intervals. A replicate samples
uniform starts on the complete evaluation calendar, concatenates blocks until
the original calendar span is filled, and truncates the final block to the
remaining calendar length. Blocks wrap from the last month to the first.
The loss sum and observed-month mask are resampled together; the statistic is
the sum of observed losses divided by the number of observed forecasts drawn.
Missing months do not become zero losses. Empty replicates are dropped and
counted; an all-empty draw set raises an error rather than producing an interval.

Circular starts give every calendar position equal inclusion opportunity,
avoiding a systematic endpoint discount. They also create artificial adjacency
between endpoints. The construction assumes enough local stability for blocks
to approximate dependence; changing eligibility, gaps, rare episodes and a
30-year retrospective sample weaken that assumption. These are sensitivity
analyses, not a uniquely correct bootstrap. The 6/12/24 choices retain Keefer's
shorter, annual and biennial scales; the single-month target does not mandate
a six-month overlap correction. No scale was selected based on significance.

The JSON includes actual retained draw counts and the fraction of bootstrap
means at or below zero. That fraction is **not a null-centered test p-value**.
The independent-row interval is retained as a diagnostic comparison only;
it discards temporal dependence. No multiple-setting multiplicity adjustment
is claimed, and the shortest interval is not a primary test selected after inspection.

### Calendar HAC and approximate DM-style comparison

Let n be observed forecasts, z_t = d_t − mean(d) at observed dates, and z_t = 0
at missing calendar dates. The variance of the observed-forecast mean is

`[sum(z_t²) + 2 sum_{k=1..L} (1 − k/(L+1)) sum_t(z_t z_{t+k})] / n²`.

Thus only observed pairs separated by exactly k calendar months contribute.
Zeros represent missing **centered estimating-equation contributions**, not
imputed observed losses. The Bartlett kernel is nonnegative up to rounding;
lag L includes weights through L, with denominator L+1. When L exceeds the
calendar span only available cross-products contribute, retaining those weights.
There is no n/(n−1) small-sample correction. Dense-calendar tests compare with
statsmodels Newey–West, and a sparse-calendar sandwich check independently
verifies the observed-count normalization.

The statistic is mean(d) / HAC-SE. We retain Keefer's two-sided Student-t(n−1)
reference as an **approximation** and label its distribution and degrees of
freedom in JSON. It is not an exact t-test, an episode-level adjustment or a
Harvey–Leybourne–Newbold horizon correction. Lags 5/12/24 remain explicit
sensitivities; lag 5 is not compelled by overlapping six-month outcome windows.
Only three primary recession episodes make asymptotic/reference-distribution
accuracy doubtful. Constant or singleton series have undefined statistics and
p-values (`null`). No “50 independent months” claim or effective-sample-size
field remains: a variance ratio does not establish independent observations
or valid degrees of freedom in this selected, gappy sample.

### Sparse-calendar diagnostic

`cross_product_ratio_by_calendar_lag` divides the average centered product
over observed lag-k pairs by the global observed variance. Pair counts accompany
each value. This ratio is **unbounded and is not a correlation coefficient**;
the 80% sample's lag-24 ratio is about 3.3566 from only three pairs. It is a
descriptive diagnostic, not an input to HAC, a stability estimate or an effective
sample-size calculation. Undefined cases return `null`.

### Average precision and training-prevalence benchmark

`average_precision` sums precision times recall increments at distinct score
thresholds, grouping ties. It is stepwise **average precision**, not trapezoidal
precision–recall area. No positives gives `null`; all positives gives 1.
Evaluation prevalence is reported as a constant-score reference, not a
forecast available in real time or the exact finite-sample expected AP of random ranks.

For Brier skill, the reference probability at **each origin** is
`training_recession_count / training_count` from that saved row. It is not the
whole evaluation sample's recession rate. Counts must be valid nonnegative
integers with positives <= total and total > 0. Skill is
`1 − model Brier / reference Brier` (undefined for zero reference loss).
These are the paired training windows, not the separate full-history A benchmark.

## Results and sensitivity

Exact values, settings, input SHA-256 hashes and all diagnostics are in
[`results/paired_inference.json`](results/paired_inference.json).

| Coverage | n / positives / episodes | Mean B−A | AP A | AP B | Evaluation prevalence | Training-rate Brier |
|---|---|---:|---:|---:|---:|---:|
| 70% primary | 183 / 15 / 3 | +0.029867 | 0.168785 | 0.072249 | 0.081967 | 0.082971 |
| 75% | 102 / 9 / 2 | +0.040664 | 0.629547 | 0.099982 | 0.088235 | 0.089063 |
| 80% | 45 / 4 / 1 | +0.056555 | 0.916667 | 0.160387 | 0.088889 | 0.090071 |

| Coverage | 6-month 95% interval | 12-month 95% interval | 24-month 95% interval | Empty draws, 6/12/24 |
|---|---|---|---|---|
| 70% | [0.003055, 0.059422] | [−0.000991, 0.061378] | [−0.003891, 0.058321] | 0 / 0 / 0 |
| 75% | [0.005031, 0.084646] | [0.003819, 0.086938] | [0.004438, 0.077902] | 0 / 0 / 0 |
| 80% | [−0.001972, 0.133719] | [−0.002745, 0.145452] | [−0.002191, 0.142801] | 1 / 4 / 31 |

| Coverage | Approximate p, HAC lag 5 | Approximate p, lag 12 | Approximate p, lag 24 |
|---|---:|---:|---:|
| 70% | 0.039930 | 0.059871 | 0.047918 |
| 75% | 0.045109 | 0.051270 | 0.024629 |
| 80% | 0.091107 | 0.100313 | 0.096629 |

The primary 12/24-month intervals include zero and the lag-12 approximate p-value
exceeds .05; other settings differ. Even the nested 75% sample's lag-12 p-value
exceeds .05 despite its percentile intervals excluding zero. Bootstrap intervals
and HAC references are different approximations. We do not select the favorable
one or describe significance as robust. The descriptive conclusion remains no
demonstrated incremental benefit from these features on the selected sample.

### What this uncertainty leaves out

Resampling conditions on saved predictions, training outcomes, selected eligible
months and the fitted pipeline. It does not refit models or propagate training
estimation, lyric matching, language/recording identity, classifier measurement,
source rights, coverage-policy selection or economic-vintage uncertainty. It
cannot establish real-time forecasting validity or causal effects. Few recession
episodes, nested samples and nonstationarity remain major limits.

PR #1's calibration (44 months in [0.3,0.8), one recession), eight-parameter
events ratio (46–61 / 8 = 5.75–7.625), continuous percentage-point spread,
probability interpretation, sample distinctions and original predictions remain
unchanged. Historical NLP manifests retain their original hashes; the reported
7,490 identical shared vectors and 8,800 songs without a current-hash rerun are
not newly rerun inference here. Full reconstruction still needs external raw
inputs, weights and historical/current caches; see [reproduction levels](closeout_validation.md#full-inference-is-a-separate-reproduction-level).

## Offline reproduction and validation

After installing the documented numerical environment, run from the checkout
root. Replace FRESH with a new unused name. No dataset or model downloads,
model fitting or real lyric inference are needed for these commands.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/run_paired_inference.py --out /tmp/music-paired-FRESH.json
cmp docs/results/paired_inference.json /tmp/music-paired-FRESH.json
python scripts/verify_saved_results.py --output /tmp/music-verified-FRESH.json
python scripts/make_research_figures.py --out /tmp/music-figures-FRESH
git diff --check
```

Validation on November 17, 2026 used an index-only release copy, excluding the
original workspace's ignored raw sources, model caches and untracked work.
`PYTHONPATH` pointed to that release's `src` and root (the existing virtualenv's
editable install otherwise points to the dirty workspace). Model-hub offline
flags and `PYTHONDONTWRITEBYTECODE=1` were set; pytest cache writing was disabled.
The interpreter was Python 3.9.6 with NumPy 2.0.2, SciPy 1.13.1, statsmodels
0.14.6, matplotlib 3.9.4 and pytest 8.4.2. This tests the release in that existing
environment, not a fresh dependency installation on all supported platforms.

- Full integrated suite: **215 passed in 15.74 seconds; 0 failures, 0 skips**.
  This comprises 191 existing tests and 24 inference tests. The focused inference
  run also passed all 24 tests in 1.16 seconds.
- Two fresh CLI regenerations of the paired JSON were byte-identical to each
  other and the bundled JSON. SHA-256:
  `7e0412784e29a62ef65fcadb30081f98054ce67b0325b5b4d5fc4c3c87a775b9`.
- PR #1 saved-result verification passed for all three runs and reproduced
  `closeout_verification.json` byte for byte: original metrics, calibration,
  benchmark, coefficients/probabilities and historical asset/code checks passed.
- The original figure path generated **10 PNGs**, with a results-summary JSON
  exactly equal to the bundled summary. Figure bytes are not claimed portable
  across fonts/renderers; the generated images were not staged.
- **89 local file links and 21 Markdown anchors passed**, with zero failures.
  External URLs were not exhaustively probed. `git diff --check` and
  `git diff --cached --check` passed.
- Of 112 main files, 106 were byte-identical to merged main in the release;
  only six existing documentation files differ.
  The five added files are this methods record, paired analysis JSON,
  inference module, CLI and tests. Models, original predictions, configurations,
  historical manifests and NLP code remain exact.
- Original bootstrap outputs match Keefer's prior JSON exactly; HAC statistics
  and p-values agree within 1e-14 (floating-point library rounding). The new
  schema names AP and the unbounded diagnostic precisely, removes the misleading
  effective-observations field, and adds source hashes and scope metadata.
- No failing validation checks remain. Full raw-data/current-code NLP inference,
  external reviewer experiments and real-time validity remain unverified within
  the documented reproduction boundary.

Tested analysis-file SHA-256 values (the validation prose was added after these
checks; the executable/test bytes were unchanged):

- src/music_recession/inference.py: `9e4d8954cd3a55ca7c70e417667ba81e70e81958439f0c5e987fb47b214e1775`.

- scripts/run_paired_inference.py: `4c96933454d2fae05ae92dd8ed25b8cca8a1e9e90de05f68d9f10ebb518f40e3`.

- tests/test_inference.py: `9581491a955b2161fe61c49bc343b6d7fabebce09d1f939b1fc185615f37390d`.

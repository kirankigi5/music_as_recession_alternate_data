# Exploratory results: six-month recession-status prediction

The original experiment is complete under the exploratory coursework policy.
Automated screens are not verified recording/language/release evidence. Source
use rests on reported instructor guidance, not a licence. See the
[policy](exploratory_course_policy.md) and [source review](osf_source_use_review.md).

## Original paired results

| Coverage | Forecasts | Recession months | A Brier | B Brier | B − A | A ROC-AUC | B ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 70% primary | 183 | 15 | 0.0903818731 | 0.1202485779 | +0.0298667048 | 0.6462 | 0.4306 |
| 75% sensitivity | 102 | 9 | 0.0739858409 | 0.1146495991 | +0.0406637582 | 0.8722 | 0.5424 |
| 80% sensitivity | 45 | 4 | 0.0640448014 | 0.1206001394 | +0.0565553380 | 0.9878 | 0.7012 |

Both models use identical training rows, forecast dates and outcomes within each
run. Brier is a loss, so positive B − A means higher observed error for B. All
three runs record zero fit failures. This supports no demonstrated incremental
benefit from these features in this exploratory sample; it does not establish
statistical degradation or identify a cause.

The primary run has 16,290 reported scored songs and 612 feature-table months.
Predictor dates span November 1993–December 2023; targets span May 1994–June 2024.
The 183 evaluation dates are nonconsecutive, with 15 positives (8.20%). The target
is status at t+6. Issuance is assumed on the 15th of t+1, so the horizon is indexed
from the predictor month, not the day of issuance.

## Sample selection and benchmark checks

| Coverage | Coverage-only months | Feature-eligible months | 2001 / GFC / COVID positive months |
|---|---:|---:|---|
| 70% | 433 | 429 | 8 / 5 / 2 |
| 75% | 352 | 348 | 7 / 0 / 2 |
| 80% | 295 | 291 | 4 / 0 / 0 |

Four otherwise covered months (December 1976–March 1977) have missing chart rank
slots. The final feature gate excludes them. Threshold evaluation dates are
nested; entire episodes disappear, so these are related sensitivity checks.

| Forecast, evaluated on the same 183 primary dates | Brier |
|---|---:|
| Expanding paired-training recession rate | **0.0829708** |
| Full-history Model A, trained from April 1953 | **0.0847541** |
| Paired Model A, eligible history from January 1973 | 0.0903819 |
| Model B, same paired rows plus emotions | 0.1202486 |

The full-history score is selected from existing baseline forecasts on B's exact
dates. It changes training history and supplies a practical benchmark, while the
paired experiment isolates the added features. The separate 630-forecast score
0.094921 is not directly comparable to Model B.

## Calibration, error concentration and complexity

B assigns probabilities in **[0.3, 0.8)** to **44 evaluated months, with one
recession**. Its 0.4–0.5 bin contains 12 months and one recession (1/12); the other
populated bins in that range contain no recessions. The original claim of 42
months with zero recession frequency throughout was incorrect.

B improves on A in 97 of 183 months but loses overall. Target years 1999 and
2001 account for 79.21% of its net extra loss, covering false alarms and misses.
Primary training windows have 46–61 positive labels and eight fitted parameters
including the intercept: **5.75–7.625 events per parameter**. This ratio and the
error pattern do not prove overfitting. A local improvement in 2023–24 does not
establish a stable music signal. Reviewer-reported alternative-model results
remain unverified unless their code and outputs are supplied.

As a **post-hoc descriptive sensitivity only**, excluding March–April 2020 targets
leaves 181 forecasts and 13 positives: A Brier 0.0864971, B 0.1155202. COVID remains
in the original primary analysis. No models are refitted for this check.

## Uncertainty and PR-AUC belong to PR #2

[PR #2](https://github.com/kirankigi5/music_as_recession_alternate_data/pull/2)
contains Keefer's calendar-aware block-bootstrap/HAC implementation and average
precision. It remains a separate follow-up. An earlier local investigation used
another bootstrap convention; neither is designated canonical by PR #1.
Independent-month intervals do not establish significance in this dependent,
few-episode sample. PR #1 reports point estimates and descriptive checks only.
No PR #2 confidence intervals, p-values or PR-AUC scores are adopted here.

## Coverage and measurement limits

Monthly median popularity-weight coverage falls from 0.924 in the 1980s to 0.608
in 2020–2023; featured-credit and version-matching gaps are important contributors.
The evaluated songs/months change over time. GoEmotions was trained on Reddit
comments and has not been validated on these lyrics. Final economic values,
retrospective NBER labels, uncertain lyric availability and a modern pretrained
model prevent a historically executable real-time forecasting claim.

## Reproduce the descriptive evidence

```sh
python scripts/verify_saved_results.py
python scripts/make_research_figures.py --out /tmp/music-recession-figures-FRESH
```

Replace `FRESH` with an unused name. The [verification JSON](results/closeout_verification.json)
records recomputed metrics and input hashes. Original outputs live in
`docs/results/exploratory_primary_cov70/`, `exploratory_sensitivity_cov75/`, and
`exploratory_sensitivity_cov80/`. Each includes predictions, metrics, coverage,
calibration, features, original manifest and provenance note. Only the previously
missing sensitivity feature tables/manifests were restored, without editing them.
Large mapping/contribution files and raw inputs are not bundled.

The historical NLP hash differs from current code. A local comparison found 7,490
shared vectors identical; 8,800 songs have not been rerun under the current hash.
The old provenance files are archival records, including references to local-only
scripts; supported reproduction levels are in the [validation record](closeout_validation.md).

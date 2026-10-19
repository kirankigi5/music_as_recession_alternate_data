# Can popular music predict U.S. recessions? Exploratory course results

**Status: exploratory coursework analysis. Not a verified-evidence research finding.**

Source use rests on reported instructor guidance for a course project, recorded as
such and not as a licence, fair-use determination, or institutional approval. The
records analysed pass only automated risk and English-language screens: no
recording is independently verified, no language is human reviewed, and no release
date is verified. See `docs/exploratory_course_policy.md`. The strict primary
policy and its results are unchanged.

## Headline result

**Adding six lyric-emotion features did not improve six-month-ahead recession
prediction. Model B was worse than Model A at every predeclared threshold.**

| Coverage threshold | Paired forecasts | Recession months | Model A Brier | Model B Brier | B − A | Model A ROC-AUC | Model B ROC-AUC |
|---|---|---|---|---|---|---|---|
| **0.70 (primary)** | **183** | **15** | **0.0903818731** | **0.1202485779** | **+0.0298667048** | **0.6462** | **0.4306** |
| 0.75 (sensitivity) | 102 | 9 | 0.0739858409 | 0.1146495991 | +0.0406637582 | 0.8722 | 0.5424 |
| 0.80 (sensitivity) | 45 | 4 | 0.0640448014 | 0.1206001394 | +0.0565553380 | 0.9878 | 0.7012 |

A positive difference means Model B is worse (Brier is a loss). Both models were
refit on identical eligible training rows and scored on identical forecast dates;
Model A was re-evaluated on exactly these dates and is **not** compared with the
earlier broad-period figure of 0.094921.

Primary window: predictor months 1993-11 to 2023-12, targets 1994-05 to 2024-06.
429 of 612 months were feature-eligible at 70% coverage; 183 reached the
240-observation training requirement. Zero fit failures across all three runs.

Model B's primary ROC-AUC of **0.4306 is below 0.5**, i.e. worse than chance
ordering on this sample.

## Is the difference robust?

Yes in direction, weak in precision.

- The sign is consistent across all three predeclared thresholds, and the gap
  widens as coverage tightens (+0.0299 → +0.0407 → +0.0566).
- Paired per-month Brier differences, primary run: mean +0.0298667,
  95% bootstrap CI **[+0.0147, +0.0458]**, which excludes zero.
- Naive paired t = +3.74. This **ignores serial correlation** between overlapping
  monthly forecasts, so the true interval is wider than quoted.

One nuance matters for interpretation: Model B is worse in only **86 of 183
months (47.0%)** yet loses clearly on average. Its deficit comes from a minority
of months with large errors rather than uniform degradation.

Calibration confirms it. Model B assigned probabilities of 0.3–0.8 to 42 months
in which the observed recession frequency was **0.0** in every one of those bins:

| Bin | Model A n / predicted / observed | Model B n / predicted / observed |
|---|---|---|
| 0.3–0.4 | 20 / 0.342 / 0.150 | 24 / 0.338 / **0.000** |
| 0.4–0.5 | 6 / 0.437 / 0.500 | 12 / 0.449 / 0.083 |
| 0.5–0.6 | 3 / 0.528 / 0.000 | 4 / 0.550 / **0.000** |
| 0.6–0.7 | 1 / 0.620 / 0.000 | 2 / 0.657 / **0.000** |
| 0.7–0.8 | — | 2 / 0.737 / **0.000** |

Model B issues confident recession calls that do not materialise.

## Monthly coverage and missing-data pattern

Popularity-weighted coverage, denominator = all observed chart weight including
songs without usable lyrics. Overall median 0.7889, mean 0.7800. 612 monthly rows,
0 incomplete chart months, 4 months with missing rank slots.

| Decade | Months | Median coverage | Mean | Min | Feature-eligible at 70% |
|---|---|---|---|---|---|
| 1970s | 84 | 0.862 | 0.857 | 0.762 | 80 |
| 1980s | 120 | 0.924 | 0.918 | 0.847 | 120 |
| 1990s | 120 | 0.843 | 0.841 | 0.697 | 118 |
| 2000s | 120 | 0.710 | 0.719 | 0.634 | 68 |
| 2010s | 120 | 0.659 | 0.658 | 0.511 | **33** |
| 2020s | 48 | 0.608 | 0.607 | 0.424 | **10** |

Missingness is strongly non-random and time-trending: coverage falls from 0.924 in
the 1980s to 0.608 in the 2020s, because featured-artist billing became common and
the upstream lyrics.ovh lookup stripped those credits. 179 of 612 months fall below
the 70% threshold, concentrated after 2000. **Coverage drift is therefore
confounded with any apparent emotion effect**, and the evaluation sample is
weighted towards earlier decades.

## Monthly emotion features

All 612 months scored. Weighted means across scored months:

| Emotion | Mean | SD | Min | Max |
|---|---|---|---|---|
| joy | 0.0312 | 0.0134 | 0.0085 | 0.1310 |
| sadness | 0.0835 | 0.0229 | 0.0292 | 0.1490 |
| fear | 0.0190 | 0.0113 | 0.0036 | 0.0647 |
| anger | 0.0238 | 0.0140 | 0.0062 | 0.0813 |
| nervousness | 0.0114 | 0.0043 | 0.0037 | 0.0301 |
| optimism | 0.0339 | 0.0093 | 0.0143 | 0.0656 |

Six separate predictors were used; they were never collapsed into one score.
Sadness dominates; nervousness is both the smallest and the weakest pretrained
label. Absolute magnitudes are low because GoEmotions is a multi-label sigmoid
model applied out of domain.

## Method actually executed

- 16,290 songs scored with real GoEmotions inference
  (`SamLowe/roberta-base-go_emotions`, revision `d750483476…`, CPU, deterministic),
  cached by lyric content hash so each song was processed once.
- Monthly aggregation weighted by `101 − weekly rank`, with each weekly chart
  appearance contributing separately; denominator is total chart weight.
- Probit, six-calendar-month horizon, minimum 240 eligible training observations,
  expanding window, forecast cutoff the 15th of t+1, training targets restricted
  to t−1 or earlier. Model B standardises features on training rows only.
- Model A: yield spread `GS10 − TB3MS` alone. Model B: spread plus the six emotions.

## Limitations

1. **Three recession episodes only** (2001, 2008–09, 2020) are reachable, because
   the corpus starts in 1973 and 240 training observations are required. This caps
   achievable precision regardless of threshold.
2. **Non-random, time-trending missingness** confounds coverage with signal.
3. **No verified recordings, language, release dates or rights.** Automated screens
   are not verification.
4. **Out-of-domain emotion model.** GoEmotions is Reddit-trained and unvalidated on
   lyrics; long-text chunk pooling is unvalidated for this use.
5. **Hindsight in the targets.** Final FRED and NBER values are revised series, and
   chart issue dates are a retrospective convention, not publication timestamps.
6. A modern pretrained transformer was not available at historical forecast
   origins, so this is retrospective by construction and not a real-time strategy.

## Conclusion

On this exploratory sample, lyric emotions do not improve six-month-ahead
recession forecasts over the yield spread; they degrade them, consistently across
all three predeclared coverage thresholds. The direction of that result is robust.
Its magnitude is not precisely estimated, and with three recession episodes it
cannot be, so the honest claim is a null-to-negative result rather than a measured
effect size.

## Result locations

```
reports/model_comparison/exploratory_primary_cov70/        # 70% primary
reports/model_comparison/exploratory_sensitivity_cov75/    # 75% sensitivity
reports/model_comparison/exploratory_sensitivity_cov80/    # 80% sensitivity
```

Each contains `metrics.json`, `predictions.csv`, `monthly_features_and_coverage.csv`,
`monthly_coverage.csv`, `calibration.csv`, `monthly_contributions.csv`,
`song_source_mapping.csv`, `lyric_coverage.png`, `probability_comparison.png`,
`manifest.json` and `CODE_VERSION_PROVENANCE.md`. No lyric text appears in any
output. These directories are generated artefacts and are not committed.

Code-version caveat: results were produced by `emotions.py` sha256 `614f862b…`
while the committed file is `e5600454…`. The sole difference is a `hasattr` guard
that cannot affect a tensor operand, and scores were verified **numerically
identical on 7,490 of 16,290 songs** scored under both versions (0 differences,
max absolute difference 0.0). The remaining 8,800 songs were not re-scored, so a
full byte-for-byte reproduction under the committed hash has not been performed.

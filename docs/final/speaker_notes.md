# Speaker notes and timing

Planned narration: **11:00** across 11 main slides. Supplied slot: **12:30**; Q&A inclusion remains unconfirmed. These are targets, not measured rehearsal times. Appendix slides have no scheduled narration.

## 01 · Can popular music predict U.S. recessions?

Target **0:00–0:40** (40 seconds).

Our question is whether emotions in popular-song lyrics add predictive value beyond the Treasury yield curve. We implemented a complete source-to-forecast pipeline and compared two Probit models on the same dates. The music model has higher observed probability error, but that is different from establishing robust statistical degradation. Today we will show the data, the exact forecast calendar, the paired results and the limits of the uncertainty analysis. Our contribution is the auditable comparison, including what this retrospective design cannot establish. This is a qualified negative finding, not a claim that music contains no economic information.

Evidence: [docs/exploratory_results.md](../../docs/exploratory_results.md), [docs/results/paired_inference.json](../../docs/results/paired_inference.json).

## 02 · Three data sources; one frozen measurement model

Target **0:40–1:35** (55 seconds).

FRED supplies the ten-year and three-month Treasury series and the recession-status labels. Weekly Billboard ranks define the music universe, rather than a year-end list. We identify 21,542 title-and-artist identities, select 19,096 OSF lyric candidates, and retain 16,290 songs after exploratory automated screens. Those songs were scored with a pinned RoBERTa GoEmotions classifier. That classifier is a measurement tool, not another observed outcome dataset. These counts describe the funnel, not verified recording quality. Matching, language and release evidence remain incomplete. Sources investigated for recovery, such as Kaggle and lyric APIs, were not additional admitted lyric sources in the reported experiment.

Evidence: [README.md](../../README.md), [docs/COMPLETE_RESEARCH_REPORT.md](../../docs/COMPLETE_RESEARCH_REPORT.md), [docs/exploratory_course_policy.md](../../docs/exploratory_course_policy.md).

## 03 · Coverage changes the population we evaluate

Target **1:35–2:35** (60 seconds).

The chart shows median monthly popularity-weight coverage, calculated from the bundled coverage table. It is highest in the 1980s and falls substantially in recent decades. Coverage uses all chart weight in its denominator; the emotion mean uses scored weight. A month can therefore have a valid emotion average but still fail the coverage gate. Four months also fail because chart rank slots are missing. After preparation, the three thresholds leave 429, 348 and 291 eligible feature months, before the training requirement. Tightening coverage changes which historical periods and recessions are represented. It does not create an independent replication or establish that higher coverage causes better forecasts. Pause briefly to read the downward pattern.

Evidence: [docs/results/exploratory_primary_cov70/monthly_coverage.csv](../../docs/results/exploratory_primary_cov70/monthly_coverage.csv), [docs/results/closeout_verification.json](../../docs/results/closeout_verification.json).

## 04 · An implemented chart-to-forecast pipeline

Target **2:35–3:40** (65 seconds).

This is the implemented architecture, not a proposed system. Chart records are validated, then title-and-artist identities are reconciled to source records and screened. The frozen classifier scores long lyrics in chunks of at most 480 content tokens; chunk outputs are pooled by token count. Each weekly appearance contributes weight 101 minus its rank. A song appearing in several weeks contributes repeatedly by design, while duplicate chart slots are rejected. We aggregate six monthly emotion means, retain coverage and exclusion information, join the economic predictors by month, and refit the two Probit specifications chronologically. Saved dates, probabilities, training counts and hashes support later verification without repeating expensive lyric inference. The delivery is a research pipeline, not a deployed forecasting service.

Evidence: [src/music_recession/music_preparation.py](../../src/music_recession/music_preparation.py), [src/music_recession/emotions.py](../../src/music_recession/emotions.py), [src/music_recession/music_model.py](../../src/music_recession/music_model.py).

## 05 · Continuous inputs, binary target, probabilities

Target **3:40–4:45** (65 seconds).

Model A uses the continuous ten-year minus three-month yield spread, in percentage points. It is not an inversion indicator. Model B adds six preselected emotion means: joy, sadness, fear, anger, nervousness and optimism. Phi converts each fitted linear index into a recession probability. The outcome alone is binary. Model B has eight parameters including the intercept; Model A has two. Both are unregularized maximum-likelihood Probit models. Model B learns scaling within each training window for numerical conditioning. We retain the six interpretable features rather than adding PCA or searching for a favorable combination after seeing the scores. The TA feedback needs clarification only if a change in spread or a different target was intended; the continuous-spread interpretation is already implemented.

Evidence: [src/music_recession/baseline.py](../../src/music_recession/baseline.py), [src/music_recession/music_model.py](../../src/music_recession/music_model.py), [config/music_model.json](../../config/music_model.json).

## 06 · The six-month index starts at the predictor month

Target **4:45–5:55** (70 seconds).

This timeline is essential. The observation period is predictor month t. Issuance is assumed on the fifteenth of t plus one, after that month ends. The outcome is recession status in the single month t plus six, not onset anywhere during six months and not six months after issuance. Both models train on the same eligible rows from January 1973. A training target must be no later than t minus one, so its predictor must be no later than t minus seven. We require at least 240 actual eligible rows and never fill gaps. The first primary predictor is November 1993: assumed issuance December fifteenth, target May 1994. This boundary does not prove that NBER labels or final economic values were historically available. Pause on the distinction between index and issuance.

Evidence: [docs/forecast_protocol.md](../../docs/forecast_protocol.md), [docs/results/exploratory_primary_cov70/predictions.csv](../../docs/results/exploratory_primary_cov70/predictions.csv).

## 07 · Observed loss rises when music is added

Target **5:55–7:05** (70 seconds).

Every bar uses exactly the same 183 primary evaluation dates. The training-rate reference predicts the recession fraction in each saved paired training window; it does not use evaluation prevalence. Its Brier loss is about .0830. The full-history spread model uses economic training history from April 1953 and scores .0848 on these same dates. Paired A and B instead share music-eligible training rows from January 1973; they score .0904 and .1202. That equal-training contrast isolates the addition of the six features. The original broad-history .094921 score uses 630 dates, so it is not a direct comparator here. Read the bar endpoints: the observed music penalty is about .0299. These are saved-result comparisons, without new model fitting in final delivery.

Evidence: [docs/results/closeout_verification.json](../../docs/results/closeout_verification.json), [reports/baseline_backtest/forecasts.csv](../../reports/baseline_backtest/forecasts.csv).

## 08 · Related samples, not independent confirmations

Target **7:05–8:05** (60 seconds).

The stricter coverage samples contain 102 and 45 forecasts, with nine and four positive months. Their dates are subsets of the primary dates, reaching two and one recession episodes rather than three. The same direction of the loss difference is therefore not three independent confirmations. Average precision is especially useful to describe ranking with rare positives. Here it is a stepwise threshold-weighted metric, with ties grouped, not trapezoidal PR area. Primary A and B score .1688 and .0722, compared with prevalence .0820 as a constant-score reference. AP and ROC-AUC vary sharply across the stricter samples because the evaluated episodes and dates change. We should compare A and B within each row, without treating cross-row increases as an isolated coverage effect.

Evidence: [docs/results/paired_inference.json](../../docs/results/paired_inference.json), [docs/results/closeout_verification.json](../../docs/results/closeout_verification.json).

## 09 · Uncertainty depends on analysis settings

Target **8:05–9:20** (75 seconds).

The point estimate stays positive, but uncertainty changes with the dependence assumption. Keefer's implementation samples circular blocks of six, twelve or twenty-four calendar months, resampling the observation mask together with losses. The twelve- and twenty-four-month percentile intervals include zero. Approximate Student-t HAC p-values also move across .05 as the lag changes. We report every existing setting rather than selecting a favorable one. Circular endpoints impose artificial adjacency, and a thirty-year gappy sample is not obviously stationary. Only three recession episodes limit the approximations. This analysis conditions on fixed predictions and selected dates; it does not refit models or propagate source matching, classifier measurement, policy selection or economic-vintage uncertainty. The target is one future month, so an overlapping six-month outcome-window rationale would be incorrect. Pause to read which intervals cross zero.

Evidence: [docs/paired_inference.md](../../docs/paired_inference.md), [docs/results/paired_inference.json](../../docs/results/paired_inference.json).

## 10 · Four limits define what we can claim

Target **9:20–10:15** (55 seconds).

There are four major limits. First, fifteen positive months are only three recession episodes, not fifteen independent downturns. Second, measurement and coverage selection remain unresolved: the corpus passes automated screens, but recording identity and language are not independently verified, and GoEmotions is unvalidated on these lyrics. Third, final economic values and retrospective labels prevent a real-time interpretation. Fourth, historical NLP scoring used a different implementation hash. An earlier local audit found 7,490 shared vectors identical, but 8,800 songs lack a current-hash rerun. We preserve that provenance rather than implying a new full inference run. Small events-per-parameter counts and calibration problems are diagnostics; they do not prove overfitting caused the result.

Evidence: [docs/lyrics_validation.md](../../docs/lyrics_validation.md), [docs/results/exploratory_primary_cov70/CODE_VERSION_PROVENANCE.md](../../docs/results/exploratory_primary_cov70/CODE_VERSION_PROVENANCE.md), [docs/results/closeout_verification.json](../../docs/results/closeout_verification.json).

## 11 · No demonstrated incremental benefit here

Target **10:15–11:00** (45 seconds).

The conclusion is deliberately limited. These six lyric-emotion features show no demonstrated incremental benefit beyond the spread under this selected retrospective design. Their observed loss is higher, but robust statistical degradation is not established. We are not claiming that music contains no economic information, and we have not completed the reviewers' alternative experiments. The project contribution is the implemented pipeline, fair paired comparison, practical same-date references and transparent reproduction boundary. The report, deck and saved artifacts provide the evidence. The most useful next research step would be measurement and historical-availability validation, rather than adding experiments simply to improve a score. Thank you.

Evidence: [docs/final/report.md](../../docs/final/report.md), [docs/final/team_handoff.md](../../docs/final/team_handoff.md).

## A1 · Appendix A · all existing uncertainty settings

Appendix — use only if asked; no scheduled narration.

Use only for a methodological question. The full tables retain all block lengths and HAC lags, with no multiplicity-adjusted confirmatory test. Circular block assumptions and Student-t reference accuracy remain uncertain with few episodes. The 75% sample has percentile intervals excluding zero but lag-12 approximate p above .05; these methods need not agree. Source values are in paired_inference.json.

Evidence: [docs/results/paired_inference.json](../../docs/results/paired_inference.json), [src/music_recession/inference.py](../../src/music_recession/inference.py).

## A2 · Appendix B · scores and descriptive diagnostics

Appendix — use only if asked; no scheduled narration.

These diagnostics describe the saved experiment and do not establish why B performed worse. The events-per-parameter denominator includes the intercept and is a training quantity, distinct from fifteen positive evaluation months. The sparse cross-product ratio is normalized by global variance, is unbounded, and is not used as a HAC input or an independent-month count.

Evidence: [docs/results/closeout_verification.json](../../docs/results/closeout_verification.json), [docs/results/paired_inference.json](../../docs/results/paired_inference.json).

## A3 · Appendix C · three reproduction levels

Appendix — use only if asked; no scheduled narration.

Use the documented fresh-output commands in the report and final README. Historical manifests and NLP implementation hashes remain unchanged. The earlier cache audit is not repeated by this deliverables work. The final validation record reports only content/export checks actually run, distinguishing them from the prior 215-test numerical validation. No raw lyrics or model caches are included in these deliverables.

Evidence: [docs/closeout_validation.md](../../docs/closeout_validation.md), [docs/paired_inference.md](../../docs/paired_inference.md), [docs/final/README.md](../../docs/final/README.md).

## A4 · Appendix D · CRISP-DM and justified decisions

Appendix — use only if asked; no scheduled narration.

This mapping addresses the supplied syllabus expectations without adding techniques mechanically. Cleaning and dimensionality decisions are documented. Random splitting would violate chronology. PCA would change the interpretation of six preselected emotions. Alternative estimators and new horizons would be separately designed experiments, not undocumented repairs selected after observing the result.

Evidence: [docs/final/report.md](../../docs/final/report.md), [docs/review_response.md](../../docs/review_response.md).

## A5 · Appendix E · references and TA clarification

Appendix — use only if asked; no scheduled narration.

The exact clarification in the team handoff separates continuous predictor, binary target and probability output. If the TA intended a different predictor or target, ask for a precise definition and timing before modifying the experiment. References and clickable evidence links are provided in report.md section seven. Source use remains exploratory coursework guidance, not verified rights clearance.

Evidence: [docs/final/report.md](../../docs/final/report.md), [docs/final/team_handoff.md](../../docs/final/team_handoff.md).

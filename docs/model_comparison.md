# Model A/B execution and current eligibility blocker

The comparison runner is implemented in `src/music_recession/music_model.py`. It
reuses economic loading, Model A fitting, calendar eligibility, Brier Score,
ROC-AUC and calibration functions. The validated Model A files and original
results remain unchanged. The existing preparation policy and review registry
are also unchanged; this implementation does not invent approvals.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m music_recession.backtest --output-dir reports/model_comparison/baseline_new_run
.venv/bin/python -m music_recession.music_model --output-dir reports/model_comparison/primary_new_run
```

Use a fresh directory for each comparison. Exit 0 means a completed comparison;
exit 2 means no common evaluation or recorded fit failures; exit 1 means an
aborted run. Do not interpret an earlier output as a successful new run.

The actual local-data execution on 2026-10-18 returned exit 2 with
`blocked_no_policy_eligible_lyrics`. No model weights were downloaded and no
real lyrics were scored. The optional inference dependencies were not installed.
After eligible sources exist, install them with:

```sh
.venv/bin/python -m pip install -e '.[music]' -c requirements-baseline.txt
```

Run the same comparison command afterward. It validates all original audit input
hashes, rebuilds reviewed source mappings, replays source locators, scores eligible
text locally, aggregates monthly features, and fits both models on paired windows.
Raw text stays under ignored `data/raw/lyrics/`. Model weights use ignored
`data/raw/models/`; the emotion cache contains scores and provenance only.

## Eligibility is not text availability

`config/music_preparation.json` currently permits only the weekly-chart OSF
source for the primary analysis. `config/music_recording_reviews.csv` has no
data rows. Its fields require a selected source reference and exact lyric hash,
named reviewer, recording evidence, language and language evidence, permission
evidence, release date and release evidence. The existing selection code rejects
flagged/ambiguous sources, unresolved aliases, instrumentals, compound singles,
concurrent recordings and conflicting clean text variants. Only reviewed English
records can enter this English emotion model. Instrumentals and missing text are
never assigned zero emotion vectors. Unsupported languages are excluded; no
translation or multilingual inference is claimed.

The selected OSF subset has 19,096 provisional identities and 91.9060% of all
observed Billboard popularity weight. This is distinct from the prior multi-source
availability audit's 20,298 identities and 96.8795%. Neither number represents
approved NLP coverage. All current policy-eligible counts are zero.

The recovery files reconcile to 925 candidates (812 LRCLIB, 113 NetEase),
including 616 promising metadata matches, 189 needing review and 120 high risk.
Of the promising records, 600 passed preliminary quality checks. All 925 have
empty recording- and rights-verification fields; none enters this experiment.
Their files exist, which establishes access, not permission or correctness.

## Emotion inference

`config/music_model.json` pins
`SamLowe/roberta-base-go_emotions` at commit
`d75048347613a25d77de8cf6412eaae9fa7b26be`. Its
[model card](https://huggingface.co/SamLowe/roberta-base-go_emotions) describes
English, multilabel classification trained on Reddit GoEmotions with 28 outputs.
We retain sigmoid probabilities for joy, sadness, fear, anger, nervousness and
optimism. We do not apply softmax or threshold labels. The model card's MIT label
does not authorize use of unrelated song lyrics. Reported nervousness performance
is weak, and Reddit performance does not validate lyric-domain accuracy.

Long lyrics are tokenized without truncation, divided into consecutive,
nonoverlapping chunks of at most 480 content tokens, scored with the frozen model,
and pooled by content-token count. Special tokens have no pooling weight. Repeated
lyric lines are retained. This pooling is an explicit modeling choice requiring
domain validation, not a proven document-level emotion estimator.

CPU inference uses evaluation mode and deterministic operations. Cache keys include
song ID, source reference, exact original-text SHA-256, pinned model revision,
configuration, inference implementation hash and package versions. A changed source,
model or implementation invalidates the cache. Cached text is never written.
The real model adapter remains unverified against actual eligible lyrics: current
tests exercise its chunking and pooling with a synthetic backend only.

## Monthly construction and timing

Every weekly chart observation belongs to its chart-issue calendar month. A song's
monthly weight is the sum of `101 - this_week` over its eligible appearances.
Repeated weeks therefore contribute repeatedly while the song/source is scored once.
Each observed chart week has equal exposure; no interpolation across month boundaries
or use of future peak ranks is performed. The emotion mean divides the weighted
sum by scored weight; coverage divides scored weight by **all** observed rank weight,
including unmatched and excluded songs. There is no imputation.

Observations must be known by day 15 of t+1 under the configured publication-delay
convention. Source chart exemplars cannot postdate that cutoff; reviewed release
dates cannot postdate contributing chart observations. Incomplete chart months and
missing rank slots are excluded. The existing 90% eligible-weight threshold is
preserved, not selected using model scores. December 1976 through March 1977 have
13 missing rank slots in total. Excluding those months is explicit, not filling
them with invented entries.

Year-end sources are excluded from the primary mapping. The universe is weekly
Billboard entries, never year-end membership. Corpus curation, final economic
values, NBER labels and the modern pretrained language model still contain
historical hindsight. This is retrospective pseudo-out-of-sample research, not
proof of a historically deployable forecasting strategy. Source publication and
recording vintages remain limitations even after permission is documented.

## Paired evaluation

The analysis bounds refer to predictor months January 1973–December 2023; their
labels extend July 1973–June 2024. Both models train on exactly the same feature-
eligible months, with target month at most t-1, and require 240 actual eligible
training observations. Both use the same six-month labels and expanding windows.
With uninterrupted eligible music starting January 1973, the first forecast would
be July 1993 for January 1994, with 366 possible forecasts through December 2023.
Missing eligible months move that start later. These are hypothetical bounds;
the current actual evaluation count is zero.

Model A calls the existing spread-only Probit directly. Model B fits an intercept,
spread and six emotions using statsmodels Probit. It learns means and population
standard deviations exclusively from each training window. Reports include original-
scale and standardized coefficients, scaling parameters, counts, date boundaries,
training-month hashes and convergence diagnostics. A multivariate linear program
checks separation. Constant/collinear inputs, single-class labels, separation,
numerical warnings and optimization failures are explicit failures; there is no
regularization, logistic substitution or automatic feature deletion.

Scores and calibration use only paired successful forecasts. Failed months remain
in `predictions.csv` and the metrics failure list; a failed B fit cannot inflate A's
comparison sample. Brier difference is B minus A, so a negative value favors music.
No significance or causal claim follows from point estimates.

## Outputs and shortest path

`reports/model_comparison/primary_2026_10_18_final/` is the current local-data run.
It contains source preparation/provenance, monthly coverage and feature columns,
forecast-status rows, empty calibration with headers, metrics, manifest and a
coverage plot. There is deliberately no two-model probability plot when neither
model has a paired evaluation probability. The full baseline verification and its
plots are in `baseline_verification_2026_10_18/`; its score is not a Model B comparator.

The shortest path is to establish a documented source-use basis for the existing
OSF corpus and fill the recording/language/release evidence for selected records.
A genuine shared corpus-level permission can be referenced by multiple records if
it covers them; private storage or an uploader's license label is not substituted
for that evidence. The metadata-only mapping contains exact locators and hashes,
so recollecting the missing songs is not necessary. Freeze eligibility before
examining forecast scores, then rerun the same command. If the fixed coverage
threshold leaves fewer than 240 training months, report that limitation rather
than tuning the period or threshold to obtain a favorable score.

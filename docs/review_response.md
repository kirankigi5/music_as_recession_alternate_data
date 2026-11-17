# PR #1 closeout: review dispositions

This release closes correctness, documentation and derived-asset issues on the
original six-month experiment. It preserves predictions, original model/config
files, historical manifests and the NLP implementation. Uncertainty and PR-AUC
remain in [Keefer's PR #2](https://github.com/kirankigi5/music_as_recession_alternate_data/pull/2),
inspected at `3e8ddc6109c76747a1ef1bf1b086ad6c227044c2`.

Review sources: [Keefer](https://github.com/kirankigi5/music_as_recession_alternate_data/pull/1#issuecomment-6301281910)
and [Nathan](https://github.com/kirankigi5/music_as_recession_alternate_data/pull/1#pullrequestreview-5592188553).

## Disposition of substantive feedback

| Reviewer point | Disposition | Evidence/action and remaining limit |
|---|---|---|
| Both models' original scores reproduce | Existing implementation | Saved probabilities reproduce all three Brier/ROC-AUC pairs; the closeout verifier reconstructs probabilities from stored coefficients/features. |
| Pairing and chronological training | Existing implementation | One selected training list feeds both models; date hashes and training counts are rechecked. This does not prove historical information availability. |
| Independent-month significance is overstated | This closeout / PR #2 | Remove significance claims from PR #1. Dependence-aware methods and their assumptions are reviewed in PR #2; neither prior bootstrap convention is canonical here. |
| Training-rate forecast beats primary paired A | This closeout | Verified Brier 0.0829708 versus A 0.0903819 and B 0.1202486 on identical 183 dates. |
| Claimed −0.089 skill over all 1993–2023 months | Unresolved | The primary paired sample has that approximate skill, but the separately described all-month experiment was not reproduced. PR #2 owns skill reporting; do not equate these populations. |
| Full-history A is stronger | This closeout | Select the same 183 dates/outcomes from existing baseline forecasts: 0.0847541. Distinguish its training history from controlled paired A/B. |
| Full-history comparison p = 0.002 | PR #2 / unresolved | Not independently verified here; requires exact method and outputs. Not adopted into PR #1. |
| t−1 is not a historical no-lookahead guarantee | This closeout | Document the assumed training-target boundary, NBER announcement/vintage gaps, and retrospective pseudo-out-of-sample scope. |
| Alternative 12-month label-lag scores | Unresolved | Reviewer-reported without supplied rerun artifacts; no label-lag change in this release. |
| Scaling does not change an unpenalized Probit's ideal predictions | This closeout | Explain invertible reparameterization and numerical conditioning; preserve correct training-only implementation. |
| Thresholds are nested, not independent confirmation | This closeout | Verify nested dates and episode positives 8/5/2, 7/0/2, 4/0/0 for 2001/GFC/COVID. |
| 1999 and 2001 dominate loss | This closeout | Target calendar years account for 79.21% of net extra loss; describe both false alarms and misses, without a causal mechanism claim. |
| Anger is time-trending; per-origin term contributions | This closeout / unresolved | Remove the stable-value interpretation of 2023 and the overfitting diagnosis. The reported per-origin contributions and correlation definitions are not established as causal explanations. |
| Single-emotion, index, ridge and detrending reruns | Unresolved | Reviewer-reported; need code, tuning details, sample definitions and outputs. Do not declare overfitting settled or general absence of signal. |
| Drop COVID | This closeout | Retain the predeclared primary sample. A labeled post-hoc exclusion, calculated without refitting, gives 181 dates/13 positives, A 0.0864971 and B 0.1155202. |
| Unmatched songs are long-tail | Unresolved | The 632-song recovery list is curated. Full candidate-population rank audit is needed before generalizing or weakening coverage caveats. |
| Twelve-month horizon or within-twelve-month target | Unresolved / optional follow-up | Distinct target constructions needing a declared design and TA clarification; not a closeout model change. |
| Continuous targets, new chart entries, genre controls | Unresolved / optional follow-up | Preserve as proposals; no new dataset or model is introduced. |
| Validate 100–200 lyrics / repair featured credits | Unresolved measurement work | Human emotion/recording validation remains unfinished and disclosed. Metadata corroboration alone cannot verify lyric identity. |
| Preregister further work | This closeout | State that subsequent variants need a declared plan and must be labeled post-hoc when applicable. |
| Remove chat residue and tooling anecdotes | This closeout | Removed from the report. |
| Reference committed results | This closeout | Use `docs/results/`; distinguish unbundled large intermediates. |
| Reconcile 433/352/295 versus 429/348/291 | This closeout | Four months with missing rank slots pass coverage alone but fail complete preparation. |
| Unavailable provenance script | This closeout | Supported instructions do not require it. Original provenance notes remain archival and byte-preserved; the local-cache limit is explained separately. |
| Missing sensitivity features/manifests | This closeout | Restore the four original derived files unchanged; verify their recorded hashes. |
| Shorten the presentation-facing report | This closeout / final delivery | Concise professor README provided; retain detailed report as archival evidence. Final timed slides remain a deliverable. |
| TA: use yield difference to find probability | Existing implementation / TA clarification | Continuous GS10 − TB3MS in percentage points already enters Probit; binary USREC is the outcome. Ask whether a spread change or changed target was intended. |

## Release boundary and provenance

The original local uncertainty script, its tests/JSON, the benchmark figure and
progress PDF are preserved in the original workspace. They are not staged into
PR #1. This closeout adds only a descriptive saved-result verifier, not another
bootstrap/HAC/PR-AUC implementation.

The verifier reads existing baseline forecasts for the full-history comparator;
it never refits a model. It verifies the original run output/config/code hashes,
acknowledging the documented NLP hash difference rather than rewriting a manifest.
Only four missing sensitivity assets are restored. Historical manifests may name
unbundled contribution/mapping files and private raw inputs; their absence is not
misrepresented as full inference reproduction.

The local cache audit found 16,290 historical-version entries, 7,490 current-version
entries, and 7,490 identical shared score vectors with matching token/chunk counts.
The other 8,800 songs have not been rerun under the current hash. A clean checkout
can verify the saved derived bundle, not that private cache comparison or the
full raw-to-feature path. See the [validation record](closeout_validation.md).

## PR #2 handoff

After PR #1's head changes, inspect PR #2's base/diff before integrating it. Four
documentation files overlap. Preserve the original prediction/model hashes and
resolve completed-versus-pending wording. Compare circular and non-circular
calendar blocks, endpoint assumptions, seeds/draw counts, normal versus Student-t
reference choices, and sparse diagnostics. Do not select methods by their
significance result. The documented sparse autocovariance ratio can exceed one;
its presentation as an autocorrelation needs explicit qualification. This release
does not resolve or import that implementation.

## Final-deliverable checklist

| Supplied syllabus expectation | Implemented evidence | Delivery status / justified non-use |
|---|---|---|
| Clear problem statement | README; report §1–2 | Incremental six-month recession-status probability forecasting is explicit. |
| Data sources | README; report §4; source manifests | FRED, weekly Billboard and admitted OSF source distinguished from investigated recovery sources. |
| CRISP-DM | README; report §5 and development history | Business/data understanding, preparation, modeling, evaluation and research delivery are mapped; no real-time deployment claimed. |
| Documented preprocessing | Audit, preparation, emotion and aggregation modules; report §5–6 | Identity normalization, chunking, rank weights, joins and calendar alignment documented. |
| Cleaning decisions | Exclusion rules and coverage reports | Missing lyrics are not zero scores; incomplete/ambiguous records remain explicit; no interpolation or translation to manufacture coverage. |
| Dimensionality / feature selection | Fixed six labels; report §6.8 and §8 | No PCA or retrospective feature search: six interpretable preselected inputs preserve the research question. Complexity is disclosed; regularization is not added mechanically after seeing loss. |
| Predictive models | Baseline and music Probit modules | Continuous spread, binary target and probabilistic outputs are explicit. Two directly comparable models answer the incremental-value question; more algorithms are not required for a useful negative result. |
| Evaluation/results | Saved predictions, Brier, ROC-AUC, calibration and closeout verifier | Original paired samples and practical benchmarks distinguished; uncertainty/PR-AUC pending separate PR #2. |
| Implemented architecture | Pipeline figure and source-module map | Raw-to-audit-to-features-to-model-to-evaluation code exists; strict versus exploratory evidence gates are explicit. |
| Final presentation/report | Professor README, full report, ten existing figures | Final slide deck and timed rehearsal remain. Confirm official date, Q&A inclusion and submission format; December 3 / 12:30 are supplied planning constraints. |

A negative finding does not justify changing targets or adding techniques simply
to obtain a better score. Remaining measurement and statistical work should be
scoped explicitly in the final presentation.

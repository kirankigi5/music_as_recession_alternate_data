# Team handoff

**Short version:** the experiment is finished. Lyric emotions made our
six-month-ahead recession forecasts **worse**, not better, at all three coverage
thresholds we committed to in advance. Everything is documented, every figure is
regenerable in seconds, and nothing needs re-running.

Full detail: [`COMPLETE_RESEARCH_REPORT.md`](COMPLETE_RESEARCH_REPORT.md).

---

## 1. What the project does

We test whether emotions in Billboard Hot 100 song lyrics improve predictions of
whether the U.S. will be in recession **six months from now**, compared with the
standard yield-curve predictor.

Two Probit models, fitted on **identical** training rows and scored on
**identical** forecast dates:

- **Model A** — 10-year minus 3-month Treasury spread only
- **Model B** — the same spread **plus** six monthly lyric-emotion features
  (joy, sadness, fear, anger, nervousness, optimism)

The question is whether Model B beats Model A out of sample. It does not.

---

## 2. Which parts are finished

| Component | Status |
|---|---|
| Economic data preparation (FRED → monthly table) | **Done** |
| Model A baseline + 630-forecast backtest | **Done** |
| Billboard universe, 21,542 identities, 1973–2023 | **Done** |
| Lyrics corpus audit and OSF evidence screening | **Done** |
| Coverage feasibility analysis at 5 thresholds | **Done** |
| MusicBrainz metadata pilots (2 × 40 records) | **Done**, full 2,255 screen deliberately not run |
| Synthetic end-to-end pipeline validation | **Done** |
| **Real GoEmotions inference on 16,290 songs** | **Done** |
| **70% primary + 75% / 80% sensitivity comparisons** | **Done** |
| Calendar-block bootstrap, Diebold–Mariano (HAC), PR-AUC | **Done**, from committed predictions |
| Figures, report, handoff | **Done** |
| Test suite | **208 passing** |
| Strict-policy Model B result | **Still blocked by design** — 0 eligible records |

---

## 3. Where the source code lives

```
src/music_recession/
  economic.py              FRED snapshots -> data/processed/economic_monthly.csv
  baseline.py              Model A Probit; select_training() is the eligibility rule
  backtest.py              expanding-window loop; brier_score, roc_auc, calibration
  lyrics_audit.py          offline corpus audit; writes locators + hashes, no text
  music_preparation.py     identity mapping, monthly aggregation, BOTH policies
  emotions.py              GoEmotions adapter, score cache, eligibility gates
  music_model.py           Model B fit, paired evaluate(), plots
  inference.py             calendar-block bootstrap, Diebold–Mariano/HAC, PR-AUC
  recording_metadata.py    MusicBrainz corroboration logic

scripts/
  run_exploratory_comparison.py   THE EXPERIMENT DRIVER
  run_paired_inference.py         intervals + tests from predictions.csv, seconds
  make_research_figures.py        rebuilds every figure, no inference
  measure_model_b_readiness.py    coverage feasibility, blind to performance
  audit_osf_evidence.py           per-candidate evidence screening
  match_recording_metadata.py     MusicBrainz triage

config/
  baseline.json                        Model A timing contract (240 / 6 / 15th of t+1)
  music_model.json                     pinned GoEmotions model + the six emotions
  music_preparation.json               STRICT policy, 0.90 coverage
  music_preparation_exploratory.json   EXPLORATORY policy, 0.70 coverage
```

The two most important functions to read first are `select_training` in
`baseline.py` (the no-lookahead rule) and `evaluate` in `music_model.py` (how the
pairing is enforced).

---

## 4. Where the experiment results live

**Committed, viewable on GitHub:**

- [`results/exploratory_primary_cov70/`](results/exploratory_primary_cov70/) — the
  primary run: `metrics.json`, `predictions.csv`, `monthly_coverage.csv`,
  `monthly_features_and_coverage.csv`, `calibration.csv`, `manifest.json`
- [`results/exploratory_sensitivity_cov75/`](results/exploratory_sensitivity_cov75/)
  and [`results/exploratory_sensitivity_cov80/`](results/exploratory_sensitivity_cov80/)
- [`figures/music_recession/`](figures/music_recession/) — 10 figures plus
  `results_summary.json`
- [`results/paired_inference.json`](results/paired_inference.json): block-bootstrap
  intervals, Diebold–Mariano tests, PR-AUC and skill against the base rate for all
  three runs

**Local only, not committed** (too large or unnecessary):

| File | Size | Why not committed | How to regenerate |
|---|---|---|---|
| `reports/model_comparison/*/monthly_contributions.csv` | 18 MB each | per-song-month rows, large | re-run the driver (Section 9) |
| `reports/model_comparison/*/song_source_mapping.csv` | 13 MB each | per-song locators, large | re-run the driver |
| `reports/osf_source_review/run_004/candidate_evidence.csv` | 17 MB | large intermediate | `scripts/audit_osf_evidence.py` |
| `data/raw/lyrics/` | 1.26 GB | **copyrighted lyric text** | obtain from [osf.io/2k7ut](https://osf.io/2k7ut/) |
| `data/raw/models/` | ~500 MB | downloaded model weights | auto-downloaded on first inference |
| `.agents/emotion_cache_explore/` | 16,290 files | private score cache | re-run inference (~50 min) |

The two in-run PNGs (`lyric_coverage.png`, `probability_comparison.png`) also stay
local; Figures 3 and 6 in `figures/music_recession/` are improved versions.

---

## 5. How the monthly music features are built

Three ideas, in order:

**1. Popularity weight.** Each weekly chart appearance gets `weight = 101 − rank`.
A #1 contributes 100, a #100 contributes 1.

**2. A song charting for several weeks contributes once per week**, each with its
own weight. A song six weeks at #1 shaped that month more than one week at #90.

**3. Weighted mean per emotion:**

```
E(month, emotion) = Σ(weight × song_score) / Σ(weight of songs with usable lyrics)
```

**Coverage uses a different denominator** — all observed chart weight, *including*
songs with no usable lyrics:

```
coverage(month) = Σ(eligible weight) / Σ(all observed weight)
```

That asymmetry is deliberate. The feature averages over what we measured; coverage
honestly reports how much we missed. Songs without lyrics are **never** scored as
zero, because a zero vector would falsely assert "no joy, no sadness, …" and drag
the monthly average down as an artefact of missing data.

Each song is scored **once** by GoEmotions and cached by lyric content hash, then
reused for every chart appearance.

---

## 6. Key research findings

| Coverage | Forecasts | Recession months | Model A Brier | Model B Brier | B − A | A AUC | B AUC |
|---|---|---|---|---|---|---|---|
| **70% primary** | **183** | **15** | **0.0903818731** | **0.1202485779** | **+0.0298667048** | 0.6462 | **0.4306** |
| 75% | 102 | 9 | 0.0739858409 | 0.1146495991 | +0.0406637582 | 0.8722 | 0.5424 |
| 80% | 45 | 4 | 0.0640448014 | 0.1206001394 | +0.0565553380 | 0.9878 | 0.7012 |

Brier is a loss, so **positive B − A means Model B is worse**. Zero fit failures
anywhere.

Six things worth knowing beyond the table:

1. **Model B's AUC of 0.4306 is below chance** at the primary threshold.
2. **Model B fails hardest exactly when it matters.** Mean per-month penalty is
   **+0.1713** on the 15 recession months versus **+0.0172** on the 168 quiet
   months.
3. **It is better in a bare majority of months** (97 of 183) yet loses badly on
   average — its losses are much larger than its gains. Classic overfitting
   signature.
4. **Its one good period is 2023**, where it correctly damped a Model A false alarm
   on the inverted curve.
5. **The evidence is thinner than 183 forecasts suggests.** Allowing for serial
   dependence, the primary 95% interval on B − A is [−0.0010, +0.0614] with
   12-month calendar blocks, and the Diebold–Mariano p-value is 0.04 to 0.06
   depending on the HAC lag. The 183 forecasts carry about as much information as
   50 independent months. The 75% and 80% runs reuse a subset of the 70% months and
   reach two and one recession episodes, so they do not independently confirm the
   result. Quote these intervals, not the i.i.d. [+0.0147, +0.0458].
6. **Model A does not beat the base rate in the primary window either.** Against a
   forecast that repeats the training recession share, its Brier skill is −0.089
   (Model B −0.449). PR-AUC is 0.169 for A and 0.072 for B, against a no-skill
   level of 0.082.

Tables for all three runs are in
[`exploratory_results.md`](exploratory_results.md#how-certain-is-the-difference).

---

## 7. Important limitations

- **Three recession episodes only** (2001, 2008–09, 2020). The corpus starts 1973
  and 240 training observations are required, so this ceiling cannot be raised
  within the current design.
- **Coverage collapses over time**: median 0.924 in the 1980s to 0.608 in the
  2020s. Only **33 of 120** months in the 2010s are eligible, and **10 of 48** in
  the 2020s. Coverage drift is confounded with any emotion effect.
- **The cause is known**: the upstream lookup stripped featured-artist credits for
  2,309 songs (11.62% of chart weight), concentrated in later decades.
- **No record is verified.** Automated screens only: `recording_verified=False`,
  no human-reviewed language, no verified release dates.
- **Rights are unresolved.** Source use rests on reported instructor guidance for a
  course project — **not** a licence or clearance.
- **GoEmotions is Reddit-trained** and applied to lyrics out of domain. The planned
  manual validation of its lyric predictions was never done.
- **8 parameters against 46–61 positive training labels** (EPV ≈ 5.8–9.2 against a
  conventional minimum of 10).
- **Inference is weak.** Roughly 50 effective independent months in the primary
  run and at most three recession episodes (finding 5 above).

**This study does not show music is economically useless.** It shows that *this*
design, on *this* corpus, with *this* emotion model, did not beat the yield curve.

---

## 8. What to work on next

Ranked by value. Full reasoning in Section 17 of the research report.

1. **Fix the featured-credit coverage gap.** Highest value by far. Would roughly
   double the forecast count, put the full 2008–09 recession in the window, and
   largely remove the coverage confound. Start by stripping Billboard's
   `(From "...")` parentheticals before querying MusicBrainz, then review the top
   322 records by weight — that recovers 50% of flagged weight.
2. **Cut the predictor count.** Test a single predeclared sentiment index
   (`sadness+fear+anger+nervousness − joy−optimism`), taking Model B to 3
   parameters. Directly tests the overfitting explanation.
3. **Regularised Probit** (ridge / elastic net / Firth), penalty chosen within each
   training window.
4. **Lagged and differenced features** — we only used contemporaneous month *t*.
5. **Validate GoEmotions on lyrics**, or fine-tune. Right now we cannot distinguish
   "lyric emotion does not predict" from "GoEmotions does not measure lyric
   emotion".
6. ~~Block bootstrap / Diebold–Mariano with HAC errors~~ **Done.** Score any new
   variant from items 2 to 4 the same way with
   `scripts/run_paired_inference.py --run-dir docs/results/exploratory_primary_cov70
   --run-dir <new run dir> --out <file>.json`.
7. **Extend back to 1959–1972** to add three more recession episodes.

---

## 9. How to avoid accidentally re-running expensive inference

**The full run costs ~50 minutes of CPU inference. You almost certainly do not
need it.**

Rules:

- **To rebuild figures or re-read results, never run the driver.** Use
  `scripts/make_research_figures.py`, which touches no model.
- **Do not edit `src/music_recession/emotions.py`.** Its file hash is part of the
  emotion cache key, so *any* edit — even a comment — invalidates all 16,290
  cached scores and forces a full re-score. This happened to us once.
- **Always pass the same `--cache-dir`.** With a warm cache the sensitivity runs
  finish in under a minute at 100% cache hits.
- **`--output-dir` must be fresh.** The driver refuses to overwrite, which protects
  existing results.
- **Use `--limit N` for smoke tests**, which scores only the first N eligible songs.
- Interrupted runs resume safely: cache writes are atomic, and re-running the same
  command picks up where it stopped.

---

## 10. Reproducing figures and metrics from derived files

No raw lyrics, no model weights, no network needed.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .          # numpy, scipy, statsmodels, matplotlib
.venv/bin/python -m pytest -q                 # expect 208 passed
.venv/bin/python scripts/make_research_figures.py
```

Figures land in `docs/figures/music_recession/`. The script reads only derived CSV
and JSON under `reports/model_comparison/exploratory_*/`.

To verify the headline numbers without running anything:

```bash
.venv/bin/python -c "
import json
for r in ('exploratory_primary_cov70','exploratory_sensitivity_cov75','exploratory_sensitivity_cov80'):
    m=json.load(open('docs/results/%s/metrics.json'%r))
    print(r, m['evaluated_months'],'forecasts |',
          'A',round(m['model_a']['brier_score'],7),
          'B',round(m['model_b']['brier_score'],7),
          'diff %+.7f'%m['brier_difference_b_minus_a'])
"
```

---

## 11. Directory map

```
.
├── README.md                     start here
├── docs/
│   ├── COMPLETE_RESEARCH_REPORT.md   <- the full 18-section report
│   ├── TEAM_HANDOFF.md              <- this file
│   ├── exploratory_results.md        concise results summary
│   ├── exploratory_course_policy.md  what the exploratory policy admits/refuses
│   ├── osf_source_use_review.md      rights analysis, 5 open questions
│   ├── forecast_protocol.md          Model A timing contract
│   ├── lyrics_validation.md          corpus audit protocol
│   ├── model_comparison.md           Model B execution requirements
│   ├── figures/music_recession/      10 committed figures
│   └── results/                      committed metrics, predictions, coverage
├── src/music_recession/          all model + pipeline code
├── scripts/                      runnable entry points
├── tests/                        208 tests
├── config/                       policies and model pins
├── data/processed/               economic_monthly.csv (tracked)
├── data/raw/                     GITIGNORED: lyrics, model weights
└── reports/                      generated; large files stay local
```

---

## 12. Experiment review checklist

Work through this to satisfy yourself the result is sound.

**Data and eligibility**

- [ ] `billboard_hot100_all.csv` covers 1973–2023 and yields 21,542 identities
- [ ] Coverage denominator includes chart entries with no usable lyrics
      (`monthly_coverage.csv`: compare `scored_weight` with `total_weight`)
- [ ] `recording_verified` is `False` for every record under the exploratory policy
- [ ] `release_date` is empty under the exploratory policy, never a chart date
- [ ] Thresholds 70/75/80 were predeclared before any score — see
      `exploratory_course_policy.md` and the performance-blind
      `reports/model_b_readiness/run_002/`

**Timing and leakage**

- [ ] `target_month = predictor_month + 6` calendar months in `predictions.csv`
- [ ] `forecast_cutoff` is the 15th of `predictor_month + 1`
- [ ] `training_target_end ≤ predictor_month − 1` for every row
- [ ] `training_end < predictor_month` for every row
- [ ] `training_count ≥ 240` for every successful forecast
- [ ] Model B feature standardisation uses training rows only (asserted in
      `tests/test_integration_model_comparison.py`)

**Pairing and metrics**

- [ ] Both models have a probability on exactly the same months
- [ ] `actual_label` is identical for both models by construction
- [ ] Brier recomputes from `predictions.csv` to the value in `metrics.json`
- [ ] `fit_failures` is empty in all three runs
- [ ] Model A was re-scored on Model B's dates — the 0.094921 broad-period figure
      is **not** used as a comparison

**Honesty checks**

- [ ] Figures 4 and 5 are not read left-to-right across thresholds
- [ ] Figure 6's lines break where no eligible month exists
- [ ] Intervals quoted anywhere are the calendar-block ones from
      `docs/results/paired_inference.json`, not the i.i.d. [+0.0147, +0.0458]
- [ ] The 75% and 80% months are subsets of the 70% months
      (`forecast_months_also_in_primary` in the same file), so sign agreement
      across thresholds is not presented as three independent checks
- [ ] `CODE_VERSION_PROVENANCE.md` is read: 7,490 of 16,290 songs verified
      identical across the two `emotions.py` hashes, 8,800 not recomputed
- [ ] No raw lyric text anywhere in the repository

---

*Questions about a specific number: check `docs/results/*/metrics.json` first,
then the matching section of `COMPLETE_RESEARCH_REPORT.md`. Every number in the
report is traceable to a saved file.*

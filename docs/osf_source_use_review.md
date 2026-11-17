# OSF source-use and candidate evidence review

Reviewed 2026-10-18 for NYU Predictive Analytics Team 5. Scope: local,
noncommercial, quantitative analysis of already collected lyrics; frozen-model
emotion inference and recession modeling; no lyric redistribution, external text
upload, lyric generation, model fine-tuning or new scraping.

**Decision: keep the current Model B eligibility gate intact.** Provenance is now
verified against OSF's published file hash. There is affirmative evidence of an
open project license and a plausible fair-use basis for the narrow analytical use.
Neither establishes a verified recording or release date, and the evidence does
not establish the depositors' authority to license all underlying lyrics. The
fair-use assessment below is a documented preliminary analysis, not an invented
rightsholder permission, adopted team decision or NYU legal approval.

## Verified provenance and published materials

The [paper](https://www.nature.com/articles/s41598-025-28327-5) reports 20,186
English-language lyrics after exclusions, collected using title/artist queries
to lyrics.ovh and Genius. It links [OSF project 2k7ut](https://osf.io/2k7ut/) for
data and code. Its publication date is December 11, 2025. Its article license is
**CC BY-NC-ND 4.0**, distinct from the OSF project license. The article's availability
statement is evidence of public deposit, not a statement of publisher permission
for every lyric.

The [OSF project API](https://api.osf.io/v2/nodes/2k7ut/) reports a public project
created in June 2024, with a 2025 project-license year and an empty copyright-holder
string. The [storage listing](https://api.osf.io/v2/nodes/2k7ut/files/osfstorage/)
contains eight files and no further page. Only OSF Storage is attached. No child
projects or wiki entries were returned. **No standalone README was found.** The
analysis notebook's introductory markdown supplies a title, contents, preregistration
link and package versions; it contains no additional lyric-use permission.

The local `data/raw/lyrics/osf_lyrics.csv` is byte-identical to OSF's
[`lyrics_dataset_LIWC_Vader_Lz77_individual_songs.csv`, version 1](https://osf.io/2k7ut/files/osfstorage/682c61225c1001cde3eac575),
deposited May 20, 2025:

```text
bytes: 1256141520
SHA-256: 7820fd555b22a56a33dd128790e5c6d5414d4706919c02340a00d74213e1cc88
```

This verifies the supplied file's provenance, not historical release availability
or recording accuracy. The corpus has 266,087 chart rows, 21,531 exact title/artist
pairs and 20,187 pairs with a nonempty `lyrics_raw` field. Whitespace/case folding
does not resolve the one-record difference from the paper's 20,186 count. There
are 19,977 distinct nonempty lyric hashes; repeated hashes must not automatically
merge covers or recordings. The paper-versus-file count difference remains open.

## Actual reuse terms and their scope

The project's [linked license record](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/)
is **Creative Commons Attribution 4.0 International**, not an absent license. Its
[legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) permits
reproduction and adaptation within rights the licensor controls. Shared licensed
material requires appropriate attribution, license information and modification
notices; no endorsement is implied. It also addresses database rights and preserves
applicable exceptions such as fair use. The license does not warrant ownership or
non-infringement and cannot grant rights its licensor lacks.

This is meaningful reuse evidence for the authors' own code, analysis and applicable
compilation rights. The examined materials do not provide lyric-publisher licenses,
a chain of rights for lyrics obtained from third parties, a scoped permission for
downstream text mining, or a study-specific fair-use/legal assessment. No explicit
prohibition on private academic analysis was found either. The article's separate
license must not be substituted for the dataset's terms.

Attribution for any eventual report should include Foramitti, Nater, Lamm and
Martins (2025), DOI `10.1038/s41598-025-28327-5`, OSF `2k7ut`, the exact file/version
and hash, the OSF CC BY 4.0 declaration, and our modifications. That attribution
does not resolve third-party lyric rights. Keep full text and third-party analysis
outputs such as LIWC-derived columns out of public artifacts unless separately
cleared. This project uses `lyrics_raw`, not those prior sentiment features.

## Documented preliminary fair-use assessment

U.S. [17 USC 107](https://www.copyright.gov/title17/92chap1.html#107) requires a
fact-specific four-factor assessment. The following applies only to the stated
Team 5 workflow, not public lyric distribution or a general license for this corpus.

| Factor | Assessment for this workflow | Remaining uncertainty |
|---|---|---|
| Purpose and character | Private, noncommercial academic analysis produces numerical emotion features and forecasts rather than substituting for reading/singing lyrics. This favors the proposed analytical use. | Confirm the final workflow remains within this scope, including any shared caches or future publication. |
| Nature | Lyrics are published creative works. Their expressive character weighs against unrestricted copying. | The status and ownership of each work were not independently researched. |
| Amount | Full text is useful to measure each song's emotion profile; arbitrary excerpts could distort the research. Copies are already local and no text is reproduced in results. | Document why whole-text processing is necessary and retain only what the project needs. |
| Market effect | Numeric, non-reconstructive outputs are not lyric substitutes. Restricted access and no text exports limit substitution and dissemination risks. | The relevant lyric-text-mining licensing market and any source contracts were not established. Their existence alone does not settle fair use, but possible effects need consideration. |

The Copyright Office's summaries of
[Authors Guild v. Google](https://www.copyright.gov/fair-use/summaries/authorsguild-google-2dcir2015.pdf)
and [Authors Guild v. HathiTrust](https://www.copyright.gov/fair-use/summaries/authorsguild-hathitrust-2dcir2014.pdf)
provide relevant analogies for transformative computational/search uses and copying
whole works where needed. They are not blanket rulings about this lyric corpus.

**Assessment:** these facts give a credible, favorable argument for the narrow
non-expressive research use. The outstanding source/contract and market facts mean
this review does not certify that every proposed use is covered. The project has
not recorded adoption of a source-use decision based on this assessment. I have
therefore not written `permission_verified=True` or invented an approval reference.

[NYU's permission guidance](https://guides.nyu.edu/copyright/permissions) describes
documenting a fair-use analysis when relying on that exception.
[NYU's fair-use guide](https://guides.nyu.edu/fairuse) cautions that educational use
is not automatically fair use and points to Open Knowledge for questions.
Fair use does **not** categorically require advance rightsholder or institutional
permission; the unresolved item here is the project's documented decision and
remaining factual assessment, not a supposed universal legal approval requirement.

## Published code: identity and language limitations

Inspected, without executing collection code:

- [`lyrics_scraping_OSF.ipynb`](https://osf.io/download/4eb29/), SHA-256
  `70809ef7f79398d976678a9f946c1c58a938b6f97410af42e3c4784664d64a69`.
- [`analysis_lyrics_1404.ipynb`](https://osf.io/download/p5bax/), SHA-256
  `8a35b11e1e5ac12f212ab942d6d2b77ef0e300852a88539619ed3b7b78b542c3`.
- [`crosscorrelation14012025.R`](https://osf.io/download/vjp5f/), SHA-256
  `d7b849c1858da5c0d31910213618bc95710c6e0db571b1bb8622677425e63870`.

The collection notebook selects weekly Billboard entries, deduplicates title/artist
queries, and calls lyrics.ovh. Its query removes featured-artist suffixes beginning
with lowercase ` feat` or capitalized ` Feat`. It saves the returned lyric text
against the original query metadata, not a returned recording identifier. The
Genius fallback sets excluded terms for remix/live results; it also saves text
without the matched recording metadata. Consequently, agreement between OSF
title/artist columns and Billboard establishes **query/slot agreement**, not
independently verified lyric-recording identity.

Language filtering uses `langdetect` for whole-text English detection. The shared
code does not set a detection seed. It contains inconsistent dataframe references
(`df` versus `df6`/`df7`) and an Excel write to a `.csv` filename. These are static
reproducibility concerns; they do not prove how the final authors' run behaved.
We did not execute or repair the upstream scraper. Its analysis notebook uses
whole-sample z-scores/outlier filtering; that procedure is not borrowed for our
forecasting experiment because it would use future sample information.

## Automated evidence and observability

`scripts/audit_osf_evidence.py` independently replays every selected source locator
and content hash, checks normalized full title/artist and Billboard date/rank slots,
and records first chart appearance from the full local chart file plus first
appearance in 1973–2023. Those dates establish **observed chart identity dates**.
They are not recording release dates, verified lyric-publication dates or historical
Billboard publication timestamps. Actual release-date and recording-ID fields are
left blank.

The audit uses isolated `langid==1.1.6`, all 97 supported languages, normalized
model scores, full text and every 100-word block. A short trailing block is joined
to its predecessor. The conservative English screen requires at least 40 words,
whole-text English score >=0.95, at least 90% of words in high-score English blocks,
and no high-score non-English block. These scores are not calibrated accuracy and
the screen can miss multilingual text or flag slang. It records automated evidence,
never `language_verified=True`. See the [language detector implementation](https://github.com/saffsd/langid.py).

Instrumental markers and very low lexical diversity trigger review. Absence of a
marker does not prove vocal content. Version qualifiers, explicitly numbered song
parts and stripped featured credits are additional review flags. Ordinary titles
such as “Live Your Life” are not misclassified as live versions solely because
they contain that word. No duplicate recording is automatically removed.

Evidence is saved separately in
`reports/osf_source_review/run_004/candidate_evidence.csv`; it is not loaded as an
approval registry. `config/music_recording_reviews.csv` and the eligibility policy
remain unchanged. Full-corpus dispositions, exact totals and monthly weighted
coverage are in that directory. The first two runs are diagnostic iterations;
`run_004` is authoritative.

`run_004` replaces `run_003` after a 2026-10-19 reproducibility review of
`scripts/audit_osf_evidence.py`. `candidate_evidence.csv`, `all_song_dispositions.csv`
and `monthly_coverage.csv` are **byte-identical** between the two runs, and every
count below is unchanged. Only `summary.json` differs, because its nested count
dictionaries are now keyed in sorted order instead of first-encounter order, and
`manifest.json` records the new `code_sha256`. The review also replaced three
unhelpful crash paths (bare `StopIteration`, `IndexError` on an empty candidate
set, `relative_to` failure for a mapping outside `--root`) with named errors, and
added a duplicate-chart-identity guard that would otherwise have silently
understated weighted coverage. No screening rule, threshold or policy changed.

### Exact final breakdown

| Mutually exclusive disposition across all 21,542 study identities | Count |
|---|---:|
| Already excluded by the existing primary source mapping | 2,446 |
| Selected OSF candidates now flagged for technical review | 2,806 |
| Selected candidates passing this automated screen, still blocked by evidence policy | 16,290 |
| Eligible under the unchanged Model B policy | 0 |

All 19,096 selected candidates match normalized full query metadata and their
Billboard date/rank slot; all have observed chart identity dates no later than the
selected source chart. For 102, the first chart observation predates 1973. None
has a recording ID or independently established release date supplied by this audit.

| Language screen, mutually exclusive within 19,096 | Count |
|---|---:|
| Consistently English by the automated rule | 18,689 |
| Confident non-English whole text | 50 |
| Mixed/non-English block requires review | 234 |
| Uncertain language | 104 |
| Too short for the rule | 19 |

Overlapping technical flags include 2,309 shortened featured-credit queries,
107 version/numbered-part qualifiers (four explicit remix qualifiers), 58 texts
containing an instrumental marker plus other words, and three very-low-diversity
texts. These counts are not additive. No selected record consists solely of an
instrumental marker; that does not certify the remainder as vocal recordings.

| Coverage numerator / denominator | Weight | Weighted coverage |
|---|---:|---:|
| All observed 1973–2023 chart entries | 13,438,037 | 100% |
| Observable selected OSF candidates | 12,350,366 | 91.9060% |
| Automated English candidates | 12,071,549 | 89.8312% |
| Candidates passing all current automated checks | 10,481,530 | 77.9990% |
| Policy-eligible records | 0 | 0% |

All 612 monthly values are saved. Only **130 months** have at least 90% total
chart weight covered by the fully screened candidate subset, even before applying
chart-completeness exclusions. Thus even approving that entire subset would not
satisfy 240 eligible training months under the current coverage policy. This is a
second blocker, independent of permission. Review of flagged candidates could
recover coverage; alternatively the team could predeclare a different threshold
and sensitivity analyses **before** obtaining forecast scores. No threshold was
changed here and no preferred period was selected from model performance.

The monthly denominator contains all 266,087 observed chart entries, weight
`101 - rank`; every repeat week contributes separately. Source exemplars after a
forecast's 15th-of-next-month cutoff cannot backfill earlier months. No year-end
membership selects the monthly universe. This retrospective audit does not establish
historical publication/vintage availability. Nonrandom exclusions and the modern
language detector remain limitations.

## Exact remaining decision and evidence questions

1. **Source-use decision:** Will the team document reliance on the narrow fair-use
   assessment above, after resolving relevant source-contract/market questions, or
   obtain an applicable permission? NYU Open Knowledge can help assess it; this
   review does not claim NYU has approved it. The decision should identify scope,
   decision-maker, date and evidence, not merely assert “academic use.”
2. **Depositor clarification:** Does the OSF CC BY declaration purport to include
   third-party lyric text, and on what authority or research-use basis? Can the
   authors provide source URLs/returned metadata or the cleaning log, and reconcile
   20,186 reported lyrics versus 20,187 nonempty file identities? No message has
   been sent to the authors.
3. **Recording/language standard:** May a future explicitly provisional primary
   analysis admit metadata-supported, automatically screened records after manual
   validation of a prespecified sample, while excluding ambiguous versions/credits?
   The current policy requires stronger reviewed evidence; this audit has not
   relabeled automatic agreement as recording verification.
4. **Timing standard:** For this retrospective chart-based study, may documented
   first chart observability replace the current requirement for an independently
   verified release date? This can be a defensible protocol change, but the fields
   must stay distinct. Do not fill `release_date` with a chart date to satisfy code.
5. **Coverage standard:** Will flagged featured/version records be investigated
   first, or will the team predeclare a revised minimum coverage policy? The
   conservative automatic subset supplies only 130 months at the current 90%
   threshold, so simply approving its source use cannot meet the 240-month rule.

The shortest path is a documented source-use decision plus a deliberately revised
evidence standard (or completion of the existing stronger evidence requirements),
not collection of another thousand songs. Any revisions must be frozen before
examining predictive scores. No Model B inference/comparison was run in this review;
the existing gate remains intact and the original Model A is unchanged.

## Reproduction

The OSF API snapshots and retrieval hashes are in `reports/osf_source_review/`.
Downloaded notebooks and the article are retained under ignored
`data/raw/osf_source_review/`; no new lyric corpus was downloaded.

```sh
.venv/bin/python -m pip install --no-deps --target data/raw/osf_source_review/python_deps langid==1.1.6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_osf_evidence.py --output reports/osf_source_review/new_run
.venv/bin/python scripts/measure_model_b_readiness.py --output reports/model_b_readiness/new_run
.venv/bin/python -m pytest -q
```

Use a fresh output directory. Model/data hashes, detector version, screening policy
and output hashes are recorded. The original 114 tests were not rewritten; five
synthetic screening tests and seven readiness-measurement tests bring the passing
total to 126. `pytest` and `python -m unittest discover -s tests` now agree, because
the added root `conftest.py` makes `tests/` able to import `scripts.*` regardless of
how the suite is launched.

`scripts/measure_model_b_readiness.py` answers the coverage-standard question in
item 5 below **without** fitting, scoring or importing Model B, so no eligibility
threshold can be chosen by inspecting forecast performance. It writes monthly
coverage, a threshold table and a blocking-axis breakdown to
`reports/model_b_readiness/run_002/`. Its measurement subsets are not approvals.

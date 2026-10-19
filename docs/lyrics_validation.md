# Lyrics validation and source traceability

This audit inspects existing local files without downloading, publishing, or copying lyric text. The prior master inventory, source mappings, scripts, and reports remain intact. The study universe is all weekly Billboard observations dated 1973–2023, independent of whether lyrics exist. The 1959–1972 investigation is outside this run. A title/artist identity is not necessarily a recording.

## Reproduce

From the repository root with the existing environment:

```sh
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m music_recession.lyrics_audit --output reports/lyrics_validation/new_run
```

Use a new output directory; an existing one is rejected. No dependency installation or network access is needed. Required local files are OSF CSV, Kaggle ZIP, LYDIA ZIP, and the tmelo yearly CSV directory under `data/raw/lyrics/`. The audit fails on absent required files rather than reporting them as successfully matched. Original raw files remain ignored by Git.

Each run writes `inventory_audited.csv`, `source_references.csv`, `mapping_issues.csv`, `source_issues.csv`, `manual_review.csv`, `yearly_coverage.csv`, `monthly_coverage.csv`, `summary.json`, and `manifest.json`. The manifest records complete input file hashes, sizes, code hash, and schema findings. Each reference identifies a local file, ZIP member if relevant, 1-based CSV data record (not physical line), source metadata, and SHA-256 of the original decoded lyric field encoded as UTF-8. Repeated identical OSF observations are represented by an exact exemplar reference and occurrence count per year/method/flag group; original rows are never removed. Different hashes remain separate candidates. File hash plus locator makes references reproducible even when URLs disappear.

## Status definitions

- **Provisional accessible:** at least one matching local record has usable-looking text. Metadata normalization, explicit provisional mappings, and OSF date/rank agreement produce candidates, not verified recordings. Known error pages, missing/very short/overlong text, instrumental markers and mismatched OSF chart slots do not count as usable text.
- **Verified recording:** requires independent version/credit/text review. There are no such approvals in this implementation; verified counts remain zero. No automated result promotes a candidate.
- **Unmatched or unusable:** no candidate passing the minimal availability screen. This does not prove lyrics do not exist.
- **Without automated flags:** a diagnostic subset, still provisional. Absence of heuristic warnings is not evidence that text is correct.

Coverage is reported for distinct title/artist identities, weekly chart observations, and popularity weight `101 - this_week`. Each repeat appearance contributes its own weight. Yearly song counts may count the same identity in multiple years; do not sum them to obtain total unique songs. All months remain in the denominator, including unmatched songs and instrumentals. `manual_review.csv` prioritizes total study-period weight for retrospective curation only, never as a forecasting input. All provisional candidates still need review, even those absent from the heuristic queue.

## Conservative identity and quality rules

NFKC, case folding, curly apostrophe and punctuation/whitespace normalization generate candidates. Accents, word order, featured contributors, mix/live/remaster/part qualifiers are retained. Only feat/ft/featuring markers are harmonized. No fuzzy title-only acceptance, collaborator deletion, cover merging, or automatic remix substitution occurs. Punctuation normalization itself can collide, so all resulting matches remain provisional. Explicit alias mappings are review suggestions. Missing source identities, broken members, mismatched source titles/artists/years, and content-hash mismatches are reported.

Multiple source text hashes and the same hash under different chart identities are review flags, not grounds for automatic deletion. They can reflect formatting differences, legitimate covers, repeated recordings, or wrong returns. Concurrent date/title/artist collisions, including **Unchained Melody**, are retained at every chart rank. A later verified recording registry must resolve such cases at observation level, not globally by normalized identity.

Instrumentals are explicitly flagged and excluded from lyric-text coverage, never assigned six zero emotion scores. Source instrumental booleans and instrumental-only text are hints; there is no complete instrumental classifier. Double-sided/slash titles are flagged conservatively (some slashes are genuine single titles). Existing LYDIA component mappings are retained as references, but the prior arbitrary 0.5/0.5 weights are not adopted. A compound title may count as provisional availability with only one component; it is not ready for features. Decide a documented component policy before aggregation.

Every candidate has `language_status=unverified`. Non-Latin script is an extra review flag; it does not identify a language, and Latin-script Spanish or multilingual songs may not trigger it. No translation or language filtering occurs. Review language and model suitability before inference. Boilerplate, long text, release dates later than first chart appearance, version keywords inconsistent with source URLs, and featured-credit metadata are also flagged. These checks cannot reliably identify every wrong song or recording.

## Timing and selection bias

These are retrospective **availability reports**, not leakage-free emotion features. Even exact source matches may carry later corrections, later releases, or annotations. Source-year differences and release-date conflicts need review. A source's year-end year is not a release date. A candidate from a later year must not be assumed available at an earlier forecast origin.

Never use a year-end list to choose the monthly song universe. Kaggle, LYDIA and tmelo are supplemental lookup catalogs only; every weekly Billboard observation remains in coverage denominators. Annual selection still creates nonrandom missingness favoring later hits, even with correct denominators. Predeclare minimum matched-weight thresholds and sensitivity analyses by source/decade/language, before looking at recession forecasting scores. Future-period year-end membership must never be a feature or inclusion criterion.

For predictor month t, select chart observations dated within t, then require documented availability by the 15th of t+1. The helper `eligible_observation` rejects adjacent-month observations even when before the cutoff, and rejects known publication delays beyond it. Without release evidence it uses chart date only as an explicitly retrospective convention. Historical chart issue dates are not proven publication timestamps. Partial months need explicit expected chart counts/coverage checks before feature generation; the current audit does not impute missing weeks or rank slots. Instrumentals and unmatched weights must be reported separately when features are built. Never use future peak rank, lifetime chart duration, or full-period popularity as monthly weights.

Later Model B must meet the same 240 eligible training months and six-month horizon, and Model A must be rerun on identical eligible training/evaluation observations. If valid music months start January 1973 and are uninterrupted, the earliest forecast month is July 1993 under the existing label-availability rule. Actual validity and historical availability are unresolved, so no final evaluation period is selected here.

## Sources and permissions (checked 2026-10-17)

Exact URLs, declared licenses, local paths, and access findings are in `config/lyrics_sources.json`.

[OSF 2k7ut](https://osf.io/2k7ut/) describes a 1973–2023 weekly-chart corpus. Its public API identifies CC BY 4.0. The existing local file contains actual text, and is the primary candidate source because its chart slots can be reconciled. Its collection and publication occurred retrospectively. The project license alone does not establish authorization over every third-party lyric; obtain institutional/rightsholder clarification before broader NLP use or sharing. [CC's own notice](https://creativecommons.org/licenses/by/4.0/) says the license may not supply all necessary permissions.

[Brian Blakely's Kaggle dataset](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019) advertises 6,500 annual top-100 rows for 1959–2023, despite the older URL slug. Its public metadata lists MIT and describes Genius-derived lyrics. Existing raw CSV fields are audited; processed POS text is not used. Annual selection and underlying lyric authorization remain limitations.

[LYDIA](https://github.com/abrahamalbert18/LYDIA) advertises top-100 songs for 1959–2020 and recommends corrected JSON records. The repository declares MIT. JSON records preserve source URLs and instrumental indicators. Repository licensing does not independently settle lyric rights; this audit only inspects the existing local archive.

[tmelo-uea](https://github.com/tmelo-uea/billboard-hot100-lyrics) advertises 2,642 annual-chart songs for 2000–2024. Its README asserts noncommercial research/education and fair use; that is a publisher statement, not verified rightsholder permission. Only source years through 2023 are considered here.

[Original LyricsFreak/songdata listing](https://www.kaggle.com/datasets/mousehead/songlyrics): the original public API returned HTTP 403, and the browser could not retrieve the dataset page. Historical mirror descriptions claim 57,650 English songs and sometimes CC0, but neither current license nor underlying permission could be verified from the original source. Exact historical year coverage is unknown. Do not treat mirror labels as authorization. No corpus download or scraping was attempted. Its title/artist/text/link schema can be ingested through the optional local adapter only after documented authorization and a pinned file hash. Missing years remain flagged and no recording is automatically verified.

[LyricWhiz/MulJam](https://arxiv.org/abs/2306.17103) describes a multilingual Jamendo transcription corpus with a CC BY-NC-SA license. Billboard overlap is unestablished and its independent-music sampling is unlikely to fill this catalog efficiently; it was not downloaded. [LyricFind](https://www.lyricfind.com/) offers licensed lyrics products and is a preferable route for clarifying bulk academic NLP access, retention, and derived-data sharing. No agreement or coverage guarantee has been obtained. No additional corpus met both verified access/permission and demonstrated Billboard utility in this run.

## Optional authorized songdata.csv ingestion

Obtain authorization first, place the file under `data/raw/lyrics/`, and create a local JSON manifest with `path`, `sha256`, `authorized_local_analysis: true`, `permission_evidence`, and `reviewer`. Do not invent these assertions; the gate checks documentation presence, not legal sufficiency. Then add `--supplement-manifest /path/to/manifest.json` to the audit command. The adapter validates location, hash and artist/song/text schema, performs no downloads, and exports only references/metadata. URLs, year/release evidence and recording review still need completion. The manifest is included in run provenance; use nonconfidential evidence references.

## This run

Use `reports/lyrics_validation/run_002/` as the current validated output. `run_001/` is an intermediate diagnostic retained for audit history: it predated whole-file quarantine and recognition of the legacy whitespace-trimmed hash convention. The six tmelo mapping hashes all match trimmed text; new lineage hashes always preserve original whitespace. The 2012 and 2015 tmelo CSVs fail field/rank structure checks, so all 328 parsed records from those two files are quarantined. Repair quoting from an authoritative source before using them; do not guess text boundaries. See `reports/lyrics_validation/validation_report.md` for reconciled totals and limits.

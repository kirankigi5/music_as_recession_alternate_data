# Music review checklist

- Confirm raw lyrics remain Git-ignored and absent from tracked/staged files; inspect metadata exports for accidental lyric columns.
- Reproduce into a new directory and compare input SHA-256 hashes and reports. Source CSV record numbers exclude headers and count logical records, not physical lines.
- Review `mapping_issues.csv` and `source_issues.csv` before accepting prior coverage claims. Reconcile unsupported prior matches and new candidates against exact local source records.
- Treat all matches as provisional until a named reviewer confirms recording, full artist credits, source text, language, and provenance. There is no automatic approval or verified-match import in this audit.
- Resolve concurrent same-title/artist chart entries at date/rank level, especially Unchained Melody. Do not merge recordings or discard legitimate duplicates.
- Inspect remix/live/remaster conflicts, missing featured credits, release dates later than chart appearances, repeated text across identities, and multiple text variants. No automated heuristic establishes text correctness.
- Decide instrumental, multilingual, and double-sided-entry handling before inference; do not use zero emotion vectors or the existing unreviewed 50/50 component weights.
- Obtain sufficient institutional/licensor authorization for NLP use, storage and derived-data sharing. Repository license labels and fair-use statements are not independently verified lyric permissions. Do not download LyricsFreak mirrors on that basis.
- Preserve all weekly Billboard songs in denominators. Do not turn year-end membership into song selection; predeclare coverage thresholds and missingness sensitivity analyses before observing forecasting scores.
- Establish historical recording release and chart-publication timing. Existing reports measure retrospective catalog availability and must not be consumed as leakage-free monthly features.
- Before Model B: freeze a reviewed recording registry with exact hashes, define language/quality eligibility and monthly aggregation, then choose feasible training/evaluation windows without consulting comparative forecast performance. Rerun Model A on identical eligible months.
- Confirm economic source files, Model A implementation/configuration, and saved baseline results remain unchanged; run the whole test suite. No BERT inference is included.

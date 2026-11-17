# PR #1 closeout validation and reproduction levels

The closeout was assembled in an isolated worktree and tested from a release
snapshot materialized only from its explicit Git index. This file does
not change the historical experiment manifests or imply a current-code NLP rerun.

## Offline saved-result verification

After installing the baseline dependencies and pytest in a chosen environment,
run from the checkout root:

```sh
python -m pytest -q
python scripts/verify_saved_results.py
figure_parent=$(mktemp -d)
python scripts/make_research_figures.py --out "$figure_parent/figures"
verification_parent=$(mktemp -d)
python scripts/verify_saved_results.py --output "$verification_parent/verification.json"
```

The verifier defaults to stdout. Its optional output is a separate derived check,
not a replacement original metric, manifest or prediction file. The figure script
reads bundled `docs/results/` features, predictions, coverage and calibration.
These commands do not need raw lyrics, model weights, private inventory/audit
intermediates, credentials, or network. The tests use synthetic data and temporary
files rather than downloading a model.

## Full inference is a separate reproduction level

Full reproduction additionally requires externally obtained source files matching
the recorded input hashes, the inventory and regenerated lyrics-audit/OSF-screen
intermediates, optional music dependencies and model weights. The isolated langid
screen installation is documented in `scripts/audit_osf_evidence.py`. Data/source
access conditions are described in [source review](osf_source_use_review.md).
The exploratory source-use basis remains reported instructor guidance for
coursework; the strict evidence registry remains empty.

Only run the following in an explicitly prepared raw-data environment, after
checking those dependencies and accepting the inference cost:

```sh
# Optional setup in that environment; not needed for offline validation.
python -m pip install -e '.[music]'
experiment_parent=$(mktemp -d)
python scripts/run_exploratory_comparison.py \
    --output-dir "$experiment_parent/primary" \
    --audit-dir reports/lyrics_validation/run_002 \
    --screen reports/osf_source_review/run_004/candidate_evidence.csv \
    --cache-dir .agents/emotion_cache_explore
```

The audit and screen paths above refer to the original local intermediates, which
are not bundled; regenerate compatible inputs or supply prepared in-repository
paths with the supported flags. For sensitivities, use new output children and
`--coverage-threshold 0.75` or `0.80`. The driver refuses existing output directories.
Its default cache path differs from the original experiment cache, so always
choose the cache explicitly. Missing weights may trigger a network download.

The historical cache has 16,290 songs under NLP hash `614f862b…`. Current code
hash `e5600454…` has only 7,490 matching-version entries; the shared vectors are
identical, but 8,800 songs remain without a current-hash rerun. A cache is reusable
only when content, code, configuration and software metadata match. The historical
~50-minute full CPU inference cost is not repeated by this closeout.

Original `CODE_VERSION_PROVENANCE.md` files are preserved as archival records.
Their reference to an undistributed `.agents/` script is not a supported clean-
checkout command. Full raw-to-feature reproduction under current NLP code has
not been performed; the old manifest hashes have not been rewritten to suggest it.

## Validation execution record

Validated November 17, 2026 on an index-only release tree with **no**
`data/raw/`, `.agents/`, `data/metadata/` or lyrics-audit intermediates. Imports
were checked to resolve to that release tree, not the original dirty workspace.
The existing numerical environment was reused; a fresh dependency installation
and other platforms were not tested.

| Component | Version |
|---|---|
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| statsmodels | 0.14.6 |
| matplotlib | 3.9.4 |
| pytest | 8.4.2 |

| Check actually run | Result |
|---|---|
| Full applicable offline suite, once | **191 passed in 6.54 s; zero failures, errors or skips** |
| Saved-result verifier | All three Brier/ROC-AUC pairs, calibration tables, paired training dates/counts/means, original output/config/code hash checks passed; documented NLP hash exception retained |
| Verifier output reproducibility | Temporary JSON byte-identical to bundled `closeout_verification.json` |
| Primary descriptive corrections | 44 months / one recession in [0.3, 0.8); eight parameters; 46–61 positive training labels; 5.75–7.625 events per parameter |
| Same-date references | Training-rate Brier 0.0829708; existing full-history A forecasts score 0.0847541 on exactly the 183 primary dates |
| Figure generation in a temporary directory | **Ten PNGs** and `results_summary.json` generated; summary equals the original bundled summary |
| Local Markdown file/anchor links | **72 local links, including 20 anchors; zero broken** at the initial release check |
| Whitespace | `git diff --cached --check` passed after the scoped CRLF attributes below |
| Preservation | Existing model/configuration files, original predictions/metrics/coverage/calibration and historical manifests remained byte-identical; restored files match original local artifacts |

The full test command used the original numerical interpreter with `PYTHONPATH`
pointing exclusively at release source/root first, `PYTHONDONTWRITEBYTECODE=1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and pytest's cache provider disabled.
The tests create only temporary synthetic artifacts. No model was downloaded and
no real lyrics were scored.

No tests failed. An initial Git whitespace check identified CRLF endings in the
two restored feature CSVs. `.gitattributes` now preserves their original bytes
with `-text` and recognizes CR-at-EOL only for those two paths, retaining other
whitespace checks. The historical manifests were not edited to accommodate a
newline conversion.

Not performed: expensive inference, raw-corpus download/replay, human lyric or
recording validation, historical vintage/announcement reconstruction, a full
current-hash NLP rerun, or adoption of PR #2's uncertainty/PR-AUC implementation.
The source caches and earlier local uncertainty investigation remain outside
this release. Final slides and official timing/Q&A confirmation remain delivery
work, not a claim of completed statistical validation.

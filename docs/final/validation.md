# Final-deliverables validation

Validated November 17, 2026 in an isolated worktree based on main's merge commit
`eef070620a6621a601287334345fa683a6b61194`. PR #2 was merged with a merge commit
guarded by expected head `c3c62b056f553c6e687a8299296c969fafa23f77`; both parents
and Keefer's original ancestry were verified. No branch was deleted.

## Scope and actual checks

This record covers documentation, content binding, export structure, timing and
visual review. The historical [PR #2 validation](../paired_inference.md#offline-reproduction-and-validation)
records 215 passing numerical tests at the validated analysis head. Its three
analysis/test file hashes and paired-result JSON hash matched the published
record before merging. There were no new PR reviews, comments, checks or blockers.
The numerical/model/statistical suite was **not repeated** for this delivery:
all numerical code and original results remain byte-identical to updated main.

The authoring review covered the archival report, professor update, integrated
methods/results, original figures and the local five-page progress PDF. The local
PDF's older test count, different uncertainty convention and local-only command
were not adopted. That draft and all unrelated local work remain preserved.

The separate export environment used **Python 3.9.6**, python-pptx **1.0.2**,
ReportLab **4.2.5**, PyMuPDF **1.26.5**, Markdown **3.9** and embedded Arial
regular/bold fonts. Package versions and font hashes are recorded in `build_manifest.json`.
The pitch-alignment rebuild (prior work, study period, percent improvement, manual-review
status, literature references) used Python 3.12.5 with the same pinned packages and font hashes.
The existing numerical environment was not altered. Initial export attempts
identified an incompatible ReportLab 5.1.0 release and layout/glyph issues;
the pinned compatible version, adjusted boxes and supported equation notation
resolved them. Staged review also caught intentional Markdown hard-break
spaces and an ASCII-header PDF treated as text; explicit line breaks and binary
attributes scoped to this bundle resolved the diff-check failures.
Failed/intermediate exports and previews remain outside the repo.

Actual successful commands from the repository root, using the separate export
interpreter and new output paths:

```sh
python scripts/build_final_deliverables.py --out /tmp/music-final-export-FRESH
python scripts/check_final_deliverables.py --output /tmp/music-final-check-FRESH.json
git diff --check
git diff --cached --check
```

`validation_results.json` records the final machine-readable result:

- **73 grouped content/export checks passed**, with no remaining failure.
- Report: **7 PDF pages**; deck: **16 PDF/PPTX slides**, comprising **11 main + 5 appendix**.
- **294 native editable text boxes**, native vector charts/shapes, embedded notes
  on every slide, and no out-of-page PPTX shapes.
- **790 PDF text spans** within page bounds; report contains **59 PDF links**.
- Reported Brier, ROC-AUC, AP, prevalence and all block/HAC tables match bundled
  outputs at the displayed precision. Benchmarks, coverage counts, first-origin
  timing example, economic-table bounds and song counts were cross-checked.
- Source/evidence hashes match; both report images are generated from the same
  saved-results-bound presentation geometry.
- **203 local file links and 28 Markdown anchors passed** with zero failures.
  External URLs were not exhaustively probed.
- Working and staged diff whitespace checks passed. Of 117 pre-existing main
  files, **115 are byte-identical**; only README navigation and the implementation
  handoff's current-stage banner change. Models, predictions, configuration,
  historical manifests, NLP code and historical validation records are untouched.

## Visual review

Every final rendered slide and report page was inspected after the glyph/layout
corrections. No clipping, collisions, missing glyphs or unreadable figure labels
remain in the reviewed PDFs.

| Pages/slides | Visual review result |
|---|---|
| Main 1–4 | Question, data funnel, coverage bars and numbered pipeline fit; chart labels and flow direction are readable. |
| Main 5–6 | Model equations render without missing glyphs; predictor month, assumed issuance and single-month target remain distinct. Timeline is marked schematic. |
| Main 7–9 | Same-date benchmark labels and Brier direction are explicit; nested samples and AP names are legible; zero-crossing intervals and all primary HAC settings are visible. |
| Main 10–11 | Measurement, episode, timing and provenance limits fit; conclusion preserves the qualified negative finding. |
| Appendix A–E | Sensitivity tables, diagnostics, reproduction levels, CRISP-DM decisions, references and TA clarification fit without clipping. |
| Report 1–7 | Section breaks, full tables, two figures, equations, references and reproduction commands are legible; no unintended overflow page or split table. |

The deck PDF uses ReportLab's shared vector geometry and explicit wrapping,
not a PowerPoint/Keynote rendering engine. Native-app font substitution or room
projection remains a presentation-machine check. PDF exports are available;
there is no missing report/deck export. Visual review of the shared-layout PDF
does not claim a native PowerPoint rendering test.

## Timing and remaining requirements

Main-slide targets are **40 / 55 / 60 / 65 / 65 / 70 / 70 / 60 / 75 / 55 / 45
seconds**, totaling **660 seconds (11:00)**. The main narration contains **1,258
words**, about 114 words/minute at that pace, including the planned chart-reading
pauses. This is a prepared timing plan, not a measured team rehearsal. Appendix
slides have no scheduled narration. The supplied 12:30 slot leaves 1:30 buffer;
whether Q&A is included remains unconfirmed.

Still to confirm: official date/time and Q&A allocation, submission deadline and
upload destination, report/deck format or length rules, final speakers and team
availability, presentation-machine behavior, and the TA's intended spread
interpretation. No email was sent. No new experiment or inference was performed.

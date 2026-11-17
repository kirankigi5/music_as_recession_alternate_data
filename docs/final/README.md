# Final project delivery bundle

**Can Popular Music Predict U.S. Recessions? · NYU Predictive Analytics · Team 5**

| Deliverable | Editable source | Fixed-layout export |
|---|---|---|
| Concise final report | [report.md](report.md) | [report.pdf](report.pdf), 7 pages |
| Presentation | [presentation.pptx](presentation.pptx), [slides.json](slides.json) | [presentation.pdf](presentation.pdf), 16 slides |
| Speaker notes and timing | [speaker_notes.md](speaker_notes.md); notes also embedded in PPTX | 11 main slides, planned 11:00 |
| Rehearsal, likely questions and unsent email | [team_handoff.md](team_handoff.md) | Draft only; nothing sent |
| Export provenance and checks | [build_manifest.json](build_manifest.json), [validation.md](validation.md), [validation_results.json](validation_results.json) | Sources, versions, geometry and actual results |

Five appendix slides have no scheduled narration. December 3, 2026 and a 12:30
slot are supplied planning constraints. Q&A inclusion remains unconfirmed; the
1:30 buffer is not a promise of sufficient Q&A time. The 11:00 duration is a
timing plan, not a measured rehearsal. Speaker ownership and submission details
still require team/course confirmation.

The authoritative numerical evidence is the integrated repository's saved
predictions, original metrics, [verification JSON](../results/closeout_verification.json)
and [paired inference JSON](../results/paired_inference.json). The report and
figures add no fitting, experiment or statistical engine. PR #2 was merged as
`eef070620a6621a601287334345fa683a6b61194`; historical validation records remain
unchanged.

## Edit and rebuild

Edit `report.md` for report prose and `slides.json` for slide text, notes and
timing. Charts bind to bundled derived results in
[build_final_deliverables.py](../../scripts/build_final_deliverables.py).
The PPTX contains native editable text and vector shapes rather than flattened
slide screenshots. Edits made directly to the PPTX are not imported back into
the JSON source; update the source for a reproducible rebuild.

Use a separate export environment; the model environment is not changed:

```sh
python3 -m venv /tmp/music-export-env-FRESH
/tmp/music-export-env-FRESH/bin/pip install -r docs/final/requirements-export.txt
/tmp/music-export-env-FRESH/bin/python scripts/build_final_deliverables.py \
  --out /tmp/music-final-export-FRESH
/tmp/music-export-env-FRESH/bin/python scripts/check_final_deliverables.py \
  --bundle /tmp/music-final-export-FRESH \
  --output /tmp/music-final-check-FRESH.json
```

Run from the checkout root. Replace FRESH with new unused paths. The exporter
refuses an existing output directory; the checker refuses an existing result
file. Installing export dependencies may use the network; subsequent building
and checking read only local sources. No model/data download is needed.

The tested environment is Python 3.9.6 with the versions in
[requirements-export.txt](requirements-export.txt). Export requires Arial regular
and bold TTF fonts. It discovers them on macOS or in a common Linux msttcorefonts
path; elsewhere provide `--font-regular /path/to/Arial.ttf` and
`--font-bold /path/to/Arial-Bold.ttf`. Fonts are embedded in PDFs and are not
bundled separately. A fresh install on every platform is not claimed.

The deck PDF is drawn from the **same vector geometry and explicit text wrapping
as the PPTX**, using ReportLab. It is not a PowerPoint/Keynote roundtrip. All
rendered PDF slides were visually reviewed; actual presentation software can
still substitute fonts, so open the native deck on the presentation machine.
The PDF is the fixed-layout fallback. The report PDF is exported directly from
Markdown; its local evidence links become repository links in the PDF.

Review exports before selectively replacing the published files. Generated
previews and failed/intermediate builds belong outside the repository. The
bundled [build manifest](build_manifest.json) records source/evidence hashes,
package versions and text-box geometry. The checker validates reported tables,
notes, timing, file links, source bindings and PDF/PPTX structure; visual review
is recorded separately rather than inferred from those checks.

## Existing artifacts inspected

The archival [research report](../COMPLETE_RESEARCH_REPORT.md),
[professor update](../../README_PROFESSOR.md), integrated results/methods and
original ten figures were reviewed before authoring. The original workspace's
five-page October 27 `mid_report.pdf` was inspected read-only: it uses an older
test count, different local uncertainty values and a local-only audit command.
It remains preserved and is not a final authority. No tracked or local project
deck was found in the inspected project folders/history. The final deck therefore
adds an editable presentation rather than overwriting an existing deck.

## Scope of the conclusion

Observed music-model loss is higher on the selected samples, but robust
statistical degradation is not established. Threshold samples are nested;
there are few recession episodes; inference is approximate and conditional on
saved predictions. Historical availability, matching/NLP measurement and the
8,800 songs without a current-hash rerun remain explicit limitations. Reviewer
experiments without available code/results are not presented as completed work.

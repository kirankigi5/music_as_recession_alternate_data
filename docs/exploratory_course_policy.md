# Exploratory course-analysis policy

Added 2026-10-18 for the NYU Predictive Analytics course project. This policy
exists **alongside** the strict primary policy, which is unchanged. Any result
produced under this policy is exploratory coursework output and must be labelled
as such wherever it appears.

## Source-use basis: reported instructor guidance

The project team reports that their instructor said lyrics may be downloaded and
analysed for this course project.

This is recorded here as **reported instructor guidance for a course project**.
It is explicitly **not**:

- a copyright licence or permission from any rightsholder,
- a determination of fair use,
- NYU legal or Open Knowledge approval,
- independent verification that any recording, language or release date is correct.

The configuration field is `"source_use_basis":
"reported_instructor_guidance_course_project_not_a_license"`, and the same
wording is written into every output manifest, so a reader cannot mistake it for
a rights clearance. The five outstanding decision and evidence questions in
`docs/osf_source_use_review.md` remain open and unaltered.

## What the policy admits

Provisional OSF records that pass the project's existing **automated** screens,
both of which already existed and were not re-tuned for this run:

- `automated_screen_pass == True` in `reports/osf_source_review/run_004/candidate_evidence.csv`
  (metadata and chart-slot agreement, no featured-credit or version risk flag, no
  instrumental marker, no chart-observability failure) — 16,290 of 19,096 candidates;
- `language_status == english_automated_consistent` (automated langid screen).

Records failing either screen are not admitted. The 2,340 reviewable
featured-credit and version-flagged records are **not** admitted.

## What the policy still refuses to assert

| Axis | Status under this policy |
|---|---|
| Recording identity | Automated metadata and chart-slot match only. `recording_verified` stays **False**. |
| Language | Automated langid screen. Not human reviewed. |
| Release date | **No verified release date.** `release_date` is left **empty**, never back-filled from a chart date. Chart observability alone gates contributions. |
| Rights | Reported instructor guidance. Not a licence or clearance. |

Leaving `release_date` empty is deliberate. `docs/osf_source_use_review.md` item 4
warns against filling that field with a chart date to satisfy code, so the
exploratory path keeps the field empty and records the weaker basis in
`release_evidence` instead. Every admitted record carries its outstanding gaps in
`readiness_blockers`.

Because `recording_verified` must remain False and `release_date` must remain
empty, the strict `require_eligible` gate cannot be reused — it demands both. The
exploratory path therefore has its own gate, `require_exploratory_eligible`, which
asserts the **opposite**: it refuses any row claiming a verified recording or a
release date. The two gates cannot be confused for one another, and
`eligibility_basis` records which one applied.

## Predeclared analysis plan

Fixed before any score was computed, and not revised afterwards:

- **Primary:** 70% minimum monthly popularity-weight coverage.
- **Sensitivity:** 75% and 80%.

Thresholds were chosen from the coverage feasibility work in
`reports/model_b_readiness/run_002/`, which was produced deliberately blind to
model performance. They were not adjusted after seeing any Brier Score.

## Known limitations that no threshold choice fixes

- Only three recession episodes (2001, 2008–09, 2020) are reachable, because the
  corpus starts in 1973 and 240 eligible training observations are required.
  That bounds the precision of any Brier comparison.
- The unflagged subset's monthly coverage declines over time (median 0.92 in the
  1980s to 0.61 in the 2020s), because featured-artist credits became common and
  the upstream lookup stripped them. Coverage drift is therefore confounded with
  any apparent emotion effect.
- GoEmotions is a Reddit-trained model applied to lyrics, out of domain and
  unvalidated for this use; `nervousness` is a weak pretrained label.
- Final FRED and NBER values contain hindsight, and chart issue dates are a
  retrospective convention rather than proven publication timestamps.

## Reproduction

```bash
# Primary, 70% coverage
python scripts/run_exploratory_comparison.py \
    --output-dir reports/model_comparison/exploratory_primary_cov70

# Predeclared sensitivity analyses
python scripts/run_exploratory_comparison.py \
    --output-dir reports/model_comparison/exploratory_sensitivity_cov75 \
    --coverage-threshold 0.75
python scripts/run_exploratory_comparison.py \
    --output-dir reports/model_comparison/exploratory_sensitivity_cov80 \
    --coverage-threshold 0.80
```

The emotion cache is keyed by lyric content hash, model configuration and
implementation hash, so songs are scored once and runs resume without repeating
completed work. No lyric text is written to any output.

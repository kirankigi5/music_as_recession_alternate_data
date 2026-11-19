# Final delivery: team rehearsal and TA update

## Current state

PR #2 is merged by commit `eef070620a6621a601287334345fa683a6b61194`, preserving Keefer's original commit and authorship. The integrated saved results and methods are authoritative. This final-deliverables branch adds a concise [report](report.md), [report PDF](report.pdf), [editable deck](presentation.pptx), [deck PDF](presentation.pdf), [slide source](slides.json) and [speaker notes](speaker_notes.md). It changes no experiment or statistical engine. Historical validation records remain intact.

The main talk has **11 slides, planned for 11:00**. Five appendix slides have no scheduled narration. The supplied slot is December 3, 2026, **12:30**. The remaining 1:30 is a contingency buffer; do not promise that it covers Q&A until the course team confirms whether Q&A is inside the slot. Timing is a plan, not a measured rehearsal result.

## Rehearsal plan

1. Confirm speakers and official instructions. Suggested allocation, subject to team agreement: Kiran slides 1–4 (3:40), Keefer slides 5–7 (3:25), Nathan slides 8–11 (3:55). Each person should still understand the model, timeline and conclusion.
2. Run one complete timed rehearsal from the PDF or deck, including speaker handoffs. Record actual times in a separate team note; the current speaker notes specify targets only. Reach slide 7 by 5:55 and slide 10 by 9:20. Use appendix slides only when asked.
3. If over time, shorten motivation and pipeline narration first. Preserve the issuance/target distinction, same-date benchmarks, uncertainty qualification and conclusion. Aim for 10:45–11:00 on the next rehearsal, leaving room for transitions.
4. Open the PPTX and PDF on the actual presentation machine. Check Arial/font substitution, aspect ratio, room/projector readability, notes access and clicker behavior. The PDF is the fixed-layout fallback. Confirm an offline copy is available.
5. Before submission, agree on attachments, filenames, who uploads and who sends the email. This draft does not assign a confirmed presenter or sender and sends nothing.

## Likely questions and evidence-based answers

| Question | Answer and evidence |
|---|---|
| Does the current model already use the yield difference? | Yes. `spread_t = GS10_t − TB3MS_t` in percentage points enters Probit continuously. The binary USREC outcome is separate from the probabilistic prediction. See [model equation and timing](report.md#3-models-calendar-and-crisp-dm) and [baseline implementation](../../src/music_recession/baseline.py). |
| Is this onset within six months? | No. It is recession status in the single month t+6. Assumed issuance is the 15th of t+1, so “six months” is indexed to the predictor month. See [forecast protocol](../forecast_protocol.md). |
| Could the training labels actually have been known? | Not established. Targets end by t−1, but NBER dating is retrospective; final economic snapshots and source timing remain hindsight risks. This is retrospective pseudo-out-of-sample evaluation. |
| Why compare 0.0904 with 0.1202 rather than 0.0949? | Paired A/B share eligible training rows and evaluation dates. The 0.094921 broad baseline uses 630 different dates. The valid full-history same-date reference is 0.0847541 on the same 183 dates. [Verification JSON](../results/closeout_verification.json). |
| What does the simple reference predict? | At each origin, the recession fraction in that saved paired training window. Its primary Brier is 0.0829708. It does not use full evaluation prevalence. |
| Is Model B significantly worse? | Robust degradation is not established. Primary 12/24-month block intervals include zero; approximate HAC p-values at lags 5/12/24 are 0.039930/0.059871/0.047918. Settings and few episodes limit inference. [All sensitivities](../paired_inference.md#results-and-sensitivity). |
| Why circular blocks and these lags? | Uniform calendar starts avoid endpoint under-inclusion, but wraparound creates artificial adjacency. Existing 6/12/24-month blocks and HAC lags 5/12/24 show shorter/annual/biennial sensitivity, not settings chosen for a favorable p-value. The target does not create overlapping six-month outcome windows. |
| What uncertainty is omitted? | It conditions on saved predictions and the selected calendar; it does not refit models or propagate training estimation, matching, NLP measurement, policy selection or data vintages. [Methods and limits](../paired_inference.md). |
| Are the three thresholds independent confirmations? | No. The 102 and 45 dates are subsets of the 183 primary dates, reaching two and one recession episodes. Primary has only three. Changing samples explains why cross-threshold AUCs are not a controlled coverage effect. |
| Does PR-AUC mean trapezoidal area? | Here the metric is stepwise average precision, with ties grouped. Primary A/B AP is 0.1688/0.0722; prevalence 0.0820 is a constant-score reference, not a real-time forecast or exact random-ranking expectation. |
| Why no PCA, random split or extra model? | Six features were preselected for interpretation; PCA would change the question. Random splitting breaks chronology. New models/horizons would be separate declared experiments, not post-result repairs. |
| Is overfitting established? | No. Primary windows have 46–61 positives / eight parameters = 5.75–7.625 events per parameter. Model B has 44 evaluated probabilities in [0.3,0.8), one recession. These diagnostics do not prove a cause. |
| Are lyrics, permissions and scores fully validated? | No. Exploratory automated screens are not recording/language/release verification or rights clearance. Strict policy admits zero records. GoEmotions is unvalidated on these lyrics. The earlier 7,490 shared vectors match, but 8,800 songs lack a current-hash rerun. |
| The pitch said 1959–2023 and nine recessions. Why 1973? | The OSF lyrics corpus covers 1973–2023, so 1959–1972 is outside this analysis. With the 240-month training minimum, paired forecasts start in November 1993 and reach three recessions. See [report Section 2](report.md#2-data-preparation-and-measurement). |
| The pitch said BERT. What model was used? | The pinned `SamLowe/roberta-base-go_emotions` checkpoint, a RoBERTa model trained on the GoEmotions labels. [Model config](../../config/music_model.json). |
| Did you do the manual review of emotion scores from the pitch? | No. It was planned and not performed, so measurement failure and absence of signal cannot be told apart. |
| What is the pitch's percent-improvement number? | 100 × (BS_A − BS_B) / BS_A is −33.0% (70%), −55.0% (75%) and −88.3% (80%). Negative means Model B is worse. |
| Did you verify reviewers' alternative experiments? | No. Ridge, single-emotion, detrending and alternate-lag claims still need code/results. They are not final-project findings. |

## Precise TA clarification

“Our implementation already uses the **continuous GS10 minus TB3MS spread in percentage points** to estimate **P(USREC in month t+6 = 1)** with Probit; only the outcome is binary. When you suggested using the yield difference to find the probability, did you mean the spread's magnitude rather than a binary inversion indicator, a **change in the spread over time**, or a **different target**? If a change was intended, which definition and forecast timing should we use?”

The first interpretation is already implemented. The other interpretations are possibilities to clarify, not accepted requirements or completed experiments. Do not change the target, horizon or predictor based on a guess.

## Requirements still to confirm

- Official presentation date/time and whether Q&A is included in the 12:30 slot; the supplied planning date is December 3, 2026.
- Required report length/template, citation style, slide format, slide limit, appendix rules and whether speaker notes should be submitted.
- Submission deadline/time zone, upload destination, repository-access expectations and attachment size limits.
- Final team speaker allocation, rehearsal availability and who submits/sends the update.
- Whether a well-qualified negative result plus the implemented pipeline meets the course team's final-project expectations; no additional experiment is assumed necessary.
- TA's intended yield-spread interpretation. Any new experiment requires a separately agreed design, not an undocumented change to these final results.

## Draft email — not sent

**Subject:** Team 5 project update and alternative meeting time

Dear Professor Bari and course team,

We have completed the paired recession-forecast comparison and integrated the calendar-aware uncertainty analysis. Adding the six lyric-emotion features increased observed Brier loss on the selected samples, but the uncertainty results depend on settings, so we do not claim robust statistical degradation. Our final report and presentation also document the data, implemented pipeline, retrospective timing and reproduction limits.

We are attaching our current project report PDF and sharing our GitHub repository: https://github.com/kirankigi5/music_as_recession_alternate_data.

Would it be possible to arrange an alternative meeting on Thursday afternoon or at a later time convenient for you? We will confirm a shared time subject to all team members' availability.

We would also appreciate clarification on the yield-spread feedback. Our Probit already uses continuous GS10 minus TB3MS, in percentage points, to estimate the probability of recession status in month t+6; the outcome alone is binary. Did you mean using that spread magnitude rather than an inversion indicator, a change in the spread, or a different target?

Could you also confirm whether Q&A is included in the 12-minute-30-second presentation slot and whether there are any specific final submission requirements?

Best regards,

Kiran Reddy Bhumireddy, Keefer Wu and Nathan Starkman

NYU Predictive Analytics — Team 5

Before sending: confirm recipients and team availability, attach `report.pdf`, and verify the linked repository is accessible. This file is only a draft; no email has been created or sent through a mail service.

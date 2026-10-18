# Missing Lyrics Tracker — 1973–2023

## Project
Can Popular Music Predict U.S. Recessions?

## Current Data Status

- Analysis period: 1973–2023
- Total unique Billboard songs: 21,542
- Previously provisionally matched: 20,298
- Additional LRCLIB lyric candidates: 812
- Additional NetEase lyric candidates: 113
- Total additional candidates: 925
- Preliminary quality-screening passes: 600
- Missing or requiring further review: approximately 632

These are provisional matches, not verified research-ready coverage.
The earlier source audit identified 12 additional exceptions outside
the 1,232-song recovery queue; these must also be reconciled.

## Collaboration

See:
reports/missing_song_links_1973_2023.csv

Contributors can search for missing songs using the links provided.

For each resolved song:
1. Verify the correct artist and recording.
2. Check remixes, covers, combined singles and instrumentals.
3. Record the source URL and the evidence for the match.
4. Confirm permission for the intended research use.
5. Submit findings through a GitHub issue or pull request.

Do not upload copyrighted lyric text to GitHub.
Keep raw lyrics in the locally ignored data/raw/lyrics/ directory.

## Modeling

Modeling proceeds independently using the existing eligible dataset.

Model A: Yield-spread baseline.
Model B: Yield spread plus monthly lyrical-emotion features.

Compare both models on the same chronological evaluation periods,
using identical recession targets and Brier scores.

The new 925 candidates must not automatically enter Model B until
recording identity, text quality and usage rights are checked.

## Scope

Current analysis: 1973–2023.
Historical extension 1959–1972 is tracked separately.

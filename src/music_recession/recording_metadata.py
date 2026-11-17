"""Recording-metadata corroboration for OSF lyric candidates.

Purpose: decide whether an authoritative metadata source (MusicBrainz) corroborates
that a Billboard chart identity -- including its featured-artist credit -- denotes a
single, datable recording, and whether the REDUCED query that the upstream
lyrics.ovh lookup actually issued (primary artist + title, with the featured credit
stripped) could have resolved to some other recording.

CRITICAL SCOPE LIMIT. Metadata corroboration is NOT verification that the OSF lyric
text belongs to the corroborated recording. The upstream lookup is unauditable, so
no outcome here may set `recording_verified`; every row this module emits carries
`recording_verified=False`. Rights/permission and verified release-date evidence are
separate axes and are untouched by this module.

All functions here are pure. Network access and caching live in
scripts/match_recording_metadata.py so the decision logic is testable offline.
"""
from __future__ import annotations

import re
import unicodedata

EVIDENCE_LEVELS = ('metadata_corroborated', 'ambiguous_requires_review',
                   'conflicting_metadata', 'no_match')

# Billboard credit grammar. Only feature markers split primary from guests; "&"
# and "X" inside a primary credit ("Macklemore & Ryan Lewis") must be preserved.
FEATURE_MARKER = re.compile(r'\s+(?:featuring|feat\.?|ft\.?|with)\s+', re.I)
GUEST_SPLIT = re.compile(r'\s*(?:,|&|\+|/|\bx\b|\band\b)\s*', re.I)

# A trailing parenthetical naming a guest, e.g. "Old Town Road (feat. Billy Ray Cyrus)".
TITLE_FEATURE = re.compile(r'^[\(\[\-\s]*(?:feat\.?|ft\.?|featuring|with)\s+(?P<who>.+?)[\)\]\s]*$', re.I)

# Version/variant vocabulary. A match means "same song, different rendition",
# which is materially weaker evidence than an exact-title match.
VERSION_WORDS = ('remix', 'mix', 'live', 'version', 'edit', 'remaster', 'remastered',
                 'instrumental', 'acoustic', 'demo', 'karaoke', 'extended', 'radio',
                 'single', 'album', 'reprise', 'intro', 'outro', 'interlude', 'dub',
                 'bonus', 'cover', 'session', 'take', 'alternate', 'mono', 'stereo',
                 'clean', 'explicit', 'original', 'rerecorded', 're recorded')

# Release-group secondary types that mean the date is NOT an original release date.
NON_ORIGINAL_SECONDARY = {'compilation', 'live', 'remix', 'dj-mix', 'mixtape/street',
                          'demo', 'soundtrack', 'interview', 'audiobook', 'spokenword'}

# Tolerances, in years.
CHART_LAG_TOLERANCE = 1      # release may trail the chart year by rounding/partial dates
COVER_RISK_TOLERANCE = 3     # an original far older than the chart suggests a cover
DIVERGENT_YEAR_SPAN = 5      # original releases spread wider than this suggest two songs

# Flags that genuinely undermine recording-identity corroboration. Release-date
# gaps are deliberately excluded: that is a separate evidence axis.
BLOCKING_FLAGS = frozenset({'featured_credit_absent_from_metadata',
                            'no_featured_artist_in_billboard_credit',
                            'divergent_original_release_years',
                            'only_version_variant_titles',
                            'exemplar_is_variant_recording',
                            'original_release_long_before_chart'})


def normalise(text: str) -> str:
    """Casefolded alphanumeric form; drops punctuation, accents and curly quotes."""
    text = unicodedata.normalize('NFKD', text or '').casefold()
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('\u2010', '-')
    return re.sub(r'[^a-z0-9]+', '', text)


def _words(text: str) -> list:
    text = unicodedata.normalize('NFKD', text or '').casefold()
    return [w for w in re.split(r'[^a-z0-9]+', text) if w]


def split_billboard_credit(artist: str) -> tuple:
    """Billboard credit -> (primary, [guest, ...]).

    Guests shorter than four normalised characters are dropped: they cannot be
    matched reliably against MusicBrainz credits and would create false positives.
    """
    parts = FEATURE_MARKER.split(artist or '', maxsplit=1)
    primary = parts[0].strip()
    guests = []
    if len(parts) > 1:
        guests = [g.strip() for g in GUEST_SPLIT.split(parts[1]) if g.strip()]
    return primary, [g for g in guests if len(normalise(g)) >= 4]


def classify_title(mb_title: str, billboard_title: str, guests=()) -> tuple:
    """Compare a MusicBrainz recording title with the Billboard title.

    Returns (kind, detail) where kind is:
      'exact'     - same title
      'featured'  - title extends the Billboard title with a guest credit
      'version'   - title extends it with a version/variant qualifier
      'different' - a different song; never a candidate

    The 'different' outcome is what stops bare prefix matching from conflating
    distinct songs, e.g. "Magic" vs "Magic Number".
    """
    want, got = normalise(billboard_title), normalise(mb_title)
    if not got:
        return 'different', 'empty title'
    if got == want:
        return 'exact', ''
    if not got.startswith(want):
        return 'different', 'title does not extend the Billboard title'

    # Recover the human-readable remainder so punctuation is available.
    raw = (mb_title or '').strip()
    idx, seen = 0, 0
    for i, ch in enumerate(raw):
        if normalise(ch):
            seen += 1
        if seen == len(want):
            idx = i + 1
            break
    remainder = raw[idx:].strip()

    feat = TITLE_FEATURE.match(remainder)
    if feat:
        who = normalise(feat.group('who'))
        if guests and any(normalise(g) in who for g in guests):
            return 'featured', feat.group('who').strip()
        return 'featured', feat.group('who').strip()

    tokens = set(_words(remainder))
    if tokens & set(VERSION_WORDS):
        return 'version', remainder.strip('()[] ')
    return 'different', 'unrecognised title extension: ' + remainder[:40]


def release_group_kind(release: dict) -> str:
    """'original' only when no secondary type disqualifies the date."""
    group = (release or {}).get('release-group') or {}
    secondary = {str(s).casefold() for s in group.get('secondary-types') or []}
    if secondary & NON_ORIGINAL_SECONDARY:
        return sorted(secondary & NON_ORIGINAL_SECONDARY)[0]
    primary = (group.get('primary-type') or '').casefold()
    if primary in ('album', 'single', 'ep', ''):
        return 'original'
    return primary or 'other'


def _year(value: str):
    value = (value or '').strip()
    return int(value[:4]) if len(value) >= 4 and value[:4].isdigit() else None


def release_dates(recording: dict) -> dict:
    """Split a recording's release dates into original vs non-original buckets.

    A compilation or reissue date must never stand in for an original release date;
    that defect produced spurious conflicts in the first pilot pass.
    """
    original, other, groups = [], [], set()
    for rel in recording.get('releases') or []:
        kind = release_group_kind(rel)
        date = (rel.get('date') or '').strip()
        group = ((rel.get('release-group') or {}).get('id'))
        if kind == 'original':
            if group:
                groups.add(group)
            if date:
                original.append(date)
        elif date:
            other.append(date)
    first = (recording.get('first-release-date') or '').strip()
    return dict(original=sorted(original), non_original=sorted(other),
                original_groups=groups,
                first_release_date=first,
                earliest_original=(min(original) if original else ''),
                earliest_any=(min(original + other + ([first] if first else []))
                              if (original or other or first) else ''))


def credit_names(recording: dict) -> str:
    return ''.join((ac.get('name') or '') + (ac.get('joinphrase') or '')
                   for ac in recording.get('artist-credit') or []).strip()


DEFAULT_MIN_SCORE = 75
"""MusicBrainz lowers a search score when the query artist is only PART of the
recording's credit. A reduced query ("LMFAO") against the true credit ("LMFAO,
Lauren Bennett, GoonRock") scores ~80-84, so a floor near 100 discards precisely
the entities that corroborate a featured credit. Measured on the Step 5 pilot."""


def assess(billboard_title: str, billboard_artist: str, first_chart_date: str,
           recordings, min_score: int = DEFAULT_MIN_SCORE) -> dict:
    """Assign an evidence level to one Billboard identity.

    `recordings` is the MusicBrainz recording-search result list for the REDUCED
    query (primary artist + title). Never sets recording_verified.
    """
    primary, guests = split_billboard_credit(billboard_artist)
    chart_year = _year(first_chart_date)

    candidates, exact, version_only, flags = [], 0, 0, set()
    for rec in recordings or []:
        if (rec.get('score') or 0) < min_score:
            continue
        kind, detail = classify_title(rec.get('title', ''), billboard_title, guests)
        if kind == 'different':
            continue
        names = normalise(credit_names(rec))
        by_credit = bool(guests) and any(normalise(g) in names for g in guests)
        by_title = kind == 'featured' and bool(guests) and any(
            normalise(g) in normalise(detail) for g in guests)
        disambig = normalise(rec.get('disambiguation', ''))
        candidates.append(dict(rec=rec, kind=kind, detail=detail,
                               guest_in_credit=by_credit, guest_in_title=by_title,
                               dates=release_dates(rec),
                               is_variant=(kind == 'version'
                                           or any(w in disambig for w in ('remix', 'live',
                                                                          'karaoke', 'cover')))))
        if kind == 'exact':
            exact += 1
        elif kind == 'version':
            version_only += 1

    if not candidates:
        return dict(evidence_level='no_match', recording_verified=False,
                    reasoning='no MusicBrainz recording matched the reduced query '
                              f'"{primary} - {billboard_title}" at score >= {min_score}',
                    review_flags='no_candidate', mb_recording_mbid='', mb_evidence_url='',
                    mb_exemplar_title='', mb_exemplar_credit='', earliest_original_release='',
                    earliest_any_release='', candidate_count=0, exact_title_matches=0,
                    guest_evidence='', primary_artist=primary, featured_artists='; '.join(guests))

    # Prefer a corroborating, non-variant exemplar.
    def rank(c):
        return (not (c['guest_in_credit'] or c['guest_in_title']),
                c['is_variant'], c['kind'] != 'exact', -(c['rec'].get('score') or 0))

    ordered = sorted(candidates, key=rank)
    best = ordered[0]
    guest_credit = [c for c in candidates if c['guest_in_credit']]
    guest_title = [c for c in candidates if c['guest_in_title']]
    corroborated = bool(guest_credit or guest_title)
    guest_evidence = ('artist_credit' if guest_credit else
                      ('recording_title' if guest_title else ''))

    originals = sorted({d for c in candidates for d in c['dates']['original']})
    non_original = sorted({d for c in candidates for d in c['dates']['non_original']})
    groups = {g for c in candidates for g in c['dates']['original_groups']}
    earliest_original = originals[0] if originals else ''
    earliest_any = min([d for d in (
        [c['dates']['earliest_any'] for c in candidates]) if d], default='')

    if not guests:
        flags.add('no_featured_artist_in_billboard_credit')
    elif not corroborated:
        flags.add('featured_credit_absent_from_metadata')
    # Release-date gaps are recorded but do NOT block identity corroboration:
    # verified release dates are a separate evidence axis in this project, and
    # MusicBrainz search returns only a subset of each recording's releases.
    if not originals and non_original:
        flags.add('only_non_original_release_dates')
    if not originals and not non_original:
        flags.add('no_release_date_available')
    # A single plus an album release group is ordinary for one song, so group count
    # alone is not evidence of two songs. Divergent original YEARS are.
    if len(groups) > 1:
        flags.add('multiple_original_release_groups')
    original_years = sorted({y for y in (_year(d) for d in originals) if y})
    if original_years and original_years[-1] - original_years[0] > DIVERGENT_YEAR_SPAN:
        flags.add('divergent_original_release_years')
    if exact == 0 and version_only:
        flags.add('only_version_variant_titles')
    if best['is_variant']:
        flags.add('exemplar_is_variant_recording')

    level = 'ambiguous_requires_review'
    reasoning = []
    og_year = _year(earliest_original)

    if og_year is not None and chart_year is not None and og_year > chart_year + CHART_LAG_TOLERANCE:
        level = 'conflicting_metadata'
        flags.add('original_release_after_first_chart')
        reasoning.append(f'earliest ORIGINAL release {earliest_original} postdates first '
                         f'chart appearance {first_chart_date} by more than '
                         f'{CHART_LAG_TOLERANCE} year(s)')
    elif og_year is not None and chart_year is not None and og_year < chart_year - COVER_RISK_TOLERANCE:
        flags.add('original_release_long_before_chart')
        reasoning.append(f'earliest original release {earliest_original} precedes the chart '
                         f'year {chart_year} by more than {COVER_RISK_TOLERANCE} years, so the '
                         'reduced query may denote an earlier different song or a cover')
    elif corroborated and (exact or guest_title) and not flags & BLOCKING_FLAGS:
        # A title-embedded guest credit ("Old Town Road (feat. ...)") is corroboration
        # in its own right, so an exact-title match is not additionally required.
        level = 'metadata_corroborated'
        kind = 'exact-title' if exact else 'title-credited'
        reasoning.append(f'{kind} recording carries the featured credit via '
                         f'{guest_evidence}; earliest original release {earliest_original or "n/a"} '
                         f'is consistent with first chart appearance {first_chart_date}')

    if level == 'ambiguous_requires_review' and not reasoning:
        reasoning.append('candidate(s) found but corroboration incomplete: '
                         + ', '.join(sorted(flags)))

    rec = best['rec']
    return dict(
        evidence_level=level,
        recording_verified=False,          # never set by metadata corroboration
        reasoning='; '.join(reasoning),
        review_flags=';'.join(sorted(flags)),
        mb_recording_mbid=rec.get('id', ''),
        mb_evidence_url=('https://musicbrainz.org/recording/' + rec['id']) if rec.get('id') else '',
        mb_exemplar_title=rec.get('title', ''),
        mb_exemplar_credit=credit_names(rec),
        earliest_original_release=earliest_original,
        earliest_any_release=earliest_any,
        candidate_count=len(candidates),
        exact_title_matches=exact,
        guest_evidence=guest_evidence,
        primary_artist=primary,
        featured_artists='; '.join(guests))

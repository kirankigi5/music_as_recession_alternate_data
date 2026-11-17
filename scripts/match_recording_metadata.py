"""Corroborate Billboard chart identities against MusicBrainz recording metadata.

Queries the REDUCED form that the upstream lyrics.ovh lookup actually issued
(primary artist + title) because that is the risk the
`upstream_credit_query_reduced` flag describes. Metadata only: no lyrics, no audio,
no lyric text read or written.

This script NEVER sets recording_verified and never makes a candidate eligible for
Model B. Rights/permission and verified release dates are separate axes.

Usage:
  python scripts/match_recording_metadata.py --input <records.csv> --output <dir>
      [--cache <dir>] [--limit N] [--max-pages N] [--offline]

Input CSV needs: song_id, song, artist, first_observed_chart_date
(optional: observable_weight, stratum -- carried through to the output).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from music_recession.recording_metadata import assess, normalise, split_billboard_credit

USER_AGENT = ('music-recession-research/0.1 '
              '(NYU Predictive Analytics; metadata-only recording corroboration)')
ENDPOINT = 'https://musicbrainz.org/ws/2/recording/'
RATE_LIMIT_SECONDS = 1.1          # MusicBrainz asks for at most one request/second
# Calibrated against the pilot: recordings carrying the FULL featured credit score
# ~80-84 for a reduced primary-artist query, because the query artist is only part
# of the credit. A floor near 90 therefore discards exactly the evidence sought.
DEFAULT_MIN_SCORE = 75

OUTPUT_FIELDS = ['song_id', 'song', 'billboard_artist', 'primary_artist',
                 'featured_artists', 'first_observed_chart_date', 'observable_weight',
                 'stratum', 'reduced_query', 'evidence_level', 'recording_verified',
                 'mb_recording_mbid', 'mb_evidence_url', 'mb_exemplar_title',
                 'mb_exemplar_credit', 'earliest_original_release',
                 'earliest_any_release', 'candidate_count', 'exact_title_matches',
                 'guest_evidence', 'review_flags', 'reasoning',
                 'api_pages_fetched', 'api_result_count']


class Client:
    """Cached, rate-limited MusicBrainz recording search with pagination."""

    def __init__(self, cache: Path, limit=100, max_pages=3, offline=False):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.limit, self.max_pages, self.offline = limit, max_pages, offline
        self.requests = self.cache_hits = 0
        self._last = 0.0

    def _get(self, title, artist, offset):
        key = self.cache / f'{normalise(title + "|" + artist)[:110]}.l{self.limit}.o{offset}.json'
        if key.exists():
            self.cache_hits += 1
            return json.loads(key.read_text())
        if self.offline:
            return {'recordings': [], 'count': 0, 'offline_miss': True}
        query = f'recording:"{title}" AND artist:"{artist}"'
        url = (ENDPOINT + '?query=' + urllib.parse.quote(query)
               + f'&fmt=json&limit={self.limit}&offset={offset}')
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        for attempt in range(4):
            wait = RATE_LIMIT_SECONDS - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=40) as response:
                    payload = json.loads(response.read().decode())
                self._last = time.monotonic()
                self.requests += 1
                key.write_text(json.dumps(payload))
                return payload
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                self._last = time.monotonic()
                self.requests += 1
                if attempt == 3:
                    return {'recordings': [], 'count': 0, 'error': str(exc)}
                time.sleep(2 ** attempt)
        return {'recordings': [], 'count': 0}

    def search(self, title, artist):
        """All pages up to max_pages. Returns (recordings, pages, reported_count)."""
        first = self._get(title, artist, 0)
        recordings = list(first.get('recordings') or [])
        count, pages = int(first.get('count') or 0), 1
        while (len(recordings) < count and pages < self.max_pages
               and len(recordings) == self.limit * pages):
            page = self._get(title, artist, self.limit * pages)
            got = page.get('recordings') or []
            if not got:
                break
            recordings.extend(got)
            pages += 1
        return recordings, pages, count


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def run(input_path: Path, output_dir: Path, cache: Path, limit, max_pages, offline,
        min_score=DEFAULT_MIN_SCORE):
    records = read_csv_rows(input_path)
    required = {'song_id', 'song', 'artist', 'first_observed_chart_date'}
    missing = required - set(records[0] if records else {})
    if missing:
        raise ValueError('Input CSV missing required columns: ' + ', '.join(sorted(missing)))

    client = Client(cache, limit=limit, max_pages=max_pages, offline=offline)
    started = time.monotonic()
    rows = []
    for index, record in enumerate(records, 1):
        primary, _ = split_billboard_credit(record['artist'])
        recordings, pages, count = client.search(record['song'], primary)
        verdict = assess(record['song'], record['artist'],
                         record['first_observed_chart_date'], recordings,
                         min_score=min_score)
        rows.append({
            'song_id': record['song_id'], 'song': record['song'],
            'billboard_artist': record['artist'],
            'primary_artist': verdict['primary_artist'],
            'featured_artists': verdict['featured_artists'],
            'first_observed_chart_date': record['first_observed_chart_date'],
            'observable_weight': record.get('observable_weight', ''),
            'stratum': record.get('stratum', ''),
            'reduced_query': f'{primary} - {record["song"]}',
            'evidence_level': verdict['evidence_level'],
            'recording_verified': 'False',      # invariant: never set here
            'mb_recording_mbid': verdict['mb_recording_mbid'],
            'mb_evidence_url': verdict['mb_evidence_url'],
            'mb_exemplar_title': verdict['mb_exemplar_title'],
            'mb_exemplar_credit': verdict['mb_exemplar_credit'],
            'earliest_original_release': verdict['earliest_original_release'],
            'earliest_any_release': verdict['earliest_any_release'],
            'candidate_count': verdict['candidate_count'],
            'exact_title_matches': verdict['exact_title_matches'],
            'guest_evidence': verdict['guest_evidence'],
            'review_flags': verdict['review_flags'],
            'reasoning': verdict['reasoning'],
            'api_pages_fetched': pages, 'api_result_count': count})
        print(f"{index:>5}/{len(records)} {verdict['evidence_level']:<26} "
              f"{record['song'][:30]:<30} {verdict['review_flags'][:46]}", flush=True)

    elapsed = time.monotonic() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'recording_metadata_evidence.csv', 'w', newline='',
              encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    levels = {}
    for row in rows:
        levels[row['evidence_level']] = levels.get(row['evidence_level'], 0) + 1
    summary = dict(
        records=len(rows), evidence_levels=levels,
        network_requests=client.requests, cache_hits=client.cache_hits,
        elapsed_seconds=round(elapsed, 1),
        seconds_per_record=round(elapsed / len(rows), 2) if rows else 0,
        pages_per_record=round(sum(r['api_pages_fetched'] for r in rows) / len(rows), 3)
        if rows else 0,
        rate_limit_seconds=RATE_LIMIT_SECONDS, search_limit=limit, max_pages=max_pages,
        min_score=min_score,
        scope='Metadata corroboration only. recording_verified is never set here; '
              'rights/permission and verified release dates are separate axes. '
              'No candidate becomes eligible for Model B.')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\n' + json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('.agents/mb_cache_v2'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--max-pages', type=int, default=3)
    parser.add_argument('--min-score', type=int, default=DEFAULT_MIN_SCORE,
                        help='MusicBrainz search score floor. Scores fall to ~80 when the '
                             'query artist is a subset of a fuller featured credit, which is '
                             'exactly the case under study, so this must not be set near 100.')
    parser.add_argument('--offline', action='store_true',
                        help='Use cached responses only; never touch the network')
    args = parser.parse_args(argv)
    run(args.input, args.output, args.cache, args.limit, args.max_pages, args.offline,
        min_score=args.min_score)


if __name__ == '__main__':
    main()

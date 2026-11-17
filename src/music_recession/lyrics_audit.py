"""Offline, metadata-only audit of local lyric candidates; never verifies recordings.

No text is written, no network access is performed, and existing inventories are
never updated. CSV record numbers are 1-based data records (not physical lines).
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

csv.field_size_limit(16 * 1024 * 1024)
REFERENCE_FIELDS = ['song', 'artist', 'source', 'source_path', 'member', 'record_number',
                    'source_title', 'source_artist', 'source_year', 'source_url',
                    'content_sha256', 'characters', 'source_occurrences', 'method',
                    'flags', 'language_status', 'release_date', 'source_chart_date',
                    'source_chart_rank', 'featured_artists']


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(title, artist):
    # Preserve the existing inventory's identity convention, NOT a recording ID.
    return title.strip().casefold(), artist.strip().casefold()


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = value.replace('’', "'").replace('‘', "'")
    return ' '.join(re.sub(r'[^\w]+', ' ', value).split())


def credits(value):
    # Only explicit featured-credit markers; never drop contributors or reorder.
    return normalize(re.sub(r'\b(?:feat\.?|ft\.?|featuring)\s+', ' featuring ', value, flags=re.I))


def candidate_key(title, artist):
    return normalize(title), credits(artist)


def lyric_flags(text, instrumental=False):
    flags = set()
    stripped = text.strip()
    if not stripped or stripped.casefold() in {'nan', 'none', 'null', 'n/a'}:
        flags.add('missing_text')
    elif len(stripped) < 40 or len(stripped.split()) < 8:
        flags.add('short_text_review')
    if len(stripped) > 50000:
        flags.add('unusually_long_text_review')
    if instrumental or re.fullmatch(r'\[?instrumental\]?[.!]?', stripped, re.I):
        flags.add('instrumental_review')
    if re.search(r'<!doctype|<html|access denied|lyrics (?:not found|unavailable)|404 not found', text, re.I):
        flags.add('error_or_html')
    if re.search(r'you might also like|\d+Embed\b|\d+ Contributors\b|Read More', text):
        flags.add('scraper_boilerplate_review')
    # Script heuristic only: not language identification; Latin multilingual text
    # remains unknown and must be reviewed too.
    if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text):
        flags.add('non_latin_script_review')
    return flags


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None and not rows:
        # An empty result must name its own schema rather than raise IndexError.
        raise ValueError('Explicit field names are required for an empty report: ' + str(path))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def eligible_observation(chart_date, predictor_month, available_on=None):
    """Month membership AND availability by the assumed 15th-of-next-month cutoff.

    Caller must supply evidence-backed available_on for real-time claims. Missing
    availability uses chart date solely as a retrospective convention.
    """
    first = date.fromisoformat(predictor_month + '-01')
    following = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    cutoff = following.replace(day=15)
    observation = date.fromisoformat(chart_date)
    available = date.fromisoformat(available_on or chart_date)
    return first <= observation < following and available <= cutoff


def source_records(root):
    """Yield local records with stable file/member/record locators. Never extract ZIPs."""
    raw = root / 'data/raw/lyrics'
    p = raw / 'osf_lyrics.csv'
    for number, r in enumerate(read_csv(p), 1):
        yield 'osf', p, '', number, r['Song Title'], r['Artist'], r.get('lyrics_raw', ''), r['Date'][:4], '', False, r['Date'], r['Rank'], '', ''
    p = raw / 'kaggle/top100.zip'
    with zipfile.ZipFile(p) as z:
        member = 'all_songs_data.csv'
        with z.open(member) as f:
            for number, r in enumerate(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig')), 1):
                try:
                    featured = ast.literal_eval(r.get('Featured Artists') or '[]')
                    featured = ';'.join(a.get('name', '') if isinstance(a, dict) else str(a) for a in featured)
                except (ValueError, SyntaxError, TypeError):
                    featured = 'unparseable_featured_credits'
                yield 'kaggle', p, member, number, r['Song Title'], r['Artist'], r['Lyrics'], r['Year'].split('.')[0], r['Song URL'], False, '', '', r.get('Release Date', ''), featured
    p = raw / 'lydia/lydia.zip'
    with zipfile.ZipFile(p) as z:
        for member in sorted(z.namelist()):
            if not member.startswith('Lyrics Project Data/Lyrics/') or not member.endswith('.json'):
                continue
            r = json.loads(z.read(member))
            featured = ';'.join(a.get('name', '') for a in r.get('featured_artists', []))
            yield 'lydia', p, member, 1, r['title'], r['artist'], r.get('lyrics') or '', member.split('/')[-2], r.get('url', ''), r.get('instrumental') is True, '', '', '', featured
    for p in sorted((raw / 'tmelo_uea').glob('billboard-year-end-*-lyrics.csv')):
        year = p.name.split('-')[3]
        rows = list(read_csv(p))
        malformed = any(None in r or any(v is None for v in r.values()) or
                        not str(r.get('Rank', '')).isdigit() for r in rows)
        for number, r in enumerate(rows, 1):
            if malformed:
                # Broken multiline quoting can truncate earlier valid-looking
                # records too. Quarantine the entire file, never export fragments.
                yield 'tmelo', p, '', number, None, None, None, year, '', False, '', '', '', ''
                continue
            yield 'tmelo', p, '', number, r['Musica'], r['Artista'], r.get('Letra') or '', year, '', False, '', '', '', ''


def load_supplement(root, manifest_path):
    """Optional local songdata.csv adapter, gated by documented use authorization.

    A boolean is not legal advice: reviewer/evidence must be supplied by the user
    or institution. This adapter does not fetch or license material itself.
    """
    m = json.loads(manifest_path.read_text())
    if not (m.get('authorized_local_analysis') is True and m.get('permission_evidence') and m.get('reviewer')):
        raise ValueError('Supplement requires documented permission and reviewer')
    p = (root / m['path']).resolve()
    if (root / 'data/raw/lyrics').resolve() not in p.parents:
        raise ValueError('Supplement must remain in ignored data/raw/lyrics')
    if file_hash(p) != m['sha256']:
        raise ValueError('Supplement hash mismatch')
    for number, r in enumerate(read_csv(p), 1):
        if not {'artist', 'song', 'text'} <= r.keys():
            raise ValueError('Expected songdata.csv artist/song/text schema')
        yield 'lyricsfreak', p, '', number, r['song'], r['artist'], r['text'], '', r.get('link', ''), False, '', '', '', ''


def audit(root, output, supplement=None):
    if output.exists():
        raise ValueError('Output already exists; choose a new run directory to preserve results')
    charts = []
    slots = {}
    by_song = defaultdict(list)
    same_date = Counter()
    for r in read_csv(root / 'billboard_hot100_all.csv'):
        if not '1973-01-01' <= r['date'] <= '2023-12-31':
            continue
        date.fromisoformat(r['date'])
        rank = int(r['this_week'])
        if not 1 <= rank <= 100 or not r['song'].strip() or not r['artist'].strip():
            raise ValueError('Invalid Billboard row')
        slot = r['date'], str(rank)
        if slot in slots:
            raise ValueError('Duplicate Billboard date/rank')
        key = identity(r['song'], r['artist'])
        slots[slot] = key
        r['_key'] = key
        charts.append(r)
        by_song[key].append(r)
        same_date[(r['date'], key)] += 1
    collisions = {key for (_, key), count in same_date.items() if count > 1}
    if not charts:
        raise ValueError('No study-period Billboard observations')
    inventory = defaultdict(list)
    for r in read_csv(root / 'data/metadata/song_lyrics_inventory.csv'):
        inventory[identity(r['song'], r['artist'])].append(r)
    index = defaultdict(set)
    for key, rows in by_song.items():
        index[candidate_key(rows[0]['song'], rows[0]['artist'])].add(key)
    aliases = defaultdict(set)
    explicit = defaultdict(list)
    mapping_issues = []
    paths = sorted((root / 'config').glob('*aliases.csv')) + sorted((root / 'config').glob('*sources.csv'))
    mapping_rows = []
    for p in paths:
        for number, r in enumerate(read_csv(p), 1):
            target = identity(r['song'], r['billboard_artist'])
            mapping_rows.append((str(p.relative_to(root)), number, r, target))
            if target not in by_song:
                mapping_issues.append(dict(mapping=p.name, record=number, issue='target_outside_study_universe'))
            if 'osf_artist' in r:
                aliases[('osf', candidate_key(r['song'], r['osf_artist']))].add(target)
            elif 'kaggle_artist' in r:
                aliases[('kaggle', candidate_key(r['song'], r['kaggle_artist']))].add(target)
            elif 'lydia_file' in r:
                explicit[('lydia', r['lydia_file'], 1)].append((target, r, p.name, number))
            elif 'source_csv' in r:
                explicit[('tmelo', r['source_csv'], int(r['source_record_number']))].append((target, r, p.name, number))
            else:
                mapping_issues.append(dict(mapping=p.name, record=number, issue='missing_source_identity'))
    candidates = defaultdict(dict)
    file_manifest = {}
    source_counts = Counter()
    source_headers = {}
    seen_explicit = set()
    missing_text_counts = Counter()
    source_issues = []
    hash_keys = defaultdict(set)
    flag_cache = {}
    streams = [source_records(root)]
    if supplement:
        streams.append(load_supplement(root, supplement))
    for stream in streams:
        for source, path, member, number, title, artist, text, year, url, instrumental, chart_date, rank, release, featured in stream:
            relative = str(path.relative_to(root))
            if relative not in file_manifest:
                file_manifest[relative] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
                if source == 'osf':
                    with path.open(encoding='utf-8-sig', newline='') as f:
                        headers = next(csv.reader(f))
                    source_headers[relative] = {'duplicate_headers': [k for k, v in Counter(headers).items() if v > 1]}
            source_counts[source] += 1
            # No 2024+ year-end records or OSF observations enter this study.
            if year and not 1973 <= int(year) <= 2023:
                continue
            if not isinstance(title, str) or not isinstance(artist, str) or not title.strip() or not artist.strip() or not isinstance(text, str):
                issue = 'quarantined_malformed_csv' if source == 'tmelo' and title is None and text is None else 'invalid_required_metadata_or_text_type'
                source_issues.append(dict(source=source, source_path=relative, member=member, record_number=number, issue=issue))
                continue
            sk = candidate_key(title, artist)
            targets = {k: 'normalized_title_and_full_credit' for k in index.get(sk, ())}
            targets.update({k: 'explicit_artist_alias' for k in aliases.get((source, sk), ()) if k in by_song})
            explicit_key = (source, member if source == 'lydia' else path.name, number)
            for target, mapping, mapping_name, mapping_number in explicit.get(explicit_key, []):
                seen_explicit.add((mapping_name, mapping_number))
                errors = []
                expected_title = mapping.get('lydia_title', mapping.get('source_title', mapping.get('component_title', '')))
                expected_artist = mapping.get('lydia_artist', mapping.get('source_artist', ''))
                if normalize(expected_title) != normalize(title):
                    errors.append('source_title_mismatch')
                if expected_artist and credits(expected_artist) != credits(artist):
                    errors.append('source_artist_mismatch')
                if mapping.get('lyrics_sha256') and mapping['lyrics_sha256'] != sha(text.encode()):
                    if mapping['lyrics_sha256'] == sha(text.strip().encode()):
                        mapping_issues.append(dict(mapping=mapping_name, record=mapping_number, issue='hash_matches_trimmed_text_only'))
                    else:
                        errors.append('content_hash_mismatch')
                if str(mapping.get('lydia_year', mapping.get('source_year', year))) != year:
                    errors.append('source_year_mismatch')
                for error in errors:
                    mapping_issues.append(dict(mapping=mapping_name, record=mapping_number, issue=error))
                if target in by_song and not errors:
                    targets[target] = 'explicit_component_review' if 'component_title' in mapping else 'explicit_record_mapping'
            digest = sha(text.encode('utf-8'))
            cache_key = digest, instrumental
            if cache_key not in flag_cache:
                flag_cache[cache_key] = lyric_flags(text, instrumental)
            flags = flag_cache[cache_key]
            if 'missing_text' in flags:
                missing_text_counts[source] += 1
            if not targets:
                continue
            for key, method in targets.items():
                local_flags = set(flags)
                if featured:
                    local_flags.add('featured_credits_review')
                if not year:
                    local_flags.add('source_year_unknown')
                version_words = {'remix', 'mix', 'live', 'remastered', 'acoustic', 'instrumental'}
                chart_versions = set(normalize(by_song[key][0]['song']).split()) & version_words
                source_versions = set(normalize(title + ' ' + url).split()) & version_words
                if chart_versions != source_versions:
                    local_flags.add('version_metadata_conflict')
                if method.startswith('explicit_'):
                    local_flags.add('explicit_mapping_needs_review')
                if year and int(year) not in {int(r['date'][:4]) for r in by_song[key]}:
                    local_flags.add('source_year_not_chart_year')
                if release:
                    try:
                        if date.fromisoformat(release) > date.fromisoformat(by_song[key][0]['date']):
                            local_flags.add('release_after_first_chart_review')
                    except ValueError:
                        local_flags.add('invalid_release_date')
                if source == 'osf':
                    if slots.get((chart_date, rank)) != key:
                        # Do not certify an OSF candidate from matching text alone.
                        local_flags.add('chart_slot_mismatch')
                    else:
                        method += '+chart_slot'
                if key in collisions:
                    local_flags.add('concurrent_recordings_same_identity')
                if '/' in by_song[key][0]['song'] or method == 'explicit_component_review':
                    local_flags.add('compound_title_review')
                token = (source, title, artist, digest, method, tuple(sorted(local_flags)), year)
                if token in candidates[key]:
                    candidates[key][token]['source_occurrences'] += 1
                    continue
                hash_keys[digest].add(key)
                candidates[key][token] = dict(source=source, source_path=relative, member=member, record_number=number,
                    source_title=title, source_artist=artist, source_year=year, source_url=url,
                    content_sha256=digest, characters=len(text), source_occurrences=1,
                    method=method, flags=';'.join(sorted(local_flags)), language_status='unverified',
                    release_date=release, source_chart_date=chart_date, source_chart_rank=rank,
                    featured_artists=featured)
    for (source, locator, record), mappings in explicit.items():
        for target, mapping, name, number in mappings:
            if (name, number) not in seen_explicit:
                mapping_issues.append(dict(mapping=name, record=number, issue='source_record_not_resolved'))
    lineage = []
    audited = []
    covered = set()
    clean_candidates = set()
    old = set()
    for key, rows in sorted(by_song.items()):
        inv = inventory.get(key, [])
        previous = inv[0]['lyrics_status'] if len(inv) == 1 else 'missing_or_duplicate_inventory'
        if previous.startswith('matched_'):
            old.add(key)
        issues = set()
        if len(inv) != 1:
            issues.add('inventory_missing_or_duplicate')
        found = list(candidates[key].values())
        accessible = []
        for c in found:
            flags = set(filter(None, c['flags'].split(';')))
            if len(hash_keys[c['content_sha256']]) > 1 and c['characters'] >= 40:
                flags.add('same_content_multiple_chart_identities')
            c['flags'] = ';'.join(sorted(flags))
            issues.update(flags)
            # Availability is NOT correctness. Suspicious text is excluded here.
            if not flags & {'missing_text', 'error_or_html', 'instrumental_review', 'short_text_review', 'chart_slot_mismatch', 'unusually_long_text_review'}:
                accessible.append(c)
            lineage.append(dict(song=rows[0]['song'], artist=rows[0]['artist'], **c))
        if accessible:
            covered.add(key)
        if any(not c['flags'] for c in accessible):
            clean_candidates.add(key)
        if len({c['content_sha256'] for c in accessible}) > 1:
            issues.add('multiple_content_variants')
        if previous.startswith('matched_') and not accessible:
            issues.add('previous_match_not_supported')
        if previous.startswith('matched_'):
            expected = {'OSF:2k7ut': 'osf', 'Kaggle:brianblakely': 'kaggle', 'LYDIA:Bonela2024': 'lydia', 'tmelo-uea:billboard-hot100-lyrics': 'tmelo'}.get(inv[0]['lyrics_source'])
            if not any(c['source'] == expected for c in accessible):
                issues.add('claimed_source_not_supported')
        audited.append(dict(song_id=inv[0]['song_id'] if len(inv) == 1 else '', song=rows[0]['song'], artist=rows[0]['artist'],
            first_study_chart_date=min(r['date'] for r in rows), last_study_chart_date=max(r['date'] for r in rows),
            prior_status=previous, audit_status='provisional_accessible' if accessible else 'unmatched_or_unusable',
            verified_recording=False, accessible_candidates=len(accessible), content_variants=len({c['content_sha256'] for c in accessible}),
            popularity_weight=sum(101-int(r['this_week']) for r in rows), flags=';'.join(sorted(issues))))
    def coverage(group):
        keys = {r['_key'] for r in group}
        weights = sum(101-int(r['this_week']) for r in group)
        covered_weight = sum(101-int(r['this_week']) for r in group if r['_key'] in covered)
        return dict(songs=len(keys), provisional_songs=len(keys & covered), verified_songs=0,
                    song_coverage_pct=round(100*len(keys & covered)/len(keys), 6),
                    chart_entries=len(group), covered_entries=sum(r['_key'] in covered for r in group),
                    entry_coverage_pct=round(100*sum(r['_key'] in covered for r in group)/len(group), 6),
                    popularity_weight=weights, covered_weight=covered_weight,
                    weighted_coverage_pct=round(100*covered_weight/weights, 6), verified_weighted_coverage_pct=0)
    period_reports = {}
    for name, length in [('yearly', 4), ('monthly', 7)]:
        groups = defaultdict(list)
        for r in charts:
            # Only observations in the predictor month; no next-month chart rows.
            if not eligible_observation(r['date'], r['date'][:7]):
                raise ValueError('Invalid retrospective chart timing')
            groups[r['date'][:length]].append(r)
        period_reports[name] = [dict(period=p, **coverage(group)) for p, group in sorted(groups.items())]
    summary = dict(scope='1973-01-01 through 2023-12-31; retrospective candidate availability, not real-time features',
        **coverage(charts), prior_candidate_songs=len(old), prior_unmatched_songs=len(by_song)-len(old),
        newly_accessible_candidates=len(covered-old), prior_candidates_without_usable_reference=len(old-covered),
        remaining_unmatched_or_unusable=len(by_song)-len(covered), candidates_without_automated_flags=len(clean_candidates),
        source_records_scanned=dict(source_counts), source_missing_text_records=dict(missing_text_counts),
        song_issue_counts=dict(Counter(flag for r in audited for flag in r['flags'].split(';') if flag)),
        mapping_issue_count=len(mapping_issues), source_issue_count=len(source_issues), concurrent_identity_count=len(collisions),
        lowest_five_years=sorted(period_reports['yearly'], key=lambda r:r['weighted_coverage_pct'])[:5])
    protected = ['billboard_hot100_all.csv', 'data/metadata/song_lyrics_inventory.csv'] + [str(p.relative_to(root)) for p in paths]
    if (root/'config/lyrics_sources.json').exists():
        protected.append('config/lyrics_sources.json')
    manifest = dict(schema_version=1, inputs={**file_manifest, **{p:{'sha256':file_hash(root/p), 'bytes':(root/p).stat().st_size} for p in protected}},
        source_schema_findings=source_headers, record_number_convention='1-based CSV data record; JSON member is record 1',
        content_hash_convention='SHA-256 of decoded original lyric field encoded as UTF-8, no trimming',
        code_sha256=file_hash(Path(__file__)), supplement_manifest=json.loads(supplement.read_text()) if supplement else None)
    output.mkdir(parents=True)
    write_csv(output/'inventory_audited.csv', audited)
    write_csv(output/'source_references.csv', lineage, REFERENCE_FIELDS)
    write_csv(output/'mapping_issues.csv', mapping_issues, ['mapping', 'record', 'issue'])
    write_csv(output/'source_issues.csv', source_issues, ['source', 'source_path', 'member', 'record_number', 'issue'])
    for name, rows in period_reports.items():
        write_csv(output/(name+'_coverage.csv'), rows)
    write_csv(output/'manual_review.csv', sorted((r for r in audited if r['flags'] or r['audit_status'] != 'provisional_accessible'), key=lambda r:-r['popularity_weight']), list(audited[0]))
    for name, data in [('summary.json', summary), ('manifest.json', manifest)]:
        (output/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--supplement-manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), args.output.resolve(), args.supplement_manifest), indent=2))


if __name__ == '__main__':
    main()

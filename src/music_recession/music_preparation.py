"""Deterministic metadata-only Model B preparation. No inference or fitting.

Source selection is provisional. Recording/language/permission evidence gates
NLP; chart-date availability is only a retrospective timing convention.
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from .baseline import BaselineConfig, load_economic
from .economic import add_months
from .lyrics_audit import file_hash, identity, read_csv, sha, write_csv

EMOTIONS = ['joy', 'sadness', 'fear', 'anger', 'nervousness', 'optimism']
REVIEW_FIELDS = ['song_id', 'source_reference_id', 'content_sha256', 'reviewer',
                 'recording_evidence', 'language', 'language_evidence',
                 'permission_evidence', 'release_date', 'release_evidence']


def flags(value):
    return set(filter(None, value.split(';')))


def reference_id(r):
    return sha(json.dumps([r[k] for k in ['source', 'source_path', 'member',
        'record_number', 'content_sha256']], separators=(',', ':')).encode())


EXPLORATORY_MODE = 'exploratory_course'
EXPLORATORY_BASIS = 'exploratory_automated_screen'


def validate_policy(policy, baseline):
    """Strict policy by default; an explicitly declared exploratory course policy
    is also accepted. The exploratory policy admits automatically screened
    provisional records and NEVER asserts verified recordings, reviewed language,
    verified release dates or cleared rights. See docs/exploratory_course_policy.md.
    """
    if policy.get('analysis_mode') == EXPLORATORY_MODE:
        fixed = dict(schema_version=1, analysis_mode=EXPLORATORY_MODE,
            start_month='1973-01-01', end_month='2023-12-01',
            allowed_sources=['osf'], emotions=EMOTIONS, instrumentals='exclude_no_zero_vector',
            combined_singles='exclude_pending_component_and_weight_evidence',
            languages='automated_screen_english_unreviewed', unknown_languages='block_nlp',
            year_end_sources='exclude_from_primary_mapping',
            ambiguous_versions='exclude_no_automatic_resolution',
            unresolved_aliases='exclude_target_identity',
            recording_and_permission_evidence='reported_instructor_guidance_exploratory_not_verified',
            release_date_evidence='chart_observability_only_no_verified_release_date',
            source_use_basis='reported_instructor_guidance_course_project_not_a_license',
            chart_timing_basis='retrospective_chart_date_convention')
    else:
        fixed = dict(schema_version=1, start_month='1973-01-01', end_month='2023-12-01',
            allowed_sources=['osf'], emotions=EMOTIONS, instrumentals='exclude_no_zero_vector',
            combined_singles='exclude_pending_component_and_weight_evidence',
            languages='reviewed_english_only', unknown_languages='block_nlp',
            year_end_sources='exclude_from_primary_mapping', ambiguous_versions='exclude_no_automatic_resolution',
            unresolved_aliases='exclude_target_identity', recording_and_permission_evidence='required_for_nlp',
            chart_timing_basis='retrospective_chart_date_convention')
    for key, value in fixed.items():
        if policy.get(key) != value:
            raise ValueError('Unsupported preparation policy: '+key)
    if type(policy['chart_publication_delay_days']) is not int or policy['chart_publication_delay_days'] < 0:
        raise ValueError('Publication delay must be a nonnegative integer')
    if not 0 < policy['minimum_ready_weight_coverage'] <= 1:
        raise ValueError('Invalid coverage threshold')
    if (baseline.min_training_months, baseline.horizon_months,
        baseline.training_target_lag_months, baseline.forecast_cutoff_month_offset,
        baseline.forecast_cutoff_day) != (240, 6, 1, 1, 15):
        raise ValueError('Model A timing contract changed; review comparison protocol')


def select_mapping(song, references, alias_excluded=False, review=None, screen=None):
    """One selected locator or an explicit exclusion; never infer approval."""
    reasons = set()
    aggregate = flags(song['flags'])
    if alias_excluded:
        reasons.add('unresolved_alias_identity')
    for flag, reason in [('instrumental_review', 'instrumental_pending_review'),
                         ('compound_title_review', 'combined_or_slash_title'),
                         ('concurrent_recordings_same_identity', 'concurrent_recordings')]:
        if flag in aggregate:
            reasons.add(reason)
    primary = [r for r in references if r['source'] == 'osf']
    # All automated source warnings are blocking for automatic source selection.
    candidates = [r for r in primary if not flags(r['flags']) and
                  '+chart_slot' in r['method'] and int(r['characters']) >= 40]
    if not candidates:
        reasons.add('no_clean_weekly_source')
    if len({r['content_sha256'] for r in candidates}) > 1:
        reasons.add('conflicting_weekly_text_variants')
    selected = None
    if not reasons:
        selected = min(candidates, key=lambda r:(r['source_chart_date'], int(r['record_number']), reference_id(r)))
    row = dict(song_id=song['song_id'], song=song['song'], artist=song['artist'],
        mapping_status='provisional_selected' if selected else 'excluded',
        source_reference_id=reference_id(selected) if selected else '',
        source=selected['source'] if selected else '', source_path=selected['source_path'] if selected else '',
        member=selected['member'] if selected else '', record_number=selected['record_number'] if selected else '',
        content_sha256=selected['content_sha256'] if selected else '',
        source_chart_date=selected['source_chart_date'] if selected else '',
        source_chart_rank=selected['source_chart_rank'] if selected else '',
        source_title=selected['source_title'] if selected else '', source_artist=selected['source_artist'] if selected else '',
        source_url=selected['source_url'] if selected else '',
        quality_flags=song['flags'], exclusion_reasons=';'.join(sorted(reasons)),
        language='unknown', release_date='', recording_verified=False, nlp_ready=False,
        readiness_blockers='source_excluded' if not selected else 'recording_evidence_missing;language_unknown;permission_evidence_missing;release_evidence_missing',
        reviewer='', recording_evidence='', language_evidence='', permission_evidence='', release_evidence='',
        eligibility_basis='')
    if review and screen:
        raise ValueError('Supply a strict review record or an exploratory screen row, not both')
    if screen is not None:
        # EXPLORATORY COURSE POLICY ONLY. Automated screens are not verification:
        # recording_verified stays False, release_date is left EMPTY rather than
        # back-filled from a chart date, and every outstanding gap is recorded in
        # readiness_blockers so nothing here can be mistaken for approved evidence.
        if not selected:
            return row
        if screen.get('automated_screen_pass') not in (True, 'True'):
            row['readiness_blockers'] = 'exploratory_automated_screen_not_passed'
            return row
        if screen.get('language_status') != 'english_automated_consistent':
            row['readiness_blockers'] = 'exploratory_language_screen_not_consistent_english'
            return row
        row.update(language='en', nlp_ready=True, recording_verified=False,
            eligibility_basis=EXPLORATORY_BASIS,
            reviewer='exploratory_course_policy_no_human_review',
            recording_evidence='automated_metadata_and_chart_slot_match_only_not_independently_verified',
            language_evidence='automated_langid_consistent_english_not_human_reviewed',
            permission_evidence=('reported_instructor_guidance_for_course_project'
                                 '_not_a_license_or_rights_clearance'),
            release_date='',
            release_evidence='chart_observability_only_no_verified_release_date',
            readiness_blockers=('recording_not_independently_verified;language_not_human_reviewed;'
                                'release_date_not_verified;source_use_rights_unresolved'))
        return row
    if review:
        if not selected or review['source_reference_id'] != row['source_reference_id'] or review['content_sha256'] != row['content_sha256']:
            raise ValueError('Review points to excluded, stale, or different source: '+song['song_id'])
        if any(not review.get(k, '').strip() for k in REVIEW_FIELDS):
            raise ValueError('Incomplete review evidence: '+song['song_id'])
        date.fromisoformat(review['release_date'])
        row.update({k:review[k] for k in ['reviewer', 'recording_evidence', 'language_evidence',
            'permission_evidence', 'release_date', 'release_evidence']})
        row['language'] = review['language']
        row['recording_verified'] = True  # only a complete, named evidence record
        row['nlp_ready'] = review['language'] == 'en'
        row['readiness_blockers'] = '' if row['nlp_ready'] else 'non_english_or_multilingual_excluded'
    return row


def replay_references(root, rows, source_manifest):
    """Check file hashes and replay selected locators without exporting text."""
    grouped = defaultdict(dict)
    for r in rows:
        if r['mapping_status'] != 'provisional_selected':
            continue
        path = r['source_path']
        resolved = (root/path).resolve()
        if (root/'data/raw/lyrics').resolve() not in resolved.parents:
            raise ValueError('Raw source outside ignored lyrics directory')
        if path not in source_manifest['inputs']:
            raise ValueError('Source absent from audit manifest')
        number = int(r['record_number'])
        if number < 1:
            raise ValueError('Invalid source locator')
        grouped[path].setdefault(number, []).append(r)
    count = 0
    for path, wanted in sorted(grouped.items()):
        if file_hash(root/path) != source_manifest['inputs'][path]['sha256']:
            raise ValueError('Changed raw source: '+path)
        # The primary policy is weekly OSF only. Do not reinterpret other schemas.
        for number, actual in enumerate(read_csv(root/path), 1):
            for r in wanted.pop(number, []):
                if (actual['Song Title'], actual['Artist'], actual['Date'], actual['Rank'], sha(actual['lyrics_raw'].encode())) != (
                    r['source_title'], r['source_artist'], r['source_chart_date'], r['source_chart_rank'], r['content_sha256']):
                    raise ValueError('Source reference replay failed: '+r['song_id'])
                count += 1
        if wanted:
            raise ValueError('Unresolved source records')
    return count


def month_range(start, end):
    while start <= end:
        yield start
        start = add_months(start, 1)


def prepare_months(charts, mapping, economic, policy):
    """Weight only observations in t, known by assumed cutoff in t+1.

    Returns song/month contributions and monthly preparation rows. Emotion values
    are empty, never fabricated zeros. Release review gates approved contributions.
    """
    grouped = defaultdict(list)
    seen = set()
    by_key = {identity(r['song'], r['artist']):r for r in mapping}
    if len(by_key) != len(mapping) or len({r['song_id'] for r in mapping}) != len(mapping):
        raise ValueError('Duplicate mapping identities or IDs')
    for r in charts:
        day = date.fromisoformat(r['date'])
        if not date(1973,1,1) <= day <= date(2023,12,31):
            continue
        rank = int(r['this_week'])
        if not 1 <= rank <= 100 or (day, rank) in seen:
            raise ValueError('Invalid or duplicate chart slot')
        seen.add((day, rank))
        key = identity(r['song'], r['artist'])
        if key not in by_key:
            raise ValueError('Chart identity absent from mapping')
        grouped[day.replace(day=1)].append((r, by_key[key]))
    contributions, months = [], []
    for month in month_range(date(1973,1,1), date(2023,12,1)):
        cutoff = add_months(month, 1).replace(day=15)
        per_song = {}
        dates = set()
        for raw, mapped in grouped[month]:
            key = mapped['song_id']
            day = date.fromisoformat(raw['date'])
            dates.add(day)
            if key not in per_song:
                per_song[key] = dict(predictor_month=month.isoformat(), forecast_cutoff=cutoff.isoformat(),
                    song_id=key, source_reference_id=mapped['source_reference_id'],
                    chart_entries=0, total_weight=0, known_weight=0, provisional_weight=0,
                    nlp_ready_weight=0, excluded_weight=0, first_chart_date=day.isoformat(), last_chart_date=day.isoformat(),
                    reasons=set(filter(None, (mapped['exclusion_reasons']+';'+mapped['readiness_blockers']).split(';'))))
            c = per_song[key]
            w = 101-int(raw['this_week'])
            c['chart_entries'] += 1
            c['total_weight'] += w
            c['first_chart_date'] = min(c['first_chart_date'], day.isoformat())
            c['last_chart_date'] = max(c['last_chart_date'], day.isoformat())
            available = day + timedelta(days=policy['chart_publication_delay_days'])
            if available > cutoff:
                c['reasons'].add('chart_after_cutoff')
                continue
            c['known_weight'] += w
            if mapped['mapping_status'] != 'provisional_selected':
                continue
            # Prevent a later chart lookup exemplar from backfilling earlier origins.
            if date.fromisoformat(mapped['source_chart_date']) > cutoff:
                c['reasons'].add('source_chart_reference_after_cutoff')
                continue
            c['provisional_weight'] += w
            if mapped['nlp_ready']:
                # Strict policy always carries a validated release_date, so this
                # branch is unchanged for it. The exploratory policy has no verified
                # release date and relies on chart observability alone, which is
                # recorded in release_evidence rather than invented here.
                release = mapped['release_date']
                if release and date.fromisoformat(release) > day:
                    c['reasons'].add('recording_release_after_chart')
                else:
                    c['nlp_ready_weight'] += w
        rows = list(per_song.values())
        for r in rows:
            r['excluded_weight'] = r['total_weight']-r['nlp_ready_weight']
            r['reasons'] = ';'.join(sorted(r['reasons']))
        contributions.extend(sorted(rows, key=lambda r:r['song_id']))
        totals = {k:sum(r[k] for r in rows) for k in ['chart_entries', 'total_weight', 'known_weight', 'provisional_weight', 'nlp_ready_weight', 'excluded_weight']}
        n = totals['total_weight']
        ready_ratio = totals['nlp_ready_weight']/n if n else 0
        econ = economic.get(month)
        if econ and econ.target_month != add_months(month, 6):
            raise ValueError('Economic target misalignment')
        # Historical weekly issue-day conventions vary: >8-day gaps or fewer
        # than four charts are conservatively incomplete, never imputed.
        ordered = sorted(dates)
        incomplete = (len(dates) < 4 or (ordered and (ordered[0].day > 8 or
            (add_months(month,1)-ordered[-1]).days > 8 or
            any((b-a).days > 8 for a,b in zip(ordered,ordered[1:])))))
        rank_slots_missing = len(dates)*100-totals['chart_entries']
        eligible = bool(econ and econ.target is not None and not incomplete and rank_slots_missing == 0 and
                        ready_ratio >= policy['minimum_ready_weight_coverage'])
        months.append(dict(predictor_month=month.isoformat(), target_month=add_months(month,6).isoformat(),
            forecast_cutoff=cutoff.isoformat(), spread_t=econ.spread if econ else '', target_t=econ.target if econ else '',
            **totals, chart_dates=len(dates), missing_rank_slots=rank_slots_missing, incomplete_chart_month=bool(incomplete),
            provisional_weight_coverage=totals['provisional_weight']/n if n else 0,
            nlp_ready_weight_coverage=ready_ratio, preparation_eligible=eligible,
            emotion_status='not_generated', **{label:'' for label in EMOTIONS}))
    return contributions, months


def comparison_plan(months):
    """Shared Model A/B calendar contract; no fit, forecast or Brier calculation."""
    result = []
    for r in months:
        t = date.fromisoformat(r['predictor_month'])
        latest = add_months(t,-1)
        training = [s['predictor_month'] for s in months if s['preparation_eligible'] and
                    date.fromisoformat(s['target_month']) <= latest and s['target_t'] in (0,1)]
        common = bool(r['preparation_eligible'] and len(training) >= 240)
        result.append(dict(predictor_month=t.isoformat(), target_month=r['target_month'],
            forecast_cutoff=r['forecast_cutoff'], latest_allowed_training_target=latest.isoformat(),
            training_count=len(training), training_months=';'.join(training),
            common_preparation_eligible=common, runnable=False,
            blocker='emotion_features_not_generated' if common else 'insufficient_ready_months_or_current_month_excluded'))
    return result


def build(root, audit_dir, output, config_path=None, reviews_path=None):
    root = root.resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory; existing results are preserved')
    config_path = config_path or root/'config/music_preparation.json'
    reviews_path = reviews_path or root/'config/music_recording_reviews.csv'
    policy = json.loads(config_path.read_text())
    baseline = BaselineConfig.load(root/'config/baseline.json')
    validate_policy(policy, baseline)
    manifest = json.loads((audit_dir/'manifest.json').read_text())
    for path, value in manifest['inputs'].items():
        if file_hash(root/path) != value['sha256']:
            raise ValueError('Audit input changed: '+path)
    songs = list(read_csv(audit_dir/'inventory_audited.csv'))
    if len({r['song_id'] for r in songs}) != len(songs) or any(not r['song_id'] for r in songs):
        raise ValueError('Missing/duplicate song IDs in audit')
    references = defaultdict(list)
    for r in read_csv(audit_dir/'source_references.csv'):
        references[identity(r['song'],r['artist'])].append(r)
    alias_path = root/'config/kaggle_additional_aliases.csv'
    alias_rows = list(read_csv(alias_path))
    aliases = {identity(r['song'],r['billboard_artist']) for r in alias_rows}
    reviews = {}
    for r in read_csv(reviews_path):
        if r['song_id'] in reviews:
            raise ValueError('Duplicate recording review')
        reviews[r['song_id']] = r
    if set(reviews)-{r['song_id'] for r in songs}:
        raise ValueError('Unknown song in recording reviews')
    mapping = [select_mapping(s, references[identity(s['song'],s['artist'])],
        identity(s['song'],s['artist']) in aliases, reviews.get(s['song_id'])) for s in songs]
    mapping.sort(key=lambda r:r['song_id'])
    replayed = replay_references(root, mapping, manifest)
    economic = {r.predictor_month:r for r in load_economic(root/'data/processed/economic_monthly.csv')}
    contributions, months = prepare_months(read_csv(root/'billboard_hot100_all.csv'), mapping, economic, policy)
    plan = comparison_plan(months)
    per_song = defaultdict(Counter)
    for r in contributions:
        per_song[r['song_id']].update({k:r[k] for k in ['total_weight', 'provisional_weight', 'nlp_ready_weight']})
    priority = []
    for r in mapping:
        counts = per_song[r['song_id']]
        priority.append(dict(song_id=r['song_id'], song=r['song'], artist=r['artist'],
            missing_nlp_weight=counts['total_weight']-counts['nlp_ready_weight'],
            missing_provisional_weight=counts['total_weight']-counts['provisional_weight'],
            exclusions=r['exclusion_reasons'], readiness_blockers=r['readiness_blockers'], quality_flags=r['quality_flags']))
    priority.sort(key=lambda r:(-r['missing_provisional_weight'], -r['missing_nlp_weight'], r['song_id']))
    coverage = {}
    for key in ['total_weight','known_weight','provisional_weight','nlp_ready_weight']:
        coverage[key] = sum(m[key] for m in months)
    yearly = []
    for year in range(1973,2024):
        subset = [m for m in months if m['predictor_month'].startswith(str(year))]
        weights = {k:sum(m[k] for m in subset) for k in coverage}
        yearly.append(dict(year=year, **weights,
            provisional_weight_coverage=weights['provisional_weight']/weights['total_weight'] if weights['total_weight'] else 0,
            nlp_ready_weight_coverage=weights['nlp_ready_weight']/weights['total_weight'] if weights['total_weight'] else 0))
    summary = dict(identity_count=len(mapping), selected_provisional_sources=sum(r['mapping_status']=='provisional_selected' for r in mapping),
        excluded_identities=sum(r['mapping_status']=='excluded' for r in mapping),
        recording_verified=sum(r['recording_verified'] for r in mapping), nlp_ready_identities=sum(r['nlp_ready'] for r in mapping),
        **coverage, provisional_weight_coverage=coverage['provisional_weight']/coverage['total_weight'],
        nlp_ready_weight_coverage=coverage['nlp_ready_weight']/coverage['total_weight'],
        source_references_replayed=replayed, explicitly_excluded_alias_mappings=len(alias_rows),
        monthly_rows=len(months), preparation_eligible_months=sum(m['preparation_eligible'] for m in months),
        common_preparation_forecasts=sum(p['common_preparation_eligible'] for p in plan), runnable_forecasts=0,
        exclusion_counts=dict(Counter(reason for r in mapping for reason in r['exclusion_reasons'].split(';') if reason)),
        year_end_selected_sources=sum(r['source'] in ['kaggle','lydia','tmelo'] for r in mapping),
        warnings=['Selection is retrospective, not proof of historical lyric availability.',
                  'No emotion scores generated; current mapping is not an approved Model B input.'])
    inputs = [audit_dir/'inventory_audited.csv', audit_dir/'source_references.csv', audit_dir/'manifest.json',
        config_path,reviews_path,alias_path,root/'config/baseline.json',root/'data/processed/economic_monthly.csv',
        root/'billboard_hot100_all.csv']
    provenance = dict(inputs={str(p.relative_to(root)):file_hash(p) for p in inputs},
        code_sha256=file_hash(Path(__file__)), raw_source_hashes=manifest['inputs'],
        policy=policy, model_a_timing=dict(horizon_months=6,min_training_months=240,training_target_lag_months=1,cutoff='15th of t+1'),
        comparison='Rerun both Probit models on identical ready feature/training/evaluation dates; compare paired out-of-sample Brier Scores. No scores calculated here.')
    output.mkdir(parents=True)
    write_csv(output/'song_source_mapping.csv',mapping)
    write_csv(output/'monthly_contributions.csv',contributions)
    write_csv(output/'monthly_preparation.csv',months)
    write_csv(output/'comparison_calendar.csv',plan)
    write_csv(output/'review_priority.csv',priority)
    write_csv(output/'yearly_coverage.csv',yearly)
    write_csv(output/'alias_dispositions.csv',[dict(**r, disposition='excluded', reason='no_reproducible_source_identity') for r in alias_rows],
        list(alias_rows[0])+['disposition','reason'] if alias_rows else ['song','billboard_artist','disposition','reason'])
    for name, value in [('summary.json',summary),('manifest.json',provenance)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--audit',type=Path,default=Path('reports/lyrics_validation/run_002'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.root.resolve(),args.audit.resolve(),args.output.resolve()),indent=2))


if __name__=='__main__':
    main()

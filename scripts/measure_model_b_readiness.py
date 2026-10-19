"""Measure monthly lyric coverage and blocking axes without touching Model B.

Deliberately blind to forecast performance: this script never imports the Model B
estimator, never scores lyrics and never reads predictions, so an eligibility
policy cannot be chosen by peeking at Brier Scores. It only counts Billboard
popularity weight (101 - chart_rank) carried by candidate subsets.

Candidate subsets are measurement scenarios, NOT approvals. Nothing here grants
source use, verifies a recording, verifies a language or verifies a release date.
No lyric text is read or written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path

from music_recession.economic import add_months
from music_recession.lyrics_audit import file_hash, identity, read_csv, write_csv

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
MIN_TRAINING_MONTHS = 240
HORIZON_MONTHS = 6
TRAINING_TARGET_LAG_MONTHS = 1
# Flags that a human reviewer could plausibly resolve for a specific recording.
# Language, instrumental and metadata/slot failures are deliberately NOT here:
# they are different evidence axes, not credit-string or version annotations.
REVIEWABLE_FLAGS = {'upstream_featured_credit_removed_review',
                    'version_specific_recording_review'}
SCENARIOS = ['unflagged', 'unflagged_plus_reviewable']


def truthy(value):
    if value in (True, 'True'):
        return True
    if value in (False, 'False', ''):
        return False
    raise ValueError('Expected a boolean-valued column, got ' + repr(value))


def reasons(value):
    return set(filter(None, value.split(';')))


def cutoff_for(month):
    """Information-availability cutoff: day 15 of t+1, matching the Model A contract."""
    return add_months(month, 1).replace(day=15)


def scenario_membership(evidence):
    """Which measurement scenarios each selected candidate belongs to."""
    membership = {}
    for e in evidence:
        flags = reasons(e['technical_review_reasons'])
        unflagged = truthy(e['automated_screen_pass'])
        if unflagged != (not flags):
            raise ValueError('Screen-pass flag disagrees with review reasons: ' + e['song_id'])
        reviewable = bool(flags) and flags <= REVIEWABLE_FLAGS
        membership[e['song_id']] = dict(unflagged=unflagged,
                                        unflagged_plus_reviewable=unflagged or reviewable,
                                        reviewable_only=reviewable)
    return membership


def monthly_coverage(root, evidence, membership):
    """Weighted coverage per predictor month, per scenario, plus per-identity weight."""
    by_key = {}
    for e in evidence:
        key = identity(e['song'], e['artist'])
        if key in by_key:
            raise ValueError('Duplicate chart identity in candidate evidence')
        by_key[key] = e
    months = defaultdict(Counter)
    observable_weight = Counter()
    for r in read_csv(root / 'billboard_hot100_all.csv'):
        if not '1973-01-01' <= r['date'] <= '2023-12-31':
            continue
        month = date.fromisoformat(r['date']).replace(day=1)
        weight = 101 - int(r['this_week'])
        counts = months[month]
        counts['chart_entries'] += 1
        counts['all_chart_weight'] += weight
        candidate = by_key.get(identity(r['song'], r['artist']))
        if candidate is None:
            continue
        counts['selected_candidate_weight'] += weight
        # A source exemplar from a later chart week cannot backfill an earlier month.
        if candidate['selected_source_chart_date'] > cutoff_for(month).isoformat():
            continue
        counts['observable_selected_weight'] += weight
        observable_weight[candidate['song_id']] += weight
        for scenario in SCENARIOS:
            if membership[candidate['song_id']][scenario]:
                counts[scenario + '_weight'] += weight
    if not months:
        raise ValueError('No study-period Billboard observations')
    return months, observable_weight


def merge_preparation(months, preparation):
    """Attach the existing month-level chart-completeness and label gates."""
    prepared = {date.fromisoformat(r['predictor_month']): r for r in preparation}
    rows = []
    for month in sorted(months):
        counts = months[month]
        prep = prepared.get(month)
        if prep is None:
            raise ValueError('Preparation month missing: ' + month.isoformat())
        total = counts['all_chart_weight']
        if int(prep['total_weight']) != total:
            raise ValueError('Chart weight disagrees with preparation: ' + month.isoformat())
        row = dict(predictor_month=month.isoformat(),
                   target_month=add_months(month, HORIZON_MONTHS).isoformat(),
                   forecast_cutoff=cutoff_for(month).isoformat(),
                   chart_entries=counts['chart_entries'], all_chart_weight=total,
                   selected_candidate_weight=counts['selected_candidate_weight'],
                   observable_selected_weight=counts['observable_selected_weight'],
                   chart_dates=int(prep['chart_dates']),
                   missing_rank_slots=int(prep['missing_rank_slots']),
                   incomplete_chart_month=truthy(prep['incomplete_chart_month']),
                   has_economic_label=prep['target_t'] != '')
        row['chart_month_usable'] = bool(not row['incomplete_chart_month']
                                         and row['missing_rank_slots'] == 0
                                         and row['has_economic_label'])
        for scenario in SCENARIOS:
            weight = counts[scenario + '_weight']
            row[scenario + '_weight'] = weight
            row[scenario + '_weight_coverage'] = weight / total
        rows.append(row)
    return rows


def longest_run(selected):
    """Longest run of consecutive calendar months in a sorted month list."""
    best = run = 0
    previous = None
    for month in selected:
        run = run + 1 if previous is not None and month == add_months(previous, 1) else 1
        best = max(best, run)
        previous = month
    return best


def attainable_forecasts(eligible):
    """Apply the repository's actual rule: >=240 eligible rows with target <= t-1.

    select_training() counts eligible observations; it does not require them to be
    contiguous. Contiguity is reported separately so the distinction stays visible.
    """
    eligible = sorted(eligible)
    targets = [add_months(month, HORIZON_MONTHS) for month in eligible]
    forecasts = []
    for month in eligible:
        latest = add_months(month, -TRAINING_TARGET_LAG_MONTHS)
        if sum(target <= latest for target in targets) >= MIN_TRAINING_MONTHS:
            forecasts.append(month)
    return forecasts


def threshold_table(rows):
    table = []
    for scenario in SCENARIOS:
        for threshold in THRESHOLDS:
            qualifying = [date.fromisoformat(r['predictor_month']) for r in rows
                          if r[scenario + '_weight_coverage'] >= threshold]
            usable = [date.fromisoformat(r['predictor_month']) for r in rows
                      if r[scenario + '_weight_coverage'] >= threshold and r['chart_month_usable']]
            forecasts = attainable_forecasts(usable)
            run = longest_run(usable)
            table.append(dict(scenario=scenario, coverage_threshold=threshold,
                qualifying_months=len(qualifying),
                qualifying_first_month=qualifying[0].isoformat() if qualifying else '',
                qualifying_last_month=qualifying[-1].isoformat() if qualifying else '',
                usable_months=len(usable),
                usable_first_month=usable[0].isoformat() if usable else '',
                usable_last_month=usable[-1].isoformat() if usable else '',
                longest_contiguous_usable_run=run,
                contiguous_240_plus_forecast_attainable=bool(run > MIN_TRAINING_MONTHS),
                cumulative_240_training_attainable=bool(len(usable) >= MIN_TRAINING_MONTHS),
                attainable_forecast_months=len(forecasts),
                first_attainable_forecast_month=forecasts[0].isoformat() if forecasts else '',
                last_attainable_forecast_month=forecasts[-1].isoformat() if forecasts else '',
                first_attainable_target_month=add_months(forecasts[0], HORIZON_MONTHS).isoformat() if forecasts else ''))
    return table


def blocking_axes(evidence, mapping, observable_weight, total_weight, selected_weight):
    """Keep rights separate from recording, language and date evidence.

    Axis (a) is a permission question about a source. Axes (b)-(d) are factual
    evidence questions about a record. A record can be strong on (b)-(d) and
    still be blocked on (a); that is why these are reported separately.
    """
    def subset(predicate, label, axis, standard, note):
        chosen = [e for e in evidence if predicate(e)]
        weight = sum(observable_weight[e['song_id']] for e in chosen)
        return dict(axis=axis, standard=standard, blocker=label,
                    blocked_candidates=len(chosen), blocked_observable_weight=weight,
                    share_of_all_chart_weight=weight / total_weight,
                    share_of_selected_observable_weight=weight / selected_weight, note=note)

    def not_english(e):
        return e['language_status'] != 'english_automated_consistent'

    def recording_doubt(e):
        return (not truthy(e['metadata_match']) or not truthy(e['chart_slot_match'])
                or truthy(e['upstream_credit_query_reduced']) or bool(e['version_terms'])
                or e['instrumental_status'] != 'no_instrumental_marker_detected')

    rows = [
        subset(lambda e: not truthy(e['permission_verified']), 'permission_not_documented',
               'a_rights_permission', 'current_policy',
               'No source-use decision or lyric-publisher permission exists for any record'),
        subset(lambda e: True, 'source_use_decision_missing',
               'a_rights_permission', 'automated_evidence',
               'CC BY 4.0 covers only rights the depositor controls; nothing to screen automatically'),
        subset(lambda e: not truthy(e['recording_verified']), 'recording_not_independently_verified',
               'b_recording_identity', 'current_policy',
               'Policy requires a reviewed recording ID; the audit supplies none'),
        subset(recording_doubt, 'automated_recording_identity_doubt',
               'b_recording_identity', 'automated_evidence',
               'Credit-stripped query, version qualifier, instrumental marker or metadata/slot mismatch'),
        subset(lambda e: truthy(e['upstream_credit_query_reduced']), 'upstream_featured_credit_removed',
               'b_recording_identity', 'automated_evidence_component',
               'lyrics.ovh query dropped the featured-artist credit'),
        subset(lambda e: bool(e['version_terms']), 'version_specific_recording',
               'b_recording_identity', 'automated_evidence_component',
               'Title carries a version or numbered-part qualifier'),
        subset(lambda e: e['instrumental_status'] != 'no_instrumental_marker_detected',
               'instrumental_or_low_diversity', 'b_recording_identity', 'automated_evidence_component',
               'Instrumental marker with other text, or very low lexical diversity'),
        subset(lambda e: not truthy(e['language_verified']), 'language_not_reviewed',
               'c_language', 'current_policy',
               'Policy requires reviewed English; only an automated screen exists'),
        subset(not_english, 'automated_screen_not_consistently_english',
               'c_language', 'automated_evidence',
               'langid screen: confident non-English, mixed block, uncertain or too short'),
        subset(lambda e: not truthy(e['release_verified']), 'release_date_not_verified',
               'd_release_and_chart_date', 'current_policy',
               'Policy requires an independently verified release date; none supplied'),
        subset(lambda e: not truthy(e['observed_by_source_chart']), 'chart_observability_failure',
               'd_release_and_chart_date', 'automated_evidence',
               'First chart observation postdates the selected source chart week'),
    ]
    # Mutually exclusive automated-evidence combinations across axes (b)-(d).
    # Axis (a) is omitted here because it blocks every record identically.
    combination = Counter()
    combination_weight = Counter()
    for e in evidence:
        axes = []
        if recording_doubt(e):
            axes.append('b_recording_identity')
        if not_english(e):
            axes.append('c_language')
        if not truthy(e['observed_by_source_chart']):
            axes.append('d_release_and_chart_date')
        label = '+'.join(axes) if axes else 'none_of_b_c_d'
        combination[label] += 1
        combination_weight[label] += observable_weight[e['song_id']]
    for label, count in sorted(combination.items()):
        weight = combination_weight[label]
        rows.append(dict(axis='b_c_d_combination', standard='automated_evidence_exclusive',
            blocker=label, blocked_candidates=count, blocked_observable_weight=weight,
            share_of_all_chart_weight=weight / total_weight,
            share_of_selected_observable_weight=weight / selected_weight,
            note='Mutually exclusive automated-evidence combination; axis (a) blocks all of these as well'))
    excluded = [r for r in mapping if r['mapping_status'] != 'provisional_selected']
    exclusion_counts = Counter(reason for r in excluded
                               for reason in reasons(r['exclusion_reasons']))
    for reason, count in sorted(exclusion_counts.items()):
        rows.append(dict(axis='pre_selection_exclusion', standard='existing_source_mapping',
            blocker=reason, blocked_candidates=count, blocked_observable_weight='',
            share_of_all_chart_weight='', share_of_selected_observable_weight='',
            note='Identity never reached OSF candidate selection; reasons overlap, so counts are not additive'))
    return rows, len(excluded)


def run(root, evidence_dir, preparation_dir, output):
    root, evidence_dir, preparation_dir, output = (p.resolve() for p in
        (root, evidence_dir, preparation_dir, output))
    if output.exists():
        raise ValueError('Choose a fresh output directory; existing results are preserved')
    for path in (evidence_dir / 'candidate_evidence.csv', preparation_dir / 'monthly_preparation.csv',
                 preparation_dir / 'song_source_mapping.csv', root / 'billboard_hot100_all.csv'):
        if not path.exists():
            raise ValueError('Missing required input: ' + str(path))
    evidence = list(read_csv(evidence_dir / 'candidate_evidence.csv'))
    preparation = list(read_csv(preparation_dir / 'monthly_preparation.csv'))
    mapping = list(read_csv(preparation_dir / 'song_source_mapping.csv'))
    membership = scenario_membership(evidence)
    months, observable_weight = monthly_coverage(root, evidence, membership)
    rows = merge_preparation(months, preparation)
    table = threshold_table(rows)
    totals = {k: sum(r[k] for r in rows) for k in
              ['all_chart_weight', 'selected_candidate_weight', 'observable_selected_weight',
               'unflagged_weight', 'unflagged_plus_reviewable_weight']}
    axes, excluded_identities = blocking_axes(evidence, mapping, observable_weight,
        totals['all_chart_weight'], totals['observable_selected_weight'])
    summary = dict(
        scope='Predictor months 1973-01 through 2023-12; measurement only, no Model B fit or score',
        study_identities=len(mapping), selected_candidates=len(evidence),
        pre_selection_excluded_identities=excluded_identities,
        unflagged_candidates=sum(m['unflagged'] for m in membership.values()),
        reviewable_only_candidates=sum(m['reviewable_only'] for m in membership.values()),
        unflagged_plus_reviewable_candidates=sum(m['unflagged_plus_reviewable'] for m in membership.values()),
        monthly_rows=len(rows),
        chart_usable_months=sum(r['chart_month_usable'] for r in rows),
        **totals,
        **{scenario + '_weight_share': totals[scenario + '_weight'] / totals['all_chart_weight']
           for scenario in SCENARIOS},
        minimum_training_months=MIN_TRAINING_MONTHS, horizon_months=HORIZON_MONTHS,
        training_rule='select_training(): eligible rows with target_month <= t-1; count only, contiguity not required',
        thresholds=THRESHOLDS, reviewable_flags=sorted(REVIEWABLE_FLAGS),
        eligible_records_under_current_policy=0,
        warnings=[
            'Scenario membership is a measurement subset, never an approval.',
            'Automated screens are not verified recording identity, language or release dates.',
            'Source-use rights for the underlying lyrics remain unresolved for every record.',
            'No Model B inference, fit or score was produced by this measurement.'])
    output.mkdir(parents=True)
    write_csv(output / 'monthly_coverage_scenarios.csv', rows)
    write_csv(output / 'coverage_thresholds.csv', table)
    write_csv(output / 'blocking_axes.csv', axes)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = dict(inputs={str(p.relative_to(root)): file_hash(p) for p in [
        evidence_dir / 'candidate_evidence.csv', preparation_dir / 'monthly_preparation.csv',
        preparation_dir / 'song_source_mapping.csv', root / 'billboard_hot100_all.csv',
        root / 'config/music_preparation.json', root / 'config/baseline.json']},
        code_sha256=file_hash(Path(__file__)),
        evidence_run=str(evidence_dir.relative_to(root)),
        preparation_run=str(preparation_dir.relative_to(root)),
        policy='Coverage denominator is all observed rank weight, including unmatched and excluded songs',
        output_hashes={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--evidence', type=Path,
                        default=Path('reports/osf_source_review/run_004'))
    parser.add_argument('--preparation', type=Path,
                        default=Path('reports/model_comparison/primary_2026_10_18_final/preparation'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p
    run(root, resolve(args.evidence), resolve(args.preparation), resolve(args.output))


if __name__ == '__main__':
    main()

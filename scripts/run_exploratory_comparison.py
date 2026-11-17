"""Exploratory course-policy Model A vs Model B comparison.

EXPLORATORY ANALYSIS. The records admitted here pass only the project's existing
AUTOMATED risk and English-language screens. They are NOT independently verified
recordings, NOT human-reviewed for language, carry NO verified release dates, and
are NOT rights-cleared. Source use rests on REPORTED INSTRUCTOR GUIDANCE for a
course project, which is documented as such and is not a copyright licence.

The strict primary policy is unchanged and its outputs are untouched: this script
never writes to the strict preparation config, the recording-reviews file, or any
existing strict report directory. It reuses the audited preparation, aggregation,
evaluation and plotting code unmodified.

Both models are refit on identical eligible training rows and scored on identical
forecast dates. No lyric text is written to any output.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from music_recession import music_preparation
from music_recession.baseline import BaselineConfig, load_economic
from music_recession.emotions import (GoEmotions, score_eligible, software_versions,
                                      validate_config)
from music_recession.lyrics_audit import file_hash, identity, read_csv, write_csv
from music_recession.music_model import aggregate, evaluate, plots
from music_recession.music_preparation import (EXPLORATORY_BASIS, prepare_months,
                                               replay_references, select_mapping,
                                               validate_policy)

BANNER = ('EXPLORATORY COURSE ANALYSIS: automated screens only; recordings not '
          'independently verified; language not human reviewed; no verified release '
          'dates; source use rests on reported instructor guidance, not a licence.')


def build_exploratory_mapping(root, audit_dir, screen_path, policy):
    manifest = json.loads((audit_dir / 'manifest.json').read_text())
    for path, value in manifest['inputs'].items():
        if file_hash(root / path) != value['sha256']:
            raise ValueError('Audit input changed: ' + path)
    songs = list(read_csv(audit_dir / 'inventory_audited.csv'))
    references = defaultdict(list)
    for row in read_csv(audit_dir / 'source_references.csv'):
        references[identity(row['song'], row['artist'])].append(row)
    aliases = {identity(r['song'], r['billboard_artist'])
               for r in read_csv(root / 'config/kaggle_additional_aliases.csv')}
    screen = {}
    for row in read_csv(screen_path):
        if row['song_id'] in screen:
            raise ValueError('Duplicate screen row: ' + row['song_id'])
        screen[row['song_id']] = row
    unknown = set(screen) - {s['song_id'] for s in songs}
    if unknown:
        raise ValueError('Screen references unknown songs: %d' % len(unknown))
    mapping = [select_mapping(s, references[identity(s['song'], s['artist'])],
                              identity(s['song'], s['artist']) in aliases,
                              None, screen.get(s['song_id'], {}))
               for s in songs]
    mapping.sort(key=lambda r: r['song_id'])
    replayed = replay_references(root, mapping, manifest)
    return mapping, manifest, replayed


def run(root, output, cache, audit_dir, screen_path, threshold, limit=None):
    root, output, cache = root.resolve(), output.resolve(), cache.resolve()
    audit_dir, screen_path = audit_dir.resolve(), screen_path.resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory; existing results are preserved')
    policy = json.loads((root / 'config/music_preparation_exploratory.json').read_text())
    if threshold is not None:
        policy['minimum_ready_weight_coverage'] = threshold
    baseline = BaselineConfig.load(root / 'config/baseline.json')
    validate_policy(policy, baseline)
    model_config = json.loads((root / 'config/music_model.json').read_text())
    validate_config(model_config)

    print(BANNER)
    print('coverage threshold        :', policy['minimum_ready_weight_coverage'])
    mapping, manifest, replayed = build_exploratory_mapping(root, audit_dir, screen_path, policy)
    eligible = [r for r in mapping if r['nlp_ready']]
    print('identities                :', len(mapping))
    print('exploratory eligible songs:', len(eligible))
    print('source references replayed:', replayed)
    if limit:
        # Smoke mode: score a deterministic subset to measure throughput only.
        keep = {r['song_id'] for r in sorted(eligible, key=lambda r: r['song_id'])[:limit]}
        for row in mapping:
            if row['nlp_ready'] and row['song_id'] not in keep:
                row['nlp_ready'] = False
                row['readiness_blockers'] = 'excluded_by_smoke_limit'
        eligible = [r for r in mapping if r['nlp_ready']]
        print('SMOKE MODE eligible songs :', len(eligible))

    economic_rows = load_economic(root / 'data/processed/economic_monthly.csv')
    economic = {r.predictor_month: r for r in economic_rows}
    contributions, months = prepare_months(read_csv(root / 'billboard_hot100_all.csv'),
                                           mapping, economic, policy)
    ready_months = sum(m['preparation_eligible'] for m in months)
    print('preparation eligible months:', ready_months, 'of', len(months))

    started = time.monotonic()
    scores = score_eligible(root, mapping, model_config, cache)
    elapsed = time.monotonic() - started
    hits = sum(1 for s in scores.values() if s['cache_hit'])
    print('scored songs              : %d (cache hits %d, new %d) in %.1fs'
          % (len(scores), hits, len(scores) - hits, elapsed))

    monthly = aggregate(months, contributions, scores)
    records, metrics = evaluate(economic_rows, monthly, baseline)

    metrics.update(
        analysis_policy='exploratory_course',
        eligibility_basis=EXPLORATORY_BASIS,
        coverage_threshold=policy['minimum_ready_weight_coverage'],
        exploratory_eligible_songs=len(eligible),
        scored_song_count=len(scores),
        emotion_cache_hits=hits,
        inference_executed=bool(len(scores) - hits),
        preparation_eligible_months=ready_months,
        source_use_basis=policy['source_use_basis'],
        evidence_status=dict(
            recording_identity='automated metadata and chart-slot match only; '
                               'NOT independently verified',
            language='automated langid screen; NOT human reviewed',
            release_date='no verified release date; chart observability only',
            rights='reported instructor guidance for a course project; '
                   'NOT a licence or rights clearance'),
        strict_policy_note='The strict primary policy and its results are unchanged. '
                           'These numbers are exploratory and must not be presented as '
                           'verified-evidence findings.',
        banner=BANNER)

    output.mkdir(parents=True)
    write_csv(output / 'monthly_features_and_coverage.csv', monthly)
    write_csv(output / 'predictions.csv', records)
    write_csv(output / 'song_source_mapping.csv', mapping)
    write_csv(output / 'monthly_contributions.csv', contributions)
    calibration = []
    for model in ('a', 'b'):
        if metrics['model_' + model]:
            calibration.extend(dict(model=model.upper(), **row)
                               for row in metrics['model_' + model]['calibration'])
    write_csv(output / 'calibration.csv', calibration,
              ['model', 'bin_lower', 'bin_upper', 'upper_inclusive', 'count',
               'mean_predicted_probability', 'observed_recession_frequency'])
    coverage_rows = [dict(predictor_month=m['predictor_month'],
                          total_weight=m['total_weight'], known_weight=m['known_weight'],
                          provisional_weight=m['provisional_weight'],
                          nlp_ready_weight=m['nlp_ready_weight'],
                          scored_weight=m['scored_weight'],
                          scored_weight_coverage=m['scored_weight_coverage'],
                          nlp_ready_weight_coverage=m['nlp_ready_weight_coverage'],
                          incomplete_chart_month=m['incomplete_chart_month'],
                          missing_rank_slots=m['missing_rank_slots'],
                          preparation_eligible=m['preparation_eligible'],
                          feature_eligible=m['feature_eligible'],
                          emotion_status=m['emotion_status']) for m in monthly]
    write_csv(output / 'monthly_coverage.csv', coverage_rows)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    plots(monthly, records, output)
    provenance = dict(
        analysis_policy='exploratory_course', banner=BANNER, policy=policy,
        preparation_config_sha256=file_hash(root / 'config/music_preparation_exploratory.json'),
        model_config_sha256=file_hash(root / 'config/music_model.json'),
        baseline_config_sha256=file_hash(root / 'config/baseline.json'),
        screen_table=str(screen_path.relative_to(root)),
        screen_table_sha256=file_hash(screen_path),
        audit_dir=str(audit_dir.relative_to(root)), raw_source_hashes=manifest['inputs'],
        software=dict(software_versions(), python=platform.python_version()),
        code_sha256={name: file_hash(Path(music_preparation.__file__).with_name(name))
                     for name in ['music_model.py', 'emotions.py', 'music_preparation.py',
                                  'baseline.py', 'backtest.py']},
        driver_sha256=file_hash(Path(__file__)),
        outputs_sha256={p.name: file_hash(p) for p in output.iterdir() if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('data/processed/emotion_cache_exploratory'))
    parser.add_argument('--audit-dir', type=Path,
                        default=Path('reports/lyrics_validation/run_002'))
    parser.add_argument('--screen', type=Path,
                        default=Path('reports/osf_source_review/run_004/candidate_evidence.csv'))
    parser.add_argument('--coverage-threshold', type=float, default=None,
                        help='Override the predeclared 0.70 primary threshold for '
                             'the predeclared 0.75 / 0.80 sensitivity analyses')
    parser.add_argument('--limit', type=int, default=None,
                        help='Smoke mode: score only the first N eligible songs')
    args = parser.parse_args(argv)
    metrics = run(args.root, args.output_dir, args.cache_dir, args.audit_dir,
                  args.screen, args.coverage_threshold, args.limit)
    print(json.dumps({k: metrics[k] for k in
                      ['status', 'coverage_threshold', 'scored_song_count',
                       'eligible_feature_months', 'evaluated_months', 'recession_months',
                       'forecast_start', 'forecast_end', 'model_a', 'model_b',
                       'brier_difference_b_minus_a']}, indent=2))
    return 0 if metrics['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())

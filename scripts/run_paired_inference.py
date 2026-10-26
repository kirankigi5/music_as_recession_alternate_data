"""Block-bootstrap intervals, Diebold-Mariano tests and PR-AUC for the committed runs.

Reads docs/results/<run>/predictions.csv and data/processed/economic_monthly.csv.
No model fitting, no emotion inference, no network; finishes in a few seconds.
Fixed seed, so reruns reproduce the saved JSON exactly.

Usage:
    python scripts/run_paired_inference.py
    python scripts/run_paired_inference.py \\
        --run-dir docs/results/exploratory_primary_cov70 \\
        --run-dir reports/model_comparison/<new_variant> --out <new>.json

Any directory holding a predictions.csv written by music_model.evaluate works.
The first --run-dir is treated as the reference sample for the overlap count.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from music_recession.inference import paired_summary, recession_episodes

RUNS = [('70% (primary)', Path('docs/results/exploratory_primary_cov70')),
        ('75% sensitivity', Path('docs/results/exploratory_sensitivity_cov75')),
        ('80% sensitivity', Path('docs/results/exploratory_sensitivity_cov80'))]
DEFAULT_OUT = Path('docs/results/paired_inference.json')


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def build(root, runs=RUNS):
    episodes = recession_episodes(read_rows(root / 'data/processed/economic_monthly.csv'))
    results, months = [], []
    for label, directory in runs:
        rows = read_rows(root / directory / 'predictions.csv')
        months.append({r['predictor_month'] for r in rows if r['status'] == 'ok'})
        results.append(dict(label=label, run=directory.as_posix(), **paired_summary(rows, episodes)))
    for result, run_months in zip(results[1:], months[1:]):
        # Stricter coverage keeps a subset of the same forecast months, so the
        # thresholds are not independent replications of each other.
        result['forecast_months_also_in_primary'] = len(run_months & months[0])
    return dict(source='<run>/predictions.csv', runs=results)


def number(value, spec):
    # AUCs, skill and test statistics are null when a sample has one class only.
    return 'n/a' if value is None else format(value, spec)


def table(result):
    lines = []
    for run in result['runs']:
        diff = run['loss_differential']
        lines.append('%s: n=%d, %d recession months from %d episode(s), mean B-A %+.4f' % (
            run['label'], run['forecasts'], run['recession_target_months'],
            len(run['recession_episodes_reached']), diff['mean']))
        for name in ('model_a', 'model_b'):
            m = run[name]
            lines.append('  %s  Brier %.4f  skill vs prevalence %s  ROC-AUC %s  PR-AUC %s' % (
                name, m['brier_score'], number(m['brier_skill_vs_training_prevalence'], '+.3f'),
                number(m['roc_auc'], '.3f'), number(m['pr_auc'], '.3f')))
        lines.append('  PR-AUC no-skill level %.3f' % run['pr_auc_no_skill_level'])
        iid = diff['iid_bootstrap']
        lines.append('  iid rows        95%% CI [%+.4f, %+.4f]' % (iid['ci_lower'], iid['ci_upper']))
        for block in diff['block_bootstrap']:
            lines.append('  blocks %2d mo    95%% CI [%+.4f, %+.4f]  share<=0 %.3f' % (
                block['block_length_months'], block['ci_lower'], block['ci_upper'],
                block['share_at_or_below_zero']))
        for dm in diff['diebold_mariano']:
            lines.append('  DM lag %2d       t=%s  p=%s  effective n=%s' % (
                dm['max_lag'], number(dm['statistic'], '.2f'),
                number(dm['p_value_two_sided'], '.3f'), number(dm['effective_observations'], '.0f')))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run-dir', type=Path, action='append',
                        help='result directory with predictions.csv; repeat for several')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.run_dir and args.out is None:
        parser.error('--out is required with --run-dir, so the committed JSON is not overwritten')
    runs = [(d.name, d) for d in args.run_dir] if args.run_dir else RUNS
    args.out = args.out or DEFAULT_OUT
    result = build(args.root.resolve(), runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(table(result))
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

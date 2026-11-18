"""SUPPLEMENTARY comparison: spread-only Probit vs spread + month-to-month spread change.

Separate from the original Model A/B experiment; does not read or modify its
predictions, configuration or manifests. No NLP inference, no network.
Both models are refit at every origin on the identical common sample (months
whose previous calendar month is observed), with config/baseline.json unchanged.

Usage:
    python scripts/run_spread_change_comparison.py                       # writes the committed directory
    python scripts/run_spread_change_comparison.py --out /tmp/spread-FRESH  # fresh check
The output directory must not already exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from music_recession.backtest import brier_score, calibration_table, log_loss
from music_recession.baseline import BaselineConfig, load_economic
from music_recession.inference import paired_summary, recession_episodes
from music_recession.spread_change import (
    END, LABEL, PREDICTION_COLUMNS, START, evaluate, spread_changes,
)

ECONOMIC = Path('data/processed/economic_monthly.csv')
CONFIG = Path('config/baseline.json')
SAVED_BASELINE = Path('reports/baseline_backtest/forecasts.csv')
DEFAULT_OUT = Path('docs/results/supplementary_spread_change')
CODE = (Path('src/music_recession/spread_change.py'), Path('src/music_recession/baseline.py'),
        Path('src/music_recession/backtest.py'), Path('src/music_recession/inference.py'),
        Path('scripts/run_spread_change_comparison.py'))


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    config = BaselineConfig.load(root / CONFIG)
    rows = load_economic(root / ECONOMIC)
    changes = spread_changes(root / ECONOMIC)
    records = evaluate(rows, changes, config, START, END)
    paired = [r for r in records if r['status'] == 'ok']
    if not paired:
        raise ValueError('No paired forecasts')
    # paired_summary expects the Model A/B column names; A = spread only, B = spread + change.
    as_ab = [dict(r, model_a_probability=r['baseline_probability'],
                  model_b_probability=r['spread_change_probability']) for r in paired]
    episodes = recession_episodes(read_rows(root / ECONOMIC))
    summary = paired_summary(as_ab, episodes)
    y = [int(r['actual_label']) for r in paired]
    models = {}
    for key, column in (('model_a', 'baseline_probability'), ('model_b', 'spread_change_probability')):
        p = [float(r[column]) for r in paired]
        models[key] = p
        summary[key].update(log_loss=log_loss(y, p)['value'], mean_probability=sum(p) / len(p),
                            calibration=calibration_table(y, p))
    summary['spread_only'] = summary.pop('model_a')
    summary['spread_plus_change'] = summary.pop('model_b')
    summary['loss_differential']['definition'] = (
        'Brier loss of spread+change minus spread-only per forecast month; positive = spread+change worse')
    # Context only: the original saved baseline (training from 1953-04) on the same months.
    saved = {r['predictor_month']: r for r in read_rows(root / SAVED_BASELINE)}
    months = [r['predictor_month'] for r in paired]
    saved_p = [float(saved[m]['predicted_probability']) for m in months]
    excluded = [dict(predictor_month=r['predictor_month'], status=r['status'], reason=r['error_message'])
                for r in records if r['status'] != 'ok']
    ab = summary['loss_differential']['mean']
    metrics = dict(
        label=LABEL,
        question='Does adding delta_spread_t = spread_t - spread_(t-1) to the spread-only Probit change '
                 'out-of-sample probability accuracy on identical forecast months?',
        definition='delta_spread_t uses the previous calendar month only; undefined if it is unobserved',
        design=dict(
            horizon='USREC at predictor month + 6 calendar months (single-month status)',
            window='expanding; refit from scratch at every origin; no tuning',
            training_rule='training targets <= t-1; >= %d eligible rows; start %s or first common month'
                          % (config.min_training_months, config.training_start),
            common_sample='rows where delta_spread_t is defined; both models use exactly these rows',
            forecast_cutoff='assumed 15th of t+1; retrospective pseudo-out-of-sample',
            scheduled_months='%s to %s' % (START.isoformat(), END.isoformat())),
        scheduled_forecasts=len(records), evaluated_forecasts=len(paired),
        evaluated_target_start=paired[0]['target_month'], evaluated_target_end=paired[-1]['target_month'],
        recession_prevalence=sum(y) / len(y), not_evaluated=excluded,
        result=dict(brier_spread_only=summary['spread_only']['brier_score'],
                    brier_spread_plus_change=summary['spread_plus_change']['brier_score'],
                    brier_difference_change_minus_spread_only=ab,
                    direction='spread+change lower Brier' if ab < 0 else
                              'spread+change higher Brier' if ab > 0 else 'identical'),
        original_saved_baseline_same_months=dict(
            note='Context only. Original reports/baseline_backtest forecasts restricted to these months; '
                 'its training windows include 1953-04, one row more than the common sample.',
            brier_score=brier_score(y, saved_p),
            max_abs_probability_difference_vs_recomputed=max(
                abs(a - b) for a, b in zip(saved_p, models['model_a']))),
        summary=summary,
        limitations=[
            'Final FRED values and retrospective NBER dating contain hindsight.',
            'Monthly errors are dependent and few recession episodes are represented; '
            'intervals are conditional sensitivities, not confirmatory tests.',
            'One pre-specified feature; no feature, date, threshold or setting search was performed.',
        ])
    return records, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists():
        parser.error('%s exists; choose a fresh output directory' % out)
    records, metrics = build(root)
    out.mkdir(parents=True)
    with open(out / 'predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    with open(out / 'calibration.csv', 'w', newline='', encoding='utf-8') as handle:
        columns = ['model', 'bin_lower', 'bin_upper', 'upper_inclusive', 'count',
                   'mean_predicted_probability', 'observed_recession_frequency']
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for model in ('spread_only', 'spread_plus_change'):
            writer.writerows(dict(model=model, **b) for b in metrics['summary'][model]['calibration'])
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    manifest = dict(
        label=LABEL,
        reproduce='python scripts/run_spread_change_comparison.py --out <fresh directory>',
        inputs_sha256={p.as_posix(): sha(root / p) for p in (ECONOMIC, CONFIG, SAVED_BASELINE)},
        code_sha256={p.as_posix(): sha(root / p) for p in CODE},
        software=dict(python=platform.python_version(), **{
            p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'statsmodels')}),
        outputs_sha256={name: sha(out / name) for name in ('predictions.csv', 'calibration.csv', 'metrics.json')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    s = metrics['summary']
    print(LABEL)
    print('forecasts %d (%s to %s), recession months %d, prevalence %.4f' % (
        s['forecasts'], s['first_predictor_month'], s['last_predictor_month'],
        s['recession_target_months'], metrics['recession_prevalence']))
    for name in ('spread_only', 'spread_plus_change'):
        m = s[name]
        print('  %-19s Brier %.6f  skill %+.4f  ROC-AUC %.4f  AP %.4f  log loss %.4f' % (
            name, m['brier_score'], m['brier_skill_vs_training_prevalence'], m['roc_auc'],
            m['average_precision'], m['log_loss']))
    print('  difference (change - spread only) %+.6f; %s' % (
        metrics['result']['brier_difference_change_minus_spread_only'], metrics['result']['direction']))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

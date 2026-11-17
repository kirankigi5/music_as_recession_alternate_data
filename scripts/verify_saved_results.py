"""Verify bundled results without lyrics, network, model fitting or inference.

Run from a checkout with the baseline numerical dependencies installed:
    python scripts/verify_saved_results.py
    python scripts/verify_saved_results.py --output /tmp/verification.json

This checks descriptive metrics and provenance only. Uncertainty and PR-AUC
remain in the separate follow-up PR #2.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from music_recession.backtest import brier_score, calibration_table, roc_auc
from music_recession.economic import add_months

RUNS = ('exploratory_primary_cov70', 'exploratory_sensitivity_cov75',
        'exploratory_sensitivity_cov80')
EMOTIONS = ('joy', 'sadness', 'fear', 'anger', 'nervousness', 'optimism')
HISTORICAL_NLP_HASH = '614f862be7c480bc40864f79c4d2af63eef5b2270f17fac222bee1c743f777d1'
CURRENT_NLP_HASH = 'e5600454033eb04373c73a2238de1625d12bc80f0e0b8a9912e0ccad16a32aeb'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    if expected is None or actual is None:
        require(actual is expected, context)
    else:
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), context)


def verify(root):
    result = dict(
        scope='Descriptive checks of saved results; no fitting, inference, uncertainty or PR-AUC.',
        runs={}, input_sha256={},
    )

    def track(path):
        result['input_sha256'][path.relative_to(root).as_posix()] = sha(path)

    economic_path = root / 'data/processed/economic_monthly.csv'
    track(economic_path)
    economic = {r['predictor_month']: r for r in rows(economic_path)}
    paired_by_run = {}
    for name in RUNS:
        directory = root / 'docs/results' / name
        for filename in ('metrics.json', 'predictions.csv', 'monthly_coverage.csv',
                         'monthly_features_and_coverage.csv', 'calibration.csv', 'manifest.json'):
            track(directory / filename)
        metrics = json.loads((directory / 'metrics.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        predictions = rows(directory / 'predictions.csv')
        paired = [r for r in predictions if r['status'] == 'ok']
        paired_by_run[name] = paired
        feature_rows = rows(directory / 'monthly_features_and_coverage.csv')
        features = {r['predictor_month']: r for r in feature_rows}
        require(len(features) == len(feature_rows), name + ': duplicate feature month')
        require(len({r['predictor_month'] for r in predictions}) == len(predictions),
                name + ': duplicate prediction month')
        require(len(predictions) == metrics['analysis_months'], name + ': analysis count')
        require(len(paired) == metrics['evaluated_months'], name + ': evaluated count')
        require(not any(r['status'] == 'failed' for r in predictions), name + ': failed fits')
        require(not metrics['fit_failures'], name + ': recorded fit failures')
        ready = sorted((r for r in feature_rows if r['feature_eligible'] == 'True'),
                       key=lambda r: r['predictor_month'])
        require(len(ready) == metrics['eligible_feature_months'], name + ': eligible count')
        for f in feature_rows:
            t = date.fromisoformat(f['predictor_month'])
            require(f['target_month'] == add_months(t, 6).isoformat(), name + ': target alignment')
            e = economic[f['predictor_month']]
            close(float(f['spread_t']), float(e['spread_t']), name + ': spread mismatch')
            require(int(f['target_t']) == int(e['target_t']), name + ': outcome mismatch')
        y = [int(r['actual_label']) for r in paired]
        close(sum(y), metrics['recession_months'], name + ': positive count')
        model_metrics = {}
        for model in ('a', 'b'):
            p = [float(r['model_' + model + '_probability']) for r in paired]
            bs, auc = brier_score(y, p), roc_auc(y, p)
            expected = metrics['model_' + model]
            close(bs, expected['brier_score'], name + ': Brier')
            close(auc, expected['roc_auc'], name + ': ROC-AUC')
            calibration = calibration_table(y, p)
            for observed, saved in zip(calibration, expected['calibration']):
                for key in ('count', 'mean_predicted_probability', 'observed_recession_frequency'):
                    close(observed[key], saved[key], name + ': calibration ' + key)
            csv_cal = [r for r in rows(directory / 'calibration.csv') if r['model'] == model.upper()]
            require(len(csv_cal) == len(calibration), name + ': calibration CSV count')
            for observed, saved in zip(calibration, csv_cal):
                for key in ('count', 'mean_predicted_probability', 'observed_recession_frequency'):
                    close(observed[key], float(saved[key]) if saved[key] else None,
                          name + ': calibration CSV ' + key)
            max_error = 0.0
            for r, probability in zip(paired, p):
                f = features[r['predictor_month']]
                coefficients = json.loads(r['model_' + model + '_fit_json'])['coefficients']
                linear = coefficients['intercept'] + sum(
                    v * float(f[k]) for k, v in coefficients.items() if k != 'intercept')
                reconstructed = .5 * (1 + math.erf(linear / math.sqrt(2)))
                max_error = max(max_error, abs(reconstructed - probability))
            require(max_error < 1e-12, name + ': probability reconstruction')
            model_metrics[model] = dict(brier_score=bs, roc_auc=auc,
                                       max_probability_reconstruction_error=max_error)
        for r in paired:
            t = date.fromisoformat(r['predictor_month'])
            f = features[r['predictor_month']]
            require(r['target_month'] == f['target_month'], name + ': paired target')
            require(int(r['actual_label']) == int(f['target_t']), name + ': paired label')
            require(r['forecast_cutoff'] == add_months(t, 1).replace(day=15).isoformat(),
                    name + ': forecast cutoff')
            training = [f for f in ready if f['target_month'] <= add_months(t, -1).isoformat()]
            require(len(training) == int(r['training_count']) and len(training) >= 240,
                    name + ': training count')
            require(sum(int(f['target_t']) for f in training) == int(r['training_recession_count']),
                    name + ': training positives')
            months_hash = hashlib.sha256('\n'.join(f['predictor_month'] for f in training).encode()).hexdigest()
            require(months_hash == r['training_months_sha256'], name + ': training dates')
            fit = json.loads(r['model_b_fit_json'])
            for key, saved in fit['training_mean'].items():
                close(sum(float(f[key]) for f in training) / len(training), saved,
                      name + ': training-only mean ' + key)
        close(model_metrics['b']['brier_score'] - model_metrics['a']['brier_score'],
              metrics['brier_difference_b_minus_a'], name + ': paired difference')
        checked_outputs = []
        for filename in ('metrics.json', 'predictions.csv', 'monthly_coverage.csv',
                         'monthly_features_and_coverage.csv', 'calibration.csv'):
            require(sha(directory / filename) == manifest['outputs_sha256'][filename],
                    name + ': historical output hash ' + filename)
            checked_outputs.append(filename)
        for filename, expected in manifest['code_sha256'].items():
            actual = sha(root / 'src/music_recession' / filename)
            if filename == 'emotions.py':
                require(expected == HISTORICAL_NLP_HASH and actual == CURRENT_NLP_HASH,
                        name + ': unexpected NLP provenance')
            else:
                require(actual == expected, name + ': changed historical model code ' + filename)
        for key, filename in (('preparation_config_sha256', 'music_preparation_exploratory.json'),
                              ('model_config_sha256', 'music_model.json'),
                              ('baseline_config_sha256', 'baseline.json')):
            track(root / 'config' / filename)
            require(sha(root / 'config' / filename) == manifest[key], name + ': config hash')
        coverage = rows(directory / 'monthly_coverage.csv')
        covered = [r for r in coverage if float(r['scored_weight_coverage']) >= metrics['coverage_threshold']]
        excluded = [r for r in covered if r['feature_eligible'] != 'True']
        require(all(int(r['missing_rank_slots']) > 0 for r in excluded), name + ': exclusion reason')
        episode_counts = dict(recession_2001=0, financial_crisis=0, covid=0)
        for r in paired:
            if int(r['actual_label']):
                year = int(r['target_month'][:4])
                key = ('recession_2001' if year == 2001 else 'financial_crisis'
                       if year in (2008, 2009) else 'covid' if year == 2020 else None)
                require(key is not None, name + ': unexpected recession episode')
                episode_counts[key] += 1
        result['runs'][name] = dict(
            forecasts=len(paired), recession_months=sum(y), models=model_metrics,
            training_rate_brier=brier_score(y, [int(r['training_recession_count']) /
                int(r['training_count']) for r in paired]), episodes=episode_counts,
            coverage_only_months=len(covered), feature_eligible_months=len(ready),
            missing_rank_excluded_months=[r['predictor_month'] for r in excluded],
            historical_output_hashes_checked=checked_outputs,
            nlp_provenance='Historical NLP hash differs; no full current-hash inference rerun claimed.',
        )
    primary = paired_by_run[RUNS[0]]
    dates = {r['predictor_month'] for r in primary}
    for previous, current in zip(RUNS, RUNS[1:]):
        require({r['predictor_month'] for r in paired_by_run[current]} <=
                {r['predictor_month'] for r in paired_by_run[previous]}, 'Threshold samples not nested')
    full_path = root / 'reports/baseline_backtest/forecasts.csv'
    track(full_path)
    full = {r['predictor_month']: r for r in rows(full_path) if r['status'] == 'ok'}
    for r in primary:
        other = full[r['predictor_month']]
        require(other['target_month'] == r['target_month'] and other['actual_label'] == r['actual_label'],
                'Full-history benchmark must use identical evaluation outcomes')
    full_score = brier_score([int(r['actual_label']) for r in primary],
                            [float(full[r['predictor_month']]['predicted_probability']) for r in primary])
    high = [r for r in primary if .3 <= float(r['model_b_probability']) < .8]
    positives = [int(r['training_recession_count']) for r in primary]
    without_covid = [r for r in primary if r['target_month'] not in ('2020-03-01', '2020-04-01')]
    delta = lambda subset: sum((float(r['model_b_probability']) - int(r['actual_label'])) ** 2 -
                              (float(r['model_a_probability']) - int(r['actual_label'])) ** 2 for r in subset)
    result['primary_checks'] = dict(
        full_history_a_same_dates=dict(forecasts=len(dates), brier_score=full_score,
            source='Existing baseline forecasts; different training history, identical evaluation dates.'),
        model_b_probability_03_to_08_months=len(high),
        model_b_probability_03_to_08_recession_months=sum(int(r['actual_label']) for r in high),
        positive_training_labels_range=[min(positives), max(positives)],
        parameters_including_intercept=8,
        events_per_parameter_range=[min(positives) / 8, max(positives) / 8],
        target_years_1999_2001_share_of_net_extra_loss=delta(
            [r for r in primary if r['target_month'][:4] in ('1999', '2001')]) / delta(primary),
        covid_exclusion=dict(status='Post-hoc descriptive sensitivity; primary retains COVID.',
            forecasts=len(without_covid), recession_months=sum(int(r['actual_label']) for r in without_covid),
            model_a_brier=brier_score([int(r['actual_label']) for r in without_covid],
                                     [float(r['model_a_probability']) for r in without_covid]),
            model_b_brier=brier_score([int(r['actual_label']) for r in without_covid],
                                     [float(r['model_b_probability']) for r in without_covid])),
    )
    result['verification_script_sha256'] = sha(Path(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional derived JSON output; never an experiment artifact')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output:
        output = args.output.resolve()
        protected = [root / 'docs/results' / name for name in RUNS]
        require(not any(p == output or p in output.parents for p in protected),
                'Cannot overwrite original result directories')
        require(output != Path(__file__).resolve() and output not in
                (root / p for p in ('data/processed/economic_monthly.csv',
                                   'reports/baseline_backtest/forecasts.csv')),
                'Cannot overwrite verification inputs')
    result = verify(root)
    content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        require(args.output.suffix == '.json', 'Output must be JSON')
        require(args.output.resolve() not in [root / p for p in result['input_sha256']],
                'Cannot overwrite verification inputs')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end='')


if __name__ == '__main__':
    main()

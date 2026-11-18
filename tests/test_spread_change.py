"""SUPPLEMENTARY spread-change comparison: calendar alignment, common sample, no look-ahead."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from music_recession.baseline import BaselineConfig, BaselineError, Observation, load_economic
from music_recession.economic import add_months
from music_recession.spread_change import (
    changes_from_spreads, evaluate, fit_spread_change_probit, spread_change_probability,
    spread_changes,
)

ROOT = Path(__file__).resolve().parents[1]
ECONOMIC = ROOT / 'data/processed/economic_monthly.csv'
BASE = date(1953, 4, 1)


def synthetic(n=120, seed=3):
    """A noisy recession process depending on spread and its change."""
    rng = np.random.default_rng(seed)
    spread = np.cumsum(rng.normal(0, .4, n)) * .3 + 1
    rows, spreads = [], {}
    for i in range(n):
        month = add_months(BASE, i)
        delta = spread[i] - spread[i - 1] if i else 0
        y = int(rng.random() < ndtr(-.8 - .7 * spread[i] + 1.5 * delta))
        value = Decimal(str(round(float(spread[i]), 2)))
        spreads[month] = value
        rows.append(Observation(month, add_months(month, 6), float(value), y))
    return rows, spreads


class CalendarAlignmentTests(unittest.TestCase):
    def test_gap_leaves_change_undefined_instead_of_using_previous_row(self):
        months = [date(2000, 1, 1), date(2000, 2, 1), date(2000, 4, 1), date(2000, 5, 1)]
        spreads = dict(zip(months, map(Decimal, ('1.00', '0.75', '0.10', '-0.20'))))
        changes = changes_from_spreads(spreads)
        self.assertNotIn(date(2000, 1, 1), changes)
        self.assertNotIn(date(2000, 4, 1), changes)          # not 0.10 - 0.75
        self.assertEqual(changes[date(2000, 2, 1)]['delta'], Decimal('-0.25'))
        self.assertEqual(changes[date(2000, 5, 1)]['delta'], Decimal('-0.30'))

    def test_year_rollover_pairs_december_with_january(self):
        spreads = {date(1999, 12, 1): Decimal('0.40'), date(2000, 1, 1): Decimal('0.15')}
        change = changes_from_spreads(spreads)[date(2000, 1, 1)]
        self.assertEqual((change['previous'], change['delta']), (Decimal('0.40'), Decimal('-0.25')))

    def test_real_data_changes_are_exact_and_only_first_month_is_undefined(self):
        changes = spread_changes(ECONOMIC)
        with ECONOMIC.open(newline='') as stream:
            raw = {r['predictor_month']: r for r in csv.DictReader(stream)}
        self.assertEqual(len(changes), len(raw) - 1)
        self.assertNotIn(date(1953, 4, 1), changes)
        october = changes[date(1973, 10, 1)]
        self.assertEqual(october['spread'], Decimal(raw['1973-10-01']['spread_t']))
        self.assertEqual(october['previous'], Decimal(raw['1973-09-01']['spread_t']))
        self.assertEqual(october['delta'], october['spread'] - october['previous'])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(BaselineConfig(), min_training_months=60)
        self.rows, spreads = synthetic()
        self.changes = changes_from_spreads(spreads)
        self.deltas = {m: float(v['delta']) for m, v in self.changes.items()}

    def test_probability_is_normal_cdf_of_unscaled_linear_index(self):
        training = [r for r in self.rows if r.predictor_month in self.deltas][:80]
        fit = fit_spread_change_probit(training, self.deltas, self.config)
        b = fit['report']['coefficients']
        expected = ndtr(b['intercept'] + b['spread_t'] * .5 + b['delta_spread_t'] * -.2)
        self.assertAlmostEqual(spread_change_probability(fit, .5, -.2), float(expected), places=14)

    def test_single_class_and_separation_fail_without_fallback(self):
        training = [r for r in self.rows if r.predictor_month in self.deltas][:80]
        single = [replace(r, target=0) for r in training]
        with self.assertRaises(BaselineError) as caught:
            fit_spread_change_probit(single, self.deltas, self.config)
        self.assertEqual(caught.exception.code, 'single_class')
        separated = [replace(r, target=int(r.spread < 1)) for r in training]
        with self.assertRaises(BaselineError) as caught:
            fit_spread_change_probit(separated, self.deltas, self.config)
        self.assertEqual(caught.exception.code, 'separation')

    def test_models_share_training_rows_and_respect_label_boundary(self):
        records = evaluate(self.rows, self.changes, self.config, add_months(BASE, 60), add_months(BASE, 110))
        ok = [r for r in records if r['status'] == 'ok']
        self.assertTrue(ok)
        for r in ok:
            t = date.fromisoformat(r['predictor_month'])
            self.assertLessEqual(date.fromisoformat(r['training_target_end']), add_months(t, -1))
            # The first row (1953-04) has no change, so the common sample starts 1953-05.
            self.assertEqual(r['training_start'], '1953-05-01')
            months = [m for m in self.deltas if date.fromisoformat(r['training_start']) <= m
                      and add_months(m, 6) <= add_months(t, -1)]
            digest = hashlib.sha256('\n'.join(m.isoformat() for m in sorted(months)).encode()).hexdigest()
            self.assertEqual(r['training_months_sha256'], digest)
            self.assertEqual(r['training_count'], len(months))
            self.assertEqual(r['target_month'], add_months(t, 6).isoformat())

    def test_future_spreads_and_labels_cannot_change_a_forecast(self):
        t = add_months(BASE, 90)
        before = evaluate(self.rows, self.changes, self.config, t, t)[0]
        changed_rows = [replace(r, spread=r.spread + 5, target=1 - r.target) if r.predictor_month > t
                        else r for r in self.rows]
        changed_spreads = {r.predictor_month: Decimal(str(r.spread)) for r in changed_rows}
        after = evaluate(changed_rows, changes_from_spreads(changed_spreads), self.config, t, t)[0]
        for key in ('baseline_probability', 'spread_change_probability'):
            self.assertAlmostEqual(before[key], after[key], places=12)

    def test_missing_previous_month_is_recorded_not_skipped(self):
        t = add_months(BASE, 90)
        changes = {m: v for m, v in self.changes.items() if m != t}
        record = evaluate(self.rows, changes, self.config, t, t)[0]
        self.assertEqual(record['status'], 'missing_delta')
        self.assertEqual(record['spread_change_probability'], '')


class CommittedResultTests(unittest.TestCase):
    directory = ROOT / 'docs/results/supplementary_spread_change'

    def test_committed_results_reproduce_and_originals_are_untouched(self):
        protected = [ROOT / 'reports/baseline_backtest/forecasts.csv',
                     ROOT / 'reports/baseline_backtest/metrics.json',
                     ROOT / 'config/baseline.json', ROOT / 'docs/data_manifest.json',
                     ROOT / 'docs/results/exploratory_primary_cov70/predictions.csv']
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'fresh'
            subprocess.run([sys.executable, str(ROOT / 'scripts/run_spread_change_comparison.py'),
                            '--root', str(ROOT), '--out', str(out)], check=True, capture_output=True)
            fresh = json.loads((out / 'metrics.json').read_text())
            # Parsed rows, not bytes: a core.autocrlf checkout rewrites line endings.
            with (out / 'predictions.csv').open(newline='') as stream:
                fresh_rows = list(csv.DictReader(stream))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
        self.assertEqual(fresh['evaluated_forecasts'], 629)
        self.assertEqual(fresh['not_evaluated'][0]['predictor_month'], '1973-10-01')
        self.assertEqual(fresh['not_evaluated'][0]['status'], 'warmup')
        saved = json.loads((self.directory / 'metrics.json').read_text())
        for model in ('spread_only', 'spread_plus_change'):
            for key in ('brier_score', 'roc_auc', 'average_precision', 'log_loss'):
                self.assertAlmostEqual(saved['summary'][model][key], fresh['summary'][model][key], places=12)
        with (self.directory / 'predictions.csv').open(newline='') as stream:
            self.assertEqual(fresh_rows, list(csv.DictReader(stream)))

    def test_recomputed_baseline_differs_from_saved_only_by_the_dropped_row(self):
        rows = load_economic(ECONOMIC)
        self.assertEqual(rows[0].predictor_month, date(1953, 4, 1))
        saved = json.loads((self.directory / 'metrics.json').read_text())
        context = saved['original_saved_baseline_same_months']
        self.assertLess(context['max_abs_probability_difference_vs_recomputed'], .02)
        self.assertIn('SUPPLEMENTARY', saved['label'])


class FigureExampleTests(unittest.TestCase):
    def test_worked_examples_match_saved_forecasts(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            from make_spread_probability_figure import worked_examples
        finally:
            sys.path.pop(0)
        examples = worked_examples(ROOT)
        first = examples[0]
        self.assertEqual((first['GS10_t'], first['TB3MS_t'], first['spread_t_pp']), ('6.79', '7.22', '-0.43'))
        self.assertEqual(first['inversion_indicator'], 1)
        for e in examples:
            self.assertLess(abs(e['reconstructed_probability'] - e['saved_probability']), 1e-12)
        committed = ROOT / 'docs/figures/yield_spread/worked_examples.json'
        self.assertEqual(json.loads(committed.read_text()), examples)


if __name__ == '__main__':
    unittest.main()

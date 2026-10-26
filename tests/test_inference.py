"""PR-AUC, gap-aware HAC and calendar block bootstrap for paired forecasts."""

import csv
import json
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import t as student_t

from music_recession.backtest import brier_score, brier_skill_score
from music_recession.economic import add_months
from music_recession.inference import (
    autocorrelation, average_precision, block_bootstrap, diebold_mariano,
    hac_variance_of_mean, iid_bootstrap, loss_differential, month_positions,
    paired_summary, recession_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def consecutive(n, start=date(1990, 1, 1)):
    return [add_months(start, i) for i in range(n)]


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


class AveragePrecisionTests(unittest.TestCase):
    def test_hand_calculated_ties_and_extremes(self):
        # Thresholds .8, .4, .35, .1 give precision 1, 1/2, 2/3, 1/2 at recall
        # 1/2, 1/2, 1, 1, so AP = 1/2 * 1 + 1/2 * 2/3.
        self.assertAlmostEqual(average_precision([0, 0, 1, 1], [.1, .4, .35, .8]), 5 / 6)
        self.assertEqual(average_precision([0, 1], [.5, .5]), .5)
        self.assertEqual(average_precision([0, 1, 1], [.1, .9, .8]), 1)
        self.assertIsNone(average_precision([0, 0], [.2, .3]))

    def test_invalid_inputs_rejected(self):
        for bad in (-.1, 1.1, np.nan, None):
            with self.assertRaises(ValueError):
                average_precision([0, 1], [.5, bad])


class DifferentialTests(unittest.TestCase):
    def test_positive_means_model_b_worse(self):
        d = loss_differential([0, 1], [0, 1], [.5, .5])
        np.testing.assert_allclose(d, [.25, .25])
        self.assertTrue((loss_differential([1], [.2], [.9]) < 0).all())

    def test_months_must_increase_and_be_first_of_month(self):
        for bad in (['1990-02-01', '1990-01-01'], ['1990-01-01', '1990-01-01'],
                    ['1990-01-15'], []):
            with self.assertRaises(ValueError):
                month_positions(bad)


class HacTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.d = np.convolve(rng.normal(size=80), np.ones(4), mode='valid')

    def test_matches_statsmodels_newey_west_without_gaps(self):
        months = consecutive(len(self.d))
        for lag in (0, 1, 5, 12):
            fit = sm.OLS(self.d, np.ones(len(self.d))).fit(
                cov_type='HAC', cov_kwds={'maxlags': lag, 'use_correction': False})
            with self.subTest(lag=lag):
                self.assertAlmostEqual(hac_variance_of_mean(self.d, months, lag),
                                       float(fit.cov_params()[0, 0]), places=12)

    def test_lag_zero_is_iid_variance_of_mean(self):
        months = consecutive(len(self.d))
        self.assertAlmostEqual(hac_variance_of_mean(self.d, months, 0),
                               self.d.var() / len(self.d))

    def test_rows_separated_by_a_gap_are_not_neighbours(self):
        d = np.array([1., -1, 1, -1, 1, -1])
        contiguous = consecutive(6)
        every_other = [add_months(date(1990, 1, 1), 2 * i) for i in range(6)]
        # Adjacent rows are negatively correlated, which shrinks the variance only
        # when they really are one month apart.
        self.assertLess(hac_variance_of_mean(d, contiguous, 1), hac_variance_of_mean(d, contiguous, 0))
        self.assertAlmostEqual(hac_variance_of_mean(d, every_other, 1),
                               hac_variance_of_mean(d, every_other, 0))

    def test_nonnegative_with_gaps(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            offsets = np.cumsum(rng.integers(1, 30, size=40))
            months = [add_months(date(1980, 1, 1), int(k)) for k in offsets]
            d = rng.normal(size=40)
            for lag in (5, 24, 120):
                self.assertGreaterEqual(hac_variance_of_mean(d, months, lag), 0)

    def test_diebold_mariano_statistic_and_degenerate_series(self):
        months = consecutive(len(self.d))
        result = diebold_mariano(self.d, months, 0)
        expected = self.d.mean() / np.sqrt(self.d.var() / len(self.d))
        self.assertAlmostEqual(result['statistic'], expected)
        self.assertAlmostEqual(result['p_value_two_sided'],
                               2 * student_t.sf(abs(expected), len(self.d) - 1))
        self.assertAlmostEqual(result['effective_observations'], len(self.d))
        flat = diebold_mariano(np.full(10, .3), consecutive(10), 5)
        self.assertIsNone(flat['statistic'])
        self.assertAlmostEqual(flat['mean'], .3)
        with self.assertRaises(ValueError):
            hac_variance_of_mean(self.d, months, -1)

    def test_autocorrelation_reports_pairs_and_skips_gaps(self):
        months = [date(1990, 1, 1), date(1990, 2, 1), date(1995, 1, 1)]
        result = autocorrelation([1., 2, 3], months, (1, 2))
        self.assertEqual(result['1']['pairs'], 1)
        self.assertEqual(result['2'], dict(value=None, pairs=0))


class BootstrapTests(unittest.TestCase):
    def test_seeded_and_reproducible(self):
        d = np.linspace(-1, 2, 30)
        months = consecutive(30)
        self.assertEqual(block_bootstrap(d, months, 6, 500, seed=1),
                         block_bootstrap(d, months, 6, 500, seed=1))
        self.assertEqual(iid_bootstrap(d, 500, seed=1), iid_bootstrap(d, 500, seed=1))

    def test_block_as_long_as_span_always_redraws_the_sample(self):
        d = np.array([.1, .4, -.2, .3])
        months = [date(2000, 1, 1), date(2000, 3, 1), date(2001, 1, 1), date(2002, 6, 1)]
        span = month_positions(months)[-1] - month_positions(months)[0] + 1
        result = block_bootstrap(d, months, int(span), 200)
        self.assertAlmostEqual(result['ci_lower'], d.mean())
        self.assertAlmostEqual(result['ci_upper'], d.mean())

    def test_constant_series_and_empty_replicates(self):
        months = [date(2000, 1, 1), date(2010, 1, 1)]
        result = block_bootstrap([.2, .2], months, 1, 400, seed=2)
        self.assertGreater(result['empty_replicates_dropped'], 0)
        self.assertEqual(result['replications'] + result['empty_replicates_dropped'], 400)
        self.assertAlmostEqual(result['ci_lower'], .2)
        self.assertAlmostEqual(result['ci_upper'], .2)

    def test_block_length_validated(self):
        months = consecutive(5)
        for bad in (0, 6):
            with self.assertRaises(ValueError):
                block_bootstrap(np.zeros(5), months, bad, 10)


class SummaryTests(unittest.TestCase):
    def test_recession_episodes_split_on_breaks(self):
        rows = [dict(predictor_month=m.isoformat(), USREC_t=v) for m, v in zip(
            consecutive(8), [0, 1, 1, 0, 1, 0, 0, 1])]
        self.assertEqual(recession_episodes(rows), [
            (date(1990, 2, 1), date(1990, 3, 1)), (date(1990, 5, 1), date(1990, 5, 1)),
            (date(1990, 8, 1), date(1990, 8, 1))])

    def test_synthetic_paired_summary(self):
        rows = []
        for i, month in enumerate(consecutive(30)):
            label = int(10 <= i < 13)
            rows.append(dict(predictor_month=month.isoformat(),
                             target_month=add_months(month, 6).isoformat(),
                             status='ok' if i % 7 else 'ineligible_features', actual_label=label,
                             model_a_probability=.6 if label else .1,
                             model_b_probability=.3 if label else .2,
                             training_recession_count=3, training_count=30))
        # Labels sit on predictor months 1990-11..1991-01, so targets are 1991-05..07.
        episodes = [(date(1990, 11, 1), date(1991, 1, 1)), (date(1991, 5, 1), date(1991, 7, 1))]
        result = paired_summary(rows, episodes, block_lengths=(3,), hac_lags=(2,), replications=200)
        ok = [r for r in rows if r['status'] == 'ok']
        self.assertEqual(result['forecasts'], len(ok))
        self.assertEqual(result['recession_target_months'], 3)
        self.assertEqual(result['recession_episodes_reached'], ['1991-05 to 1991-07'])
        y = [r['actual_label'] for r in ok]
        reference = brier_score(y, [.1] * len(y))
        model_b = brier_score(y, [r['model_b_probability'] for r in ok])
        self.assertAlmostEqual(result['model_b']['brier_skill_vs_training_prevalence'],
                               brier_skill_score(model_b, reference))
        self.assertEqual(result['model_a']['roc_auc'], 1)
        self.assertGreater(result['loss_differential']['mean'], 0)
        with self.assertRaises(ValueError):
            paired_summary([dict(rows[0], status='warmup')], episodes)


class CommittedResultTests(unittest.TestCase):
    """The saved inference JSON must follow from the committed predictions."""

    def test_primary_run_reconciles_with_metrics_and_saved_json(self):
        directory = ROOT / 'docs/results/exploratory_primary_cov70'
        metrics = json.loads((directory / 'metrics.json').read_text())
        episodes = recession_episodes(read_rows(ROOT / 'data/processed/economic_monthly.csv'))
        result = paired_summary(read_rows(directory / 'predictions.csv'), episodes)
        self.assertEqual(result['forecasts'], metrics['evaluated_months'])
        self.assertAlmostEqual(result['loss_differential']['mean'],
                               metrics['brier_difference_b_minus_a'], places=12)
        for model in ('model_a', 'model_b'):
            self.assertAlmostEqual(result[model]['brier_score'], metrics[model]['brier_score'], places=12)
            self.assertAlmostEqual(result[model]['roc_auc'], metrics[model]['roc_auc'], places=12)
        # The first-reported i.i.d. interval was [+0.0147, +0.0458].
        iid = result['loss_differential']['iid_bootstrap']
        self.assertAlmostEqual(iid['ci_lower'], .0147, delta=.002)
        self.assertAlmostEqual(iid['ci_upper'], .0458, delta=.002)
        saved_path = ROOT / 'docs/results/paired_inference.json'
        if not saved_path.exists():
            self.skipTest('paired_inference.json not generated yet')
        saved = json.loads(saved_path.read_text())['runs'][0]
        for model in ('model_a', 'model_b'):
            for key, value in result[model].items():
                self.assertAlmostEqual(saved[model][key], value, places=12)
        for fresh, old in zip(result['loss_differential']['diebold_mariano'],
                              saved['loss_differential']['diebold_mariano']):
            self.assertAlmostEqual(fresh['statistic'], old['statistic'], places=10)
        # Bootstrap draws may shift slightly across NumPy releases; the bounds should not.
        for fresh, old in zip(result['loss_differential']['block_bootstrap'],
                              saved['loss_differential']['block_bootstrap']):
            self.assertAlmostEqual(fresh['ci_lower'], old['ci_lower'], delta=.003)
            self.assertAlmostEqual(fresh['ci_upper'], old['ci_upper'], delta=.003)


if __name__ == '__main__':
    unittest.main()

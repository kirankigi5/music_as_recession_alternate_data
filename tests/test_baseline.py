"""Deterministic Probit checks, eligibility boundaries and failure handling."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, ndtri
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from music_recession.baseline import (
    BaselineConfig, BaselineError, Observation, first_eligible_forecast_month,
    fit_probit, forecast_one, load_economic, parse_month, select_training,
)
from music_recession.economic import add_months


ROOT = Path(__file__).resolve().parents[1]


def sample(spreads, targets):
    return [Observation(add_months(date(1953, 4, 1), i),
                        add_months(date(1953, 4, 1), i + 6), float(x), y)
            for i, (x, y) in enumerate(zip(spreads, targets))]


def balanced_sample():
    # Two groups with known MLE probabilities: P(y=1 | x=-1)=.75,
    # P(y=1 | x=+1)=.25. Both classes occur at both spread levels.
    return sample([-1] * 40 + [1] * 40,
                  [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30)


class ProbitTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(BaselineConfig(), min_training_months=40)
        self.rows = balanced_sample()

    def assert_failure(self, code, function, *args):
        with self.assertRaises(BaselineError) as caught:
            function(*args)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_known_probabilities_intercept_and_coefficient_direction(self):
        fitted = fit_probit(self.rows, self.config)
        self.assertIsInstance(fitted._result.model, Probit)
        coefficients = fitted.report["coefficients"]
        self.assertAlmostEqual(coefficients["intercept"], 0, places=9)
        self.assertAlmostEqual(coefficients["spread_t"], ndtri(0.25), places=9)
        self.assertLess(coefficients["spread_t"], 0)
        np.testing.assert_allclose(fitted.predict_probability([-1, 0, 1]), [.75, .5, .25], atol=1e-10)
        design = fitted._result.model.exog
        np.testing.assert_array_equal(design[:, 0], np.ones(80))
        self.assertEqual(design.shape, (80, 2))

    def test_nonzero_intercept_is_estimated(self):
        rows = sample([-1] * 40 + [1] * 40,
                      [1] * 20 + [0] * 20 + [1] * 10 + [0] * 30)
        fitted = fit_probit(rows, self.config)
        self.assertAlmostEqual(fitted.report["coefficients"]["intercept"], ndtri(.25) / 2, places=9)
        np.testing.assert_allclose(fitted.predict_probability([-1, 1]), [.5, .25], atol=1e-10)

    def test_probabilities_follow_normal_cdf_and_are_bounded(self):
        fitted = fit_probit(self.rows, self.config)
        spreads = np.array([-100, -2, -.5, 0, 2, 100])
        coeff = fitted.report["coefficients"]
        probabilities = fitted.predict_probability(spreads)
        np.testing.assert_allclose(probabilities, ndtr(coeff["intercept"] + coeff["spread_t"] * spreads))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertTrue(np.all(np.diff(probabilities) <= 0))

    def test_reproducible_fits_and_input_preservation(self):
        before = list(self.rows)
        first, second = fit_probit(self.rows, self.config), fit_probit(self.rows, self.config)
        self.assertEqual(first.report, second.report)
        np.testing.assert_array_equal(first.predict_probability([0.3]), second.predict_probability([0.3]))
        self.assertEqual(self.rows, before)

    def test_diagnostics_and_training_metadata(self):
        report = fit_probit(self.rows, self.config).report
        self.assertEqual(report["training"]["sample_count"], 80)
        self.assertEqual(report["training"]["class_counts"], {"0": 40, "1": 40})
        self.assertEqual(report["training"]["predictor_start"], "1953-04-01")
        self.assertEqual(report["training"]["target_start"], "1953-10-01")
        self.assertEqual(report["standardization"], "none; spread is in percentage points")
        self.assertTrue(report["diagnostics"]["converged"])
        self.assertGreater(report["diagnostics"]["iterations"], 0)
        self.assertLess(report["diagnostics"]["max_abs_average_score"], self.config.max_average_score)
        self.assertEqual(report["diagnostics"]["warnings"], [])

    def test_missing_and_nonfinite_inputs_rejected(self):
        for value in (None, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value):
                rows = [replace(self.rows[0], spread=value)] + self.rows[1:]
                self.assert_failure("missing_input", fit_probit, rows, self.config)
        rows = [replace(self.rows[0], target=None)] + self.rows[1:]
        self.assert_failure("missing_input", fit_probit, rows, self.config)
        self.assert_failure("missing_input", fit_probit, [], self.config)

    def test_nonbinary_target_rejected(self):
        for value in (-1, 2, .5, float("nan")):
            with self.subTest(value=value):
                rows = [replace(self.rows[0], target=value)] + self.rows[1:]
                self.assert_failure("invalid_target", fit_probit, rows, self.config)

    def test_constant_predictor_and_single_class_rejected(self):
        self.assert_failure("constant_predictor", fit_probit,
                            [replace(r, spread=1) for r in self.rows], self.config)
        for label in (0, 1):
            self.assert_failure("single_class", fit_probit,
                                [replace(r, target=label) for r in self.rows], self.config)

    def test_complete_and_quasi_complete_separation_rejected(self):
        config = replace(self.config, min_training_months=4)
        for xs, ys in [([-2, -1, 1, 2], [0, 0, 1, 1]),
                       ([-2, -1, 1, 2], [1, 1, 0, 0]),
                       ([-1, 0, 0, 1], [0, 0, 1, 1])]:
            with self.subTest(xs=xs, ys=ys):
                self.assert_failure("separation", fit_probit, sample(xs, ys), config)

    def test_ill_conditioned_design_rejected(self):
        rows = [replace(r, spread=1e12 + r.spread) for r in self.rows]
        self.assert_failure("ill_conditioned", fit_probit, rows, self.config)

    def test_real_optimizer_iteration_limit_is_failure(self):
        error = self.assert_failure("nonconvergence", fit_probit, self.rows,
                                    replace(self.config, maxiter=1))
        self.assertFalse(error.diagnostics["converged"])

    def test_optimizer_exception_has_no_fallback(self):
        with patch("music_recession.baseline.Probit.fit", side_effect=RuntimeError("forced failure")) as fit:
            error = self.assert_failure("optimization_failure", fit_probit, self.rows, self.config)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(error.diagnostics["exception_type"], "RuntimeError")

    def test_separation_warning_is_failure(self):
        original = Probit.fit

        def warning_fit(model, *args, **kwargs):
            warnings.warn("forced separation warning", PerfectSeparationWarning)
            return original(model, *args, **kwargs)

        with patch("music_recession.baseline.Probit.fit", new=warning_fit):
            self.assert_failure("separation", fit_probit, self.rows, self.config)

    def test_false_success_with_nonfinite_parameters_is_rejected(self):
        bad = SimpleNamespace(mle_retvals={"converged": True},
                              params=np.array([np.nan, 0]), bse=np.ones(2), llf=np.nan)
        with patch("music_recession.baseline.Probit.fit", return_value=bad):
            self.assert_failure("numerical_failure", fit_probit, self.rows, self.config)

    def test_large_likelihood_score_is_rejected_despite_converged_flag(self):
        with patch("music_recession.baseline.Probit.score", return_value=np.array([10., 10.])):
            self.assert_failure("nonconvergence", fit_probit, self.rows, self.config)

    def test_bad_prediction_inputs_rejected(self):
        fitted = fit_probit(self.rows, self.config)
        for values in ([], [None], [np.nan], [np.inf], [[1, 2]], 1.0, ["bad"]):
            with self.subTest(values=values):
                self.assert_failure("missing_input", fitted.predict_probability, values)

    def test_duplicate_unsorted_and_misaligned_training_rows_rejected(self):
        self.assert_failure("invalid_dates", fit_probit, [self.rows[0]] + self.rows, self.config)
        self.assert_failure("invalid_dates", fit_probit, self.rows[::-1], self.config)
        rows = [replace(self.rows[0], target_month=date(1953, 11, 1))] + self.rows[1:]
        self.assert_failure("misaligned_target", fit_probit, rows, self.config)

    def test_invalid_config_and_dates_rejected(self):
        for changes in ({"model": "logit"}, {"optimizer": "bfgs"}, {"standardize": True},
                        {"horizon_months": 5}, {"min_training_months": 0},
                        {"forecast_cutoff_day": 32}, {"forecast_cutoff_month_offset": 0},
                        {"training_target_lag_months": 0}, {"maxiter": 0},
                        {"tolerance": np.nan}, {"max_average_score": -1}):
            with self.subTest(changes=changes):
                with self.assertRaises(BaselineError):
                    replace(self.config, **changes)
        for value in ("1973-10-15", "1973-13-01", "not-a-date"):
            with self.subTest(value=value):
                self.assert_failure("invalid_date", parse_month, value)


class TimingAndSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = BaselineConfig.load(ROOT / "config/baseline.json")
        cls.rows = load_economic(ROOT / "data/processed/economic_monthly.csv")
        cls.month = date(1973, 10, 1)

    def test_first_eligible_month_and_training_dates(self):
        self.assertEqual(first_eligible_forecast_month(self.rows, self.config), self.month)
        training = select_training(self.rows, self.month, self.config)
        self.assertEqual(len(training), 240)
        self.assertEqual(training[0].predictor_month, date(1953, 4, 1))
        self.assertEqual(training[-1].predictor_month, date(1973, 3, 1))
        self.assertEqual(training[0].target_month, date(1953, 10, 1))
        self.assertEqual(training[-1].target_month, date(1973, 9, 1))
        self.assertTrue(all(r.target_month <= date(1973, 9, 1) for r in training))
        with self.assertRaises(BaselineError) as caught:
            select_training(self.rows, date(1973, 9, 1), self.config)
        self.assertEqual(caught.exception.code, "insufficient_training")
        self.assertEqual(caught.exception.diagnostics["eligible_months"], 239)

    def test_all_training_labels_independently_match_raw_future_months(self):
        with (ROOT / "fredgraph.csv").open(newline="") as stream:
            raw = {r["observation_date"]: r for r in csv.DictReader(stream)}
        training = select_training(self.rows, self.month, self.config)
        for row in training:
            target_ordinal = row.predictor_month.year * 12 + row.predictor_month.month - 1 + 6
            target_key = f"{target_ordinal // 12:04d}-{target_ordinal % 12 + 1:02d}-01"
            self.assertEqual(row.target_month.isoformat(), target_key)
            self.assertEqual(row.target, int(raw[target_key]["USREC"]))

    def test_single_forecast_cutoff_target_and_negative_slope(self):
        report = forecast_one(self.rows, self.month, self.config)
        self.assertEqual(report["forecast_cutoff"], "1973-11-15")
        self.assertEqual(report["target_month"], "1974-04-01")
        self.assertEqual(report["latest_allowed_training_target_month"], "1973-09-01")
        self.assertEqual(report["fit"]["training"]["sample_count"], 240)
        self.assertTrue(report["fit"]["diagnostics"]["converged"])
        self.assertLess(report["fit"]["coefficients"]["spread_t"], 0)
        self.assertGreater(report["recession_probability"], 0)
        self.assertLess(report["recession_probability"], 1)
        self.assertNotIn("actual_target", report)

    def test_excluded_future_predictors_and_labels_cannot_affect_forecast(self):
        baseline = forecast_one(self.rows, self.month, self.config)
        changed = [replace(r, spread=99999., target=1 - r.target)
                   if r.target_month > date(1973, 9, 1) and r.predictor_month != self.month
                   else r for r in self.rows]
        # Also remove the forecast row's future label entirely.
        changed = [replace(r, target=None) if r.predictor_month == self.month else r for r in changed]
        alternative = forecast_one(changed, self.month, self.config)
        self.assertEqual(baseline, alternative)

    def test_forecast_spread_changes_prediction_but_not_training_fit(self):
        baseline = forecast_one(self.rows, self.month, self.config)
        changed = [replace(r, spread=r.spread + 1) if r.predictor_month == self.month else r for r in self.rows]
        alternative = forecast_one(changed, self.month, self.config)
        self.assertEqual(baseline["fit"], alternative["fit"])
        self.assertLess(alternative["recession_probability"], baseline["recession_probability"])

    def test_t_minus_one_boundary_is_inclusive_and_t_is_excluded(self):
        training = select_training(self.rows, self.month, self.config)
        self.assertIn(date(1973, 9, 1), [r.target_month for r in training])
        self.assertNotIn(date(1973, 10, 1), [r.target_month for r in training])
        self.assertNotIn(date(1973, 11, 1), [r.target_month for r in training])

    def test_missing_eligible_label_is_error_not_silent_row_drop(self):
        changed = [replace(self.rows[0], target=None)] + self.rows[1:]
        with self.assertRaises(BaselineError) as caught:
            forecast_one(changed, self.month, self.config)
        self.assertEqual(caught.exception.code, "missing_input")

    def test_changed_training_start_and_lag_are_respected(self):
        later_start = replace(self.config, training_start="1953-05-01")
        self.assertEqual(first_eligible_forecast_month(self.rows, later_start), date(1973, 11, 1))
        longer_lag = replace(self.config, training_target_lag_months=2)
        self.assertEqual(first_eligible_forecast_month(self.rows, longer_lag), date(1973, 11, 1))

    def test_missing_training_month_counts_actual_observations(self):
        changed = self.rows[:100] + self.rows[101:]
        self.assertEqual(first_eligible_forecast_month(changed, self.config), date(1973, 11, 1))

    def test_cli_report_and_failure_preserves_successful_report(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke.json"
            command = [sys.executable, "-m", "music_recession.baseline",
                       "--data", str(ROOT / "data/processed/economic_monthly.csv"),
                       "--config", str(ROOT / "config/baseline.json"), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            stored = output.read_bytes()
            report = json.loads(stored)
            self.assertEqual(report["first_eligible_forecast_month"], "1973-10-01")
            self.assertEqual(report["fit"]["training"]["sample_count"], 240)
            result = subprocess.run(command + ["--predictor-month", "1973-09-01"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stderr)["code"], "insufficient_training")
            self.assertEqual(output.read_bytes(), stored)


class ProcessedInputTests(unittest.TestCase):
    def test_corrupt_spread_and_target_alignment_fail(self):
        with (ROOT / "data/processed/economic_monthly.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        for column, value, expected in (("spread_t", "99", "invalid_spread"),
                                        ("target_month", "1953-11-01", "misaligned_target"),
                                        ("GS10_t", "", "missing_input")):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "economic.csv"
                changed = [dict(rows[0], **{column: value})] + rows[1:]
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(changed)
                with self.assertRaises(BaselineError) as caught:
                    load_economic(path)
                self.assertEqual(caught.exception.code, expected)


if __name__ == "__main__":
    unittest.main()

"""Metrics, calendar eligibility, leakage, failure visibility and reproducibility."""

import csv
import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np

from music_recession.baseline import BaselineConfig, BaselineError, fit_probit, load_economic
from music_recession.backtest import (
    START, END, FORECAST_COLUMNS, brier_score, brier_skill_score, calibration_table,
    calendar_months, log_loss, main, roc_auc, run_backtest, summarize, write_reports,
)


ROOT = Path(__file__).resolve().parents[1]


class MetricTests(unittest.TestCase):
    def test_brier_and_skill_hand_calculated(self):
        model = brier_score([0, 1, 1, 0], [.1, .8, .6, .4])
        reference = brier_score([0, 1, 1, 0], [.5] * 4)
        self.assertAlmostEqual(model, .0925)
        self.assertEqual(reference, .25)
        self.assertAlmostEqual(brier_skill_score(model, reference), .63)
        self.assertEqual(brier_skill_score(.5, .25), -1)
        self.assertEqual(brier_skill_score(.25, .25), 0)

    def test_perfect_and_worst_probabilities(self):
        self.assertEqual(brier_score([0, 1], [0, 1]), 0)
        self.assertEqual(brier_score([0, 1], [1, 0]), 1)
        self.assertIsNone(brier_skill_score(0, 0))

    def test_invalid_probabilities_labels_and_missing_values_rejected(self):
        for bad in (-.01, 1.01, np.nan, np.inf, None, "bad"):
            for function in (brier_score, roc_auc, log_loss, calibration_table):
                with self.subTest(bad=bad, function=function):
                    with self.assertRaises(ValueError):
                        function([0, 1], [.5, bad])
        for actual, p in (([], []), ([0], [.5, .5]), ([[0]], [[.5]]),
                          ([0, None], [.5, .5]), ([0, 2], [.5, .5])):
            with self.subTest(actual=actual):
                with self.assertRaises(ValueError):
                    brier_score(actual, p)
        for score in (-1, 2, np.nan, np.inf):
            with self.assertRaises(ValueError):
                brier_skill_score(score, .2)

    def test_auc_ties_single_class_and_known_ranking(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)
        self.assertEqual(roc_auc([0, 1], [.5, .5]), .5)
        self.assertEqual(roc_auc([0, 1], [.9, .1]), 0)
        self.assertIsNone(roc_auc([0, 0], [.1, .2]))
        self.assertIsNone(roc_auc([1, 1], [.1, .2]))

    def test_log_loss_floors_only_zero_outcome_likelihood(self):
        self.assertEqual(log_loss([0, 1], [0, 1])["value"], 0)
        self.assertAlmostEqual(log_loss([0, 1], [.5, .5])["value"], math.log(2))
        wrong = log_loss([0, 1], [1, 0])
        self.assertEqual(wrong["clipped_observations"], 2)
        self.assertAlmostEqual(wrong["value"], -math.log(np.finfo(float).eps))
        tiny = log_loss([1], [1e-20])
        self.assertEqual(tiny["clipped_observations"], 0)
        self.assertAlmostEqual(tiny["value"], -math.log(1e-20))

    def test_calibration_bins_boundaries_empty_bins_and_counts(self):
        bins = calibration_table([0, 1, 1, 0, 1], [0, .1, .3, .9, 1])
        self.assertEqual(len(bins), 10)
        self.assertEqual([b["count"] for b in bins], [1, 1, 0, 1, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(bins[2]["mean_predicted_probability"])
        self.assertAlmostEqual(bins[-1]["mean_predicted_probability"], .95)
        self.assertEqual(bins[-1]["observed_recession_frequency"], .5)
        self.assertTrue(bins[-1]["upper_inclusive"])


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = BaselineConfig.load(ROOT / "config/baseline.json")
        cls.rows = load_economic(ROOT / "data/processed/economic_monthly.csv")
        cls.records = run_backtest(cls.rows, cls.config)
        with (ROOT / "fredgraph.csv").open(newline="") as stream:
            cls.raw = {r["observation_date"]: r for r in csv.DictReader(stream)}

    def test_independent_expected_630_months_and_final_eligibility(self):
        # Independent monthly-ordinal arithmetic, not the production date loop.
        expected = [f"{k // 12:04d}-{k % 12 + 1:02d}-01"
                    for k in range(1973 * 12 + 9, 2026 * 12 + 3)]
        self.assertEqual(len(expected), 630)
        self.assertEqual([r["predictor_month"] for r in self.records], expected)
        self.assertTrue(all(r["status"] == "ok" for r in self.records))
        first, last = self.records[0], self.records[-1]
        self.assertEqual(first["training_sample_count"], 240)
        self.assertEqual(last["training_sample_count"], 869)
        self.assertEqual(last["training_predictor_end"], "2025-08-01")
        self.assertEqual(last["training_target_end"], "2026-02-01")
        self.assertEqual(last["forecast_cutoff"], "2026-04-15")
        self.assertEqual(last["target_month"], "2026-09-01")
        beyond = run_backtest(self.rows, self.config, date(2026, 4, 1), date(2026, 4, 1))
        self.assertEqual(beyond[0]["error_code"], "missing_forecast")
        before = run_backtest(self.rows, self.config, date(1973, 9, 1), date(1973, 9, 1))
        self.assertEqual(before[0]["error_code"], "insufficient_training")

    def test_expanding_windows_targets_and_reference_independently(self):
        for index, record in enumerate(self.records):
            self.assertEqual(record["training_sample_count"], 240 + index)
            predictor = date.fromisoformat(record["predictor_month"])
            ordinal = predictor.year * 12 + predictor.month - 1
            label_limit = ordinal - 1
            eligible = []
            for source_month, raw in self.raw.items():
                number = int(source_month[:4]) * 12 + int(source_month[5:7]) - 1
                if source_month >= "1953-04-01" and number + 6 <= label_limit and raw["GS10"] and raw["TB3MS"]:
                    target_number = number + 6
                    target_key = f"{target_number // 12:04d}-{target_number % 12 + 1:02d}-01"
                    eligible.append((source_month, int(self.raw[target_key]["USREC"])))
            self.assertEqual(len(eligible), record["training_sample_count"])
            positives = sum(label for _, label in eligible)
            self.assertEqual(record["training_recession_count"], positives)
            self.assertAlmostEqual(record["reference_probability"], positives / len(eligible))
            self.assertEqual(record["reference_probability"], record["training_recession_prevalence"])
            self.assertEqual(record["training_predictor_start"], eligible[0][0])
            self.assertEqual(record["training_predictor_end"], eligible[-1][0])
            months_hash = hashlib.sha256("\n".join(m for m, _ in eligible).encode()).hexdigest()
            self.assertEqual(record["training_months_sha256"], months_hash)
            target_number = ordinal + 6
            self.assertEqual(record["target_month"], f"{target_number // 12:04d}-{target_number % 12 + 1:02d}-01")
            self.assertEqual(record["actual_label"], int(self.raw[record["target_month"]]["USREC"]))

    def test_first_prediction_exactly_matches_step3_smoke(self):
        smoke = json.loads((ROOT / "reports/baseline_smoke.json").read_text())
        first = self.records[0]
        self.assertAlmostEqual(first["predicted_probability"], smoke["recession_probability"], places=12)
        self.assertAlmostEqual(first["intercept"], smoke["fit"]["coefficients"]["intercept"], places=12)
        self.assertAlmostEqual(first["spread_coefficient"], smoke["fit"]["coefficients"]["spread_t"], places=12)

    def test_reproducibility(self):
        repeated = run_backtest(self.rows, self.config, START, date(1973, 12, 1))
        self.assertEqual(repeated, self.records[:3])
        self.assertEqual(summarize(repeated), summarize(self.records[:3]))

    def test_ineligible_future_observations_cannot_change_earlier_forecast(self):
        changed = [replace(r, spread=9999., target=1 - r.target)
                   if r.target_month > date(1973, 9, 1) and r.predictor_month != START else r
                   for r in self.rows]
        earlier = run_backtest(changed, self.config, START, START)
        self.assertEqual(earlier[0], self.records[0])

    def test_forecast_target_label_does_not_enter_fit_or_benchmark(self):
        changed = [replace(r, target=1 - r.target) if r.predictor_month == START else r for r in self.rows]
        earlier = run_backtest(changed, self.config, START, START)[0]
        for key in ("predicted_probability", "reference_probability", "coefficients_json", "training_months_sha256"):
            self.assertEqual(earlier[key], self.records[0][key])
        self.assertNotEqual(earlier["actual_label"], self.records[0]["actual_label"])

    def test_nonconvergent_forecast_retained_and_paired_metrics_explicit(self):
        calls = []

        def fit_with_one_failure(training, config):
            calls.append(len(training))
            if len(training) == 241:
                raise BaselineError("nonconvergence", "forced failure", {"converged": False, "iterations": 1})
            return fit_probit(training, config)

        with patch("music_recession.backtest.fit_probit", side_effect=fit_with_one_failure):
            records = run_backtest(self.rows, self.config, START, date(1973, 12, 1))
        self.assertEqual(calls, [240, 241, 242])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]["error_code"], "nonconvergence")
        self.assertIsNone(records[1]["predicted_probability"])
        self.assertEqual(records[1]["training_sample_count"], 241)
        self.assertFalse(records[1]["converged"])
        metrics = summarize(records)
        self.assertEqual(metrics["coverage"]["failed_forecasts"], 1)
        self.assertEqual(metrics["coverage"]["evaluated_forecasts"], 2)
        self.assertFalse(metrics["coverage"]["full_evaluation_coverage"])
        paired = [records[0], records[2]]
        self.assertEqual(metrics["reference_brier_score"],
                         brier_score([r["actual_label"] for r in paired], [r["reference_probability"] for r in paired]))
        self.assertEqual(metrics["coverage"]["not_evaluated_months"][0]["predictor_month"], "1973-11-01")

    def test_actual_optimizer_nonconvergence_is_retained(self):
        records = run_backtest(self.rows, replace(self.config, maxiter=1), START, START)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(records[0]["error_code"], "nonconvergence")
        metrics = summarize(records)
        self.assertEqual(metrics["coverage"]["evaluated_forecasts"], 0)
        self.assertIsNone(metrics["brier_score"])

    def test_nonfinite_failure_diagnostic_is_preserved_as_explicit_text(self):
        failure = BaselineError("ill_conditioned", "bad information matrix",
                                {"information_condition_number": float("inf")})
        with patch("music_recession.backtest.fit_probit", side_effect=failure):
            records = run_backtest(self.rows, self.config, START, START)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(json.loads(records[0]["diagnostics_json"])["information_condition_number"], "inf")
        json.dumps(records, allow_nan=False)

    def test_missing_forecast_row_and_missing_actual_are_visible(self):
        missing_row = [r for r in self.rows if r.predictor_month != START]
        records = run_backtest(missing_row, self.config, START, date(1973, 11, 1))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["error_code"], "missing_forecast")
        self.assertEqual(records[1]["status"], "ok")
        missing_label = [replace(r, target=None) if r.predictor_month == START else r for r in self.rows]
        records = run_backtest(missing_label, self.config, START, START)
        self.assertEqual(records[0]["status"], "missing_actual")
        self.assertEqual(records[0]["predicted_probability"], self.records[0]["predicted_probability"])
        metrics = summarize(records)
        self.assertEqual(metrics["coverage"]["successful_forecasts"], 1)
        self.assertEqual(metrics["coverage"]["evaluated_forecasts"], 0)

    def test_missing_training_label_is_a_reported_failure(self):
        rows = [replace(self.rows[0], target=None)] + self.rows[1:]
        records = run_backtest(rows, self.config, START, START)
        self.assertEqual(records[0]["error_code"], "missing_input")
        self.assertEqual(records[0]["status"], "failed")

    def test_invalid_input_aborts_instead_of_silently_dropping_rows(self):
        rows = [replace(self.rows[0], spread=np.nan)] + self.rows[1:]
        with self.assertRaises(BaselineError):
            run_backtest(rows, self.config)
        with self.assertRaises(ValueError):
            calendar_months(END, START)

    def test_full_metrics_match_independent_pairwise_auc_and_scores(self):
        metrics = summarize(self.records)
        n = len(self.records)
        manual_bs = sum((r["predicted_probability"] - r["actual_label"]) ** 2 for r in self.records) / n
        manual_ref = sum((r["reference_probability"] - r["actual_label"]) ** 2 for r in self.records) / n
        self.assertAlmostEqual(metrics["brier_score"], manual_bs, places=14)
        self.assertAlmostEqual(metrics["brier_skill_score"], 1 - manual_bs / manual_ref, places=14)
        positive = [r["predicted_probability"] for r in self.records if r["actual_label"] == 1]
        negative = [r["predicted_probability"] for r in self.records if r["actual_label"] == 0]
        manual_auc = sum((p > q) + .5 * (p == q) for p in positive for q in negative) / (len(positive) * len(negative))
        self.assertAlmostEqual(metrics["roc_auc"], manual_auc, places=14)
        self.assertEqual(sum(b["count"] for b in metrics["calibration"]), 630)

    def test_reports_preserve_stable_schema_failures_and_create_plots(self):
        records = [dict(self.records[0]), dict(self.records[1])]
        records[1].update(status="failed", predicted_probability=None, error_code="nonconvergence", error_message="test")
        metrics = summarize(records)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_reports(records, metrics, output)
            with (output / "forecasts.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(FORECAST_COLUMNS))
                written = list(reader)
            self.assertEqual(len(written), 2)
            self.assertEqual(written[1]["status"], "failed")
            self.assertEqual(len(json.loads((output / "failures.json").read_text())), 1)
            for stem in ("probability_timeline", "calibration", "brier_comparison"):
                for extension in ("png", "svg"):
                    self.assertGreater((output / f"{stem}.{extension}").stat().st_size, 100)

    def test_cli_returns_nonzero_for_incomplete_coverage_and_saves_all_months(self):
        with tempfile.TemporaryDirectory() as temp, patch("builtins.print"):
            exit_code = main(["--data", str(ROOT / "data/processed/economic_monthly.csv"),
                              "--config", str(ROOT / "config/baseline.json"),
                              "--start", "1973-09-01", "--end", "1973-10-01", "--output-dir", temp])
            self.assertEqual(exit_code, 2)
            metrics = json.loads((Path(temp) / "metrics.json").read_text())
            self.assertEqual(metrics["coverage"]["scheduled_forecasts"], 2)
            self.assertEqual(metrics["coverage"]["evaluated_forecasts"], 1)


if __name__ == "__main__":
    unittest.main()

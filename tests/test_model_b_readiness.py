"""Synthetic tests for the readiness measurement; no lyric text or corpus required."""
import unittest
from datetime import date

from music_recession.baseline import (BaselineConfig, BaselineError, Observation,
                                      select_training)
from music_recession.economic import add_months
from scripts.measure_model_b_readiness import (HORIZON_MONTHS, MIN_TRAINING_MONTHS,
                                               TRAINING_TARGET_LAG_MONTHS,
                                               attainable_forecasts, longest_run,
                                               scenario_membership)


def month(year, m):
    return date(year, m, 1)


class LongestRunTests(unittest.TestCase):
    def test_gap_breaks_the_run(self):
        months = [month(1973, 1), month(1973, 2), month(1973, 3), month(1973, 5), month(1973, 6)]
        self.assertEqual(longest_run(months), 3)

    def test_empty_and_single(self):
        self.assertEqual(longest_run([]), 0)
        self.assertEqual(longest_run([month(1973, 1)]), 1)

    def test_year_boundary_is_contiguous(self):
        self.assertEqual(longest_run([month(1973, 12), month(1974, 1)]), 2)


class AttainableForecastTests(unittest.TestCase):
    def months(self, count):
        return [month(1973 + (i // 12), i % 12 + 1) for i in range(count)]

    def test_exactly_240_training_rows_are_not_yet_a_forecast(self):
        # 240 contiguous months supply labels only up to t-1 after the 6-month lag,
        # so the first forecast needs 247 eligible months, not 240.
        self.assertEqual(attainable_forecasts(self.months(246)), [])
        self.assertEqual(attainable_forecasts(self.months(247)), [month(1993, 7)])

    def test_contiguity_is_not_required_only_the_count(self):
        # select_training() counts eligible rows; it never demands adjacency.
        # 150 eligible rows cannot reach 240 however they are spread out.
        self.assertEqual(attainable_forecasts(self.months(300)[::2]), [])
        # Every other month over 600 calendar months gives 300 eligible rows.
        # The 240th sits at calendar offset 478, and a forecast needs targets by
        # t-1, so the first eligible t is offset 486 (478 + 6 + 1, rounded up to
        # the next eligible even offset) and the last is offset 598.
        scattered = self.months(600)[::2]
        self.assertEqual(len(scattered), 300)
        forecasts = attainable_forecasts(scattered)
        self.assertEqual(forecasts[0], month(1973 + 486 // 12, 486 % 12 + 1))
        self.assertEqual(forecasts[-1], scattered[-1])
        self.assertEqual(len(forecasts), (598 - 486) // 2 + 1)


class ForecastHorizonUnitTests(unittest.TestCase):
    """Regression locks for the calendar-month vs eligible-observation distinction.

    docs/forecast_protocol.md is explicit on both halves of the rule:
      - `target_month` is "exactly six calendar months later ... looked up by that
        date rather than by row position or a fixed number of days";
      - "At least 240 actual eligible months are required. Eligibility is
        calculated using dates, not by removing a fixed number of rows."
    So the horizon is CALENDAR months and the 240 is a COUNT of rows. Treating the
    horizon as six eligible observations is the known failure mode these tests pin.
    """

    def months(self, count, start=date(1973, 1, 1)):
        return [add_months(start, i) for i in range(count)]

    def assert_no_leakage(self, eligible, forecasts):
        """Each forecast needs >=240 strictly-earlier rows observable by t-1."""
        for t in forecasts:
            latest = add_months(t, -TRAINING_TARGET_LAG_MONTHS)
            training = [i for i in eligible if add_months(i, HORIZON_MONTHS) <= latest]
            self.assertGreaterEqual(len(training), MIN_TRAINING_MONTHS, t)
            self.assertTrue(all(i < t for i in training), t)
            self.assertTrue(all(add_months(i, HORIZON_MONTHS) <= latest for i in training), t)

    def test_consecutive_months_earliest_forecast_and_boundary(self):
        # Contiguous from 1973-01: row i is eligible when i+6 <= t-1, so the count
        # at calendar offset k is k-6, which first reaches 240 at k=246 -> 1993-07.
        self.assertEqual(attainable_forecasts(self.months(246)), [])
        self.assertEqual(attainable_forecasts(self.months(247)), [month(1993, 7)])
        forecasts = attainable_forecasts(self.months(260))
        self.assertEqual(forecasts[0], month(1993, 7))
        self.assertNotIn(month(1993, 6), forecasts)
        self.assertEqual(len(forecasts), 260 - 246)
        self.assert_no_leakage(self.months(260), forecasts)

    def test_alternating_months_use_calendar_not_observation_horizon(self):
        scattered = self.months(600)[::2]
        forecasts = attainable_forecasts(scattered)
        # Six CALENDAR months spans only three alternating rows. Eligible count at
        # even offset k is (k-6)/2, first reaching 240 at k=486 -> 2013-07.
        self.assertEqual(forecasts[0], month(2013, 7))
        self.assertEqual(forecasts[-1], month(2022, 11))
        self.assertEqual(len(forecasts), 57)
        # 54 is what a six-ELIGIBLE-OBSERVATION horizon yields (12 calendar months
        # when months alternate), which the protocol does not permit.
        self.assertNotEqual(len(forecasts), 54)
        self.assertNotIn(month(2013, 5), forecasts)
        self.assert_no_leakage(scattered, forecasts)

    def test_irregular_gaps_are_counted_by_date(self):
        irregular = sorted(set(self.months(300)[::2] + self.months(400)[300:]))
        self.assertEqual(len(irregular), 250)
        forecasts = attainable_forecasts(irregular)
        self.assertEqual(forecasts[0], month(2006, 1))
        self.assertEqual(len(forecasts), 4)
        self.assert_no_leakage(irregular, forecasts)

    def test_target_availability_boundary_is_inclusive_at_t_minus_one(self):
        # 240 rows whose targets land exactly on t-1 are eligible; shifting the
        # forecast one month earlier puts the 240th target at t and breaks it.
        eligible = self.months(247)
        boundary = month(1993, 7)
        latest = add_months(boundary, -TRAINING_TARGET_LAG_MONTHS)
        observable = [i for i in eligible if add_months(i, HORIZON_MONTHS) <= latest]
        self.assertEqual(len(observable), 240)
        self.assertEqual(add_months(observable[-1], HORIZON_MONTHS), latest)
        self.assertIn(boundary, attainable_forecasts(eligible))
        earlier = add_months(boundary, -1)
        self.assertEqual(len([i for i in eligible
                              if add_months(i, HORIZON_MONTHS)
                              <= add_months(earlier, -TRAINING_TARGET_LAG_MONTHS)]), 239)

    def test_matches_model_a_select_training_on_identical_dates(self):
        """The readiness rule must not drift from the baseline's own selector."""
        config = BaselineConfig()
        eligible = self.months(300)[::2]
        rows = [Observation(m, add_months(m, HORIZON_MONTHS), 1.0, 0) for m in eligible]
        forecasts = set(attainable_forecasts(eligible))
        for m in eligible:
            try:
                training = select_training(rows, m, config)
            except BaselineError as exc:
                self.assertEqual(exc.code, 'insufficient_training')
                self.assertNotIn(m, forecasts)
            else:
                self.assertIn(m, forecasts)
                self.assertGreaterEqual(len(training), MIN_TRAINING_MONTHS)

    def test_reproduces_documented_model_a_first_forecast_month(self):
        # forecast_protocol.md records October 1973 as the first eligible predictor
        # month for training_start 1953-04-01 under the same rule.
        eligible = self.months(400, start=date(1953, 4, 1))
        self.assertEqual(attainable_forecasts(eligible)[0], month(1973, 10))


class ScenarioMembershipTests(unittest.TestCase):
    def record(self, song_id, passed, reasons):
        return dict(song_id=song_id, automated_screen_pass=passed, technical_review_reasons=reasons)

    def test_only_credit_and_version_flags_are_reviewable(self):
        rows = [self.record('a', 'True', ''),
                self.record('b', 'False', 'upstream_featured_credit_removed_review'),
                self.record('c', 'False', 'version_specific_recording_review;upstream_featured_credit_removed_review'),
                self.record('d', 'False', 'non_english_likely'),
                self.record('e', 'False', 'non_english_likely;version_specific_recording_review')]
        result = scenario_membership(rows)
        self.assertEqual([k for k, v in result.items() if v['unflagged']], ['a'])
        self.assertEqual([k for k, v in result.items() if v['reviewable_only']], ['b', 'c'])
        self.assertEqual([k for k, v in result.items() if v['unflagged_plus_reviewable']], ['a', 'b', 'c'])

    def test_inconsistent_screen_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_membership([self.record('a', 'True', 'non_english_likely')])


if __name__ == '__main__':
    unittest.main()

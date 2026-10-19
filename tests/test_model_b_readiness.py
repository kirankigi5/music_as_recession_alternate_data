"""Synthetic tests for the readiness measurement; no lyric text or corpus required."""
import unittest
from datetime import date

from scripts.measure_model_b_readiness import (attainable_forecasts, longest_run,
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

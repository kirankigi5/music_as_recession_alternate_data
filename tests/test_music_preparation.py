"""Eligibility and timing regressions; all text and review evidence are synthetic."""
import json
import unittest
from datetime import date
from pathlib import Path

from music_recession.baseline import Observation
from music_recession.economic import add_months
from music_recession.music_preparation import (
    EMOTIONS, comparison_plan, month_range, prepare_months, reference_id,
    select_mapping,
)

ROOT = Path(__file__).resolve().parents[1]


class MusicPreparationTests(unittest.TestCase):
    def setUp(self):
        self.song = dict(song_id='synthetic', song='Synthetic Song', artist='Fixture', flags='')
        self.ref = dict(source='osf', source_path='data/raw/lyrics/fixture.csv',
            member='', record_number='1', content_sha256='synthetic-hash', flags='',
            method='normalized_title_and_full_credit+chart_slot', characters='100',
            source_chart_date='1973-01-06', source_chart_rank='1',
            source_title='Synthetic Song', source_artist='Fixture', source_url='')
        self.policy = json.loads((ROOT/'config/music_preparation.json').read_text())

    def review(self, language='en'):
        return dict(song_id='synthetic', source_reference_id=reference_id(self.ref),
            content_sha256=self.ref['content_sha256'], reviewer='Synthetic test reviewer',
            recording_evidence='Synthetic identity evidence', language=language,
            language_evidence='Synthetic language evidence',
            permission_evidence='Synthetic permission evidence', release_date='1972-12-01',
            release_evidence='Synthetic release evidence')

    def test_available_text_never_implies_eligibility(self):
        row = select_mapping(self.song, [self.ref])
        self.assertEqual(row['mapping_status'], 'provisional_selected')
        self.assertFalse(row['nlp_ready'])
        self.assertFalse(row['recording_verified'])
        self.assertIn('permission_evidence_missing', row['readiness_blockers'])

    def test_stale_or_incomplete_review_rejected(self):
        for change in ({'content_sha256': 'different'}, {'permission_evidence': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_mapping(self.song, [self.ref], review=dict(self.review(), **change))

    def test_reviewed_english_only(self):
        self.assertTrue(select_mapping(self.song, [self.ref], review=self.review())['nlp_ready'])
        for language in ('es', 'en-es', 'unknown'):
            self.assertFalse(select_mapping(self.song, [self.ref],
                review=self.review(language))['nlp_ready'])

    def test_year_end_and_recovery_sources_not_automatically_admitted(self):
        for source in ('kaggle', 'lydia', 'tmelo', 'LRCLIB', 'NetEase'):
            with self.subTest(source=source):
                row = select_mapping(self.song, [dict(self.ref, source=source)])
                self.assertEqual(row['mapping_status'], 'excluded')
                self.assertFalse(row['nlp_ready'])

    def test_ambiguous_recordings_and_special_entries_excluded(self):
        for flag in ('instrumental_review', 'compound_title_review',
                     'concurrent_recordings_same_identity'):
            row = select_mapping(dict(self.song, flags=flag), [self.ref])
            self.assertEqual(row['mapping_status'], 'excluded')
        conflicting = dict(self.ref, record_number='2', content_sha256='other-hash')
        self.assertEqual(select_mapping(self.song, [self.ref, conflicting])['mapping_status'], 'excluded')
        self.assertFalse(select_mapping(self.song, [self.ref], alias_excluded=True)['nlp_ready'])

    def test_month_weighting_future_rows_and_missing_emotions(self):
        mapping = [select_mapping(self.song, [self.ref])]
        charts = [dict(date=d, song='Synthetic Song', artist='Fixture', this_week=str(rank))
                  for d, rank in [('1973-01-06', 1), ('1973-01-13', 2), ('1973-02-03', 1)]]
        econ = {date(1973, 1, 1): Observation(date(1973, 1, 1), date(1973, 7, 1), 1., 0)}
        contributions, months = prepare_months(charts, mapping, econ, self.policy)
        january = months[0]
        self.assertEqual(january['total_weight'], 199)
        self.assertEqual(january['provisional_weight'], 199)
        self.assertEqual(january['nlp_ready_weight'], 0)
        self.assertEqual(january['target_month'], '1973-07-01')
        self.assertTrue(all(january[e] == '' for e in EMOTIONS))
        self.assertFalse(january['preparation_eligible'])
        self.assertEqual((contributions, months), prepare_months(charts, mapping, econ, self.policy))
        delayed = dict(self.policy, chart_publication_delay_days=60)
        self.assertEqual(prepare_months(charts, mapping, econ, delayed)[1][0]['known_weight'], 0)
        with self.assertRaises(ValueError):
            prepare_months(charts + [charts[0]], mapping, econ, self.policy)

    def test_240_months_calendar_alignment_and_no_future_labels(self):
        months = [dict(predictor_month=t.isoformat(), target_month=add_months(t, 6).isoformat(),
            forecast_cutoff=add_months(t, 1).replace(day=15).isoformat(),
            preparation_eligible=True, target_t=0)
            for t in month_range(date(1973, 1, 1), date(2023, 12, 1))]
        plan = comparison_plan(months)
        first = next(r for r in plan if r['common_preparation_eligible'])
        self.assertEqual(first['predictor_month'], '1993-07-01')
        self.assertEqual(first['target_month'], '1994-01-01')
        self.assertEqual(first['training_count'], 240)
        self.assertEqual(first['training_months'].split(';')[-1], '1992-12-01')
        self.assertEqual(first['latest_allowed_training_target'], '1993-06-01')
        changed = [dict(m, target_t=1-m['target_t']) if m['target_month'] > '1993-06-01' else m
                   for m in months]
        self.assertEqual(first, comparison_plan(changed)[246])
        for m in months:
            m['preparation_eligible'] = False
        self.assertFalse(any(r['common_preparation_eligible'] for r in comparison_plan(months)))


if __name__ == '__main__':
    unittest.main()

"""Tests for the explicitly labelled exploratory course policy.

The point of these tests is that the exploratory path must NOT be able to claim
verified evidence: it admits automatically screened records while asserting
recording_verified is False and no release date exists. The strict path must be
unaffected.
"""
import json
import unittest
from pathlib import Path

from music_recession.emotions import (EXPLORATORY_BASIS, require_eligible,
                                      require_exploratory_eligible)
from music_recession.baseline import BaselineConfig
from music_recession.music_preparation import (EXPLORATORY_MODE, select_mapping,
                                               validate_policy)


def song(song_id='s1'):
    return dict(song_id=song_id, song='Synthetic Song', artist='Synthetic Artist', flags='')


def reference(**overrides):
    row = dict(source='osf', source_path='data/raw/lyrics/osf_lyrics.csv', member='',
               record_number='7', content_sha256='a' * 64, source_chart_date='1990-03-10',
               source_chart_rank='12', source_title='Synthetic Song',
               source_artist='Synthetic Artist', source_url='https://example.invalid',
               flags='', method='title+artist+chart_slot', characters='900')
    row.update(overrides)
    return row


def screen(**overrides):
    row = dict(song_id='s1', automated_screen_pass='True',
               language_status='english_automated_consistent')
    row.update(overrides)
    return row


class PolicyValidationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = BaselineConfig.load(Path('config/baseline.json'))

    def test_shipped_exploratory_config_validates(self):
        policy = json.loads(Path('config/music_preparation_exploratory.json').read_text())
        self.assertEqual(policy['analysis_mode'], EXPLORATORY_MODE)
        self.assertEqual(policy['minimum_ready_weight_coverage'], 0.70)
        validate_policy(policy, self.baseline)

    def test_shipped_strict_config_still_validates(self):
        policy = json.loads(Path('config/music_preparation.json').read_text())
        self.assertNotIn('analysis_mode', policy)
        self.assertEqual(policy['minimum_ready_weight_coverage'], 0.9)
        validate_policy(policy, self.baseline)

    def test_exploratory_config_cannot_claim_verified_evidence(self):
        policy = json.loads(Path('config/music_preparation_exploratory.json').read_text())
        policy['recording_and_permission_evidence'] = 'required_for_nlp'
        with self.assertRaises(ValueError):
            validate_policy(policy, self.baseline)

    def test_sensitivity_thresholds_are_accepted(self):
        policy = json.loads(Path('config/music_preparation_exploratory.json').read_text())
        for threshold in (0.70, 0.75, 0.80):
            policy['minimum_ready_weight_coverage'] = threshold
            validate_policy(policy, self.baseline)


class ExploratoryMappingTests(unittest.TestCase):
    def test_screened_record_is_eligible_but_never_verified(self):
        row = select_mapping(song(), [reference()], False, None, screen())
        self.assertEqual(row['mapping_status'], 'provisional_selected')
        self.assertTrue(row['nlp_ready'])
        self.assertIs(row['recording_verified'], False)
        self.assertEqual(row['release_date'], '')
        self.assertEqual(row['language'], 'en')
        self.assertEqual(row['eligibility_basis'], EXPLORATORY_BASIS)
        self.assertIn('recording_not_independently_verified', row['readiness_blockers'])
        self.assertIn('release_date_not_verified', row['readiness_blockers'])
        self.assertIn('instructor_guidance', row['permission_evidence'])
        self.assertIn('not_a_license', row['permission_evidence'])

    def test_failed_automated_screen_is_not_eligible(self):
        row = select_mapping(song(), [reference()], False, None,
                             screen(automated_screen_pass='False'))
        self.assertFalse(row['nlp_ready'])
        self.assertEqual(row['readiness_blockers'], 'exploratory_automated_screen_not_passed')

    def test_non_english_screen_is_not_eligible(self):
        row = select_mapping(song(), [reference()], False, None,
                             screen(language_status='non_english_likely'))
        self.assertFalse(row['nlp_ready'])
        self.assertIn('language_screen', row['readiness_blockers'])

    def test_excluded_source_stays_excluded_under_exploratory_policy(self):
        # A flagged source reference must not become eligible just because the
        # candidate passed the separate language/risk screen.
        row = select_mapping(song(), [reference(flags='error_page')], False, None, screen())
        self.assertEqual(row['mapping_status'], 'excluded')
        self.assertFalse(row['nlp_ready'])

    def test_strict_path_is_unchanged_without_review_or_screen(self):
        row = select_mapping(song(), [reference()])
        self.assertEqual(row['mapping_status'], 'provisional_selected')
        self.assertFalse(row['nlp_ready'])
        self.assertIs(row['recording_verified'], False)
        self.assertEqual(row['eligibility_basis'], '')

    def test_review_and_screen_together_are_refused(self):
        with self.assertRaises(ValueError):
            select_mapping(song(), [reference()], False, {'song_id': 's1'}, screen())


class ExploratoryGateTests(unittest.TestCase):
    def eligible_row(self, **overrides):
        row = select_mapping(song(), [reference()], False, None, screen())
        row.update(overrides)
        return row

    def test_gate_accepts_a_properly_annotated_exploratory_row(self):
        require_exploratory_eligible(self.eligible_row())
        require_eligible(self.eligible_row())        # dispatches on eligibility_basis

    def test_gate_refuses_a_row_claiming_a_verified_recording(self):
        with self.assertRaises(ValueError):
            require_exploratory_eligible(self.eligible_row(recording_verified=True))
        with self.assertRaises(ValueError):
            require_exploratory_eligible(self.eligible_row(recording_verified='True'))

    def test_gate_refuses_a_row_asserting_a_release_date(self):
        with self.assertRaises(ValueError):
            require_exploratory_eligible(self.eligible_row(release_date='1990-01-01'))

    def test_gate_refuses_missing_evidence_annotations(self):
        for field in ('reviewer', 'recording_evidence', 'language_evidence',
                      'permission_evidence', 'release_evidence', 'content_sha256'):
            with self.assertRaises(ValueError):
                require_exploratory_eligible(self.eligible_row(**{field: ''}))

    def test_gate_refuses_non_english_or_non_osf_rows(self):
        for field, value in (('language', 'es'), ('source', 'kaggle'),
                             ('mapping_status', 'excluded'),
                             ('exclusion_reasons', 'no_clean_weekly_source')):
            with self.assertRaises(ValueError):
                require_exploratory_eligible(self.eligible_row(**{field: value}))

    def test_strict_gate_still_refuses_an_unverified_row(self):
        row = self.eligible_row()
        row['eligibility_basis'] = ''      # pretend it is a strict row
        with self.assertRaises(ValueError):
            require_eligible(row)


if __name__ == '__main__':
    unittest.main()

"""Offline tests for recording-metadata corroboration. No network, no lyric text.

Covers the defects found in the Step 4 pilot: featured artists carried in
artist-credit vs in the recording title, compilation/reissue dates masquerading as
original release dates, distinct songs sharing a title, bare prefix matches
conflating different songs, and incomplete metadata.
"""
import unittest

from music_recession.recording_metadata import (EVIDENCE_LEVELS, assess, classify_title,
                                                credit_names, release_dates,
                                                release_group_kind, split_billboard_credit)


def release(date, primary='Album', secondary=(), group_id='g1'):
    return {'date': date, 'release-group': {'id': group_id, 'primary-type': primary,
                                            'secondary-types': list(secondary)}}


def recording(title, credits, releases=(), score=100, mbid='mbid-1', disambiguation=''):
    return {'id': mbid, 'score': score, 'title': title, 'disambiguation': disambiguation,
            'artist-credit': [{'name': n, 'joinphrase': j} for n, j in credits],
            'releases': list(releases)}


class CreditParsingTests(unittest.TestCase):
    def test_feature_marker_splits_primary_from_guests(self):
        self.assertEqual(split_billboard_credit('LMFAO Featuring Lauren Bennett & GoonRock'),
                         ('LMFAO', ['Lauren Bennett', 'GoonRock']))
        self.assertEqual(split_billboard_credit('Wiz Khalifa Featuring Charlie Puth'),
                         ('Wiz Khalifa', ['Charlie Puth']))

    def test_ampersand_inside_primary_credit_is_preserved(self):
        # "Macklemore & Ryan Lewis" is one act; splitting it would break the query.
        primary, guests = split_billboard_credit('Macklemore & Ryan Lewis Featuring Wanz')
        self.assertEqual(primary, 'Macklemore & Ryan Lewis')
        self.assertEqual(guests, ['Wanz'])

    def test_no_feature_marker_yields_no_guests(self):
        self.assertEqual(split_billboard_credit('Santana'), ('Santana', []))

    def test_very_short_guest_tokens_are_dropped(self):
        _, guests = split_billboard_credit('Artist Featuring JT & Reasonable Name')
        self.assertEqual(guests, ['Reasonable Name'])


class TitleClassificationTests(unittest.TestCase):
    def test_exact_and_punctuation_insensitive(self):
        self.assertEqual(classify_title('Shadowboxin’', "Shadowboxin'")[0], 'exact')
        self.assertEqual(classify_title('Somebody That I Used to Know',
                                        'Somebody That I Used To Know')[0], 'exact')

    def test_title_embedded_feature_is_recognised(self):
        kind, detail = classify_title('Old Town Road (feat. Billy Ray Cyrus)',
                                      'Old Town Road', ['Billy Ray Cyrus'])
        self.assertEqual(kind, 'featured')
        self.assertIn('Billy Ray Cyrus', detail)

    def test_version_qualifier_is_distinguished_from_feature(self):
        self.assertEqual(classify_title('Mood (remix)', 'Mood')[0], 'version')
        self.assertEqual(classify_title('Yeah! - live', 'Yeah!')[0], 'version')

    def test_prefix_of_a_different_song_is_rejected(self):
        # The Step 4 defect: "Magic" must not match "Magic Number".
        self.assertEqual(classify_title('Magic Number', 'Magic')[0], 'different')
        self.assertEqual(classify_title('Baby Got Back', 'Baby')[0], 'different')

    def test_empty_title_is_different(self):
        self.assertEqual(classify_title('', 'Anything')[0], 'different')


class ReleaseGroupTests(unittest.TestCase):
    def test_compilation_and_live_are_not_original(self):
        self.assertEqual(release_group_kind(release('2013', 'Album', ['Compilation'])),
                         'compilation')
        self.assertEqual(release_group_kind(release('1999', 'Album', ['Live'])), 'live')

    def test_plain_album_single_ep_are_original(self):
        for primary in ('Album', 'Single', 'EP', ''):
            self.assertEqual(release_group_kind(release('1999', primary)), 'original')

    def test_release_dates_separate_original_from_reissue(self):
        rec = recording('Sweet Thing', [('Rufus', '')],
                        [release('1999-11-01', 'Album', ['Compilation'], 'gc'),
                         release('1975-10-01', 'Album', (), 'go')])
        dates = release_dates(rec)
        self.assertEqual(dates['earliest_original'], '1975-10-01')
        self.assertEqual(dates['non_original'], ['1999-11-01'])

    def test_credit_names_concatenates_joinphrases(self):
        rec = recording('X', [('24kGoldn', ' feat. '), ('iann dior', '')])
        self.assertEqual(credit_names(rec), '24kGoldn feat. iann dior')


class AssessTests(unittest.TestCase):
    def test_featured_credit_in_artist_credit_corroborates(self):
        recs = [recording('Mood', [('24kGoldn', ' feat. '), ('iann dior', '')],
                          [release('2020-07-24', 'Single')])]
        out = assess('Mood', '24kGoldn Featuring iann dior', '2020-08-01', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertEqual(out['guest_evidence'], 'artist_credit')
        self.assertEqual(out['earliest_original_release'], '2020-07-24')

    def test_title_embedded_feature_corroborates(self):
        recs = [recording('Old Town Road (feat. Billy Ray Cyrus)', [('Lil Nas X', '')],
                          [release('2019-04-05', 'Single')])]
        out = assess('Old Town Road', 'Lil Nas X Featuring Billy Ray Cyrus',
                     '2019-04-13', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertEqual(out['guest_evidence'], 'recording_title')

    def test_missing_featured_credit_is_ambiguous_not_corroborated(self):
        recs = [recording('Party Rock Anthem', [('LMFAO', '')],
                          [release('2011-01-25', 'Single')])]
        out = assess('Party Rock Anthem', 'LMFAO Featuring Lauren Bennett & GoonRock',
                     '2011-02-05', recs)
        self.assertEqual(out['evidence_level'], 'ambiguous_requires_review')
        self.assertIn('featured_credit_absent_from_metadata', out['review_flags'])

    def test_reissue_only_dates_are_ambiguous_never_conflicting(self):
        # A 2014 compilation cannot contradict a 1974 chart appearance.
        recs = [recording('Earache My Eye', [('Cheech & Chong', '')],
                          [release('2014-01-01', 'Album', ['Compilation'], 'gc')])]
        out = assess('Earache My Eye', 'Cheech & Chong Featuring Alice Bowie',
                     '1974-09-28', recs)
        self.assertEqual(out['evidence_level'], 'ambiguous_requires_review')
        self.assertIn('only_non_original_release_dates', out['review_flags'])
        self.assertEqual(out['earliest_original_release'], '')

    def test_original_release_after_first_chart_conflicts(self):
        recs = [recording('Impossible Song', [('Someone', ' feat. '), ('Guest Name', '')],
                          [release('2016-05-01', 'Single')])]
        out = assess('Impossible Song', 'Someone Featuring Guest Name', '2013-03-02', recs)
        self.assertEqual(out['evidence_level'], 'conflicting_metadata')
        self.assertIn('original_release_after_first_chart', out['review_flags'])

    def test_same_title_different_recordings_are_flagged(self):
        recs = [recording('Smooth', [('Santana', '')],
                          [release('1992-06-01', 'Album', (), 'group-early')], mbid='a'),
                recording('Smooth', [('Santana', ' feat. '), ('Rob Thomas', '')],
                          [release('1999-06-29', 'Single', (), 'group-late')], mbid='b')]
        out = assess('Smooth', 'Santana Featuring Rob Thomas', '1999-07-17', recs)
        self.assertEqual(out['evidence_level'], 'ambiguous_requires_review')
        self.assertIn('divergent_original_release_years', out['review_flags'])
        self.assertIn('original_release_long_before_chart', out['review_flags'])

    def test_same_era_release_groups_do_not_block_corroboration(self):
        # A single plus an album release group is ordinary for one song and must
        # not be treated as evidence of two different songs.
        recs = [recording('No Guidance', [('Chris Brown', ' feat. '), ('Drake', '')],
                          [release('2019-06-08', 'Single', (), 'single-group'),
                           release('2019-10-04', 'Album', (), 'album-group')])]
        out = assess('No Guidance', 'Chris Brown Featuring Drake', '2019-06-22', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertIn('multiple_original_release_groups', out['review_flags'])

    def test_only_version_variants_are_flagged(self):
        recs = [recording('Yeah! (radio edit)', [('Usher', '')],
                          [release('2004-01-01', 'Single')])]
        out = assess('Yeah!', 'Usher Featuring Lil Jon & Ludacris', '2004-01-17', recs)
        self.assertEqual(out['evidence_level'], 'ambiguous_requires_review')
        self.assertIn('only_version_variant_titles', out['review_flags'])

    def test_incomplete_release_metadata_is_flagged_not_disqualifying(self):
        """Release-date evidence is a SEPARATE axis from recording identity.

        A missing release date is recorded as a flag and leaves release_verified
        untouched, but it does not by itself undermine corroboration that the
        credited recording exists. Absence of evidence is not contradiction.
        """
        recs = [recording('See You Again', [('Wiz Khalifa', ' feat. '), ('Charlie Puth', '')])]
        out = assess('See You Again', 'Wiz Khalifa Featuring Charlie Puth', '2015-04-04', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertIn('no_release_date_available', out['review_flags'])
        self.assertEqual(out['earliest_original_release'], '')
        self.assertIs(out['recording_verified'], False)

    def test_reissue_only_dates_do_not_create_a_false_conflict(self):
        # Corroborated credit, but the only date is a 2013 compilation. Must not
        # be 'conflicting_metadata' against a 2008 chart date.
        recs = [recording('Example Song', [('Primary Act', ' feat. '), ('Guest Name', '')],
                          [release('2013-10-14', 'Album', ['Compilation'], 'comp')])]
        out = assess('Example Song', 'Primary Act Featuring Guest Name', '2008-05-03', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertIn('only_non_original_release_dates', out['review_flags'])

    def test_no_candidates_is_no_match(self):
        out = assess('Baby', 'LL Cool J Featuring The-Dream', '2008-09-20', [])
        self.assertEqual(out['evidence_level'], 'no_match')
        self.assertEqual(out['review_flags'], 'no_candidate')
        self.assertEqual(out['candidate_count'], 0)

    def test_different_song_prefix_yields_no_match(self):
        recs = [recording('Magic Number', [('B.o.B', '')], [release('1979-01-01', 'Album')])]
        out = assess('Magic', 'B.o.B Featuring Rivers Cuomo', '2010-04-24', recs)
        self.assertEqual(out['evidence_level'], 'no_match')

    def test_subset_artist_query_score_still_corroborates(self):
        """Regression lock for the score-floor calibration defect.

        A reduced primary-artist query scores ~80-84 against a recording carrying
        the full featured credit, because the query artist is only part of that
        credit. A floor near 100 silently discarded all such evidence.
        """
        recs = [recording('Party Rock Anthem',
                          [('LMFAO', ', '), ('Lauren Bennett', ' & '), ('GoonRock', '')],
                          [release('2011-01-25', 'Single')], score=84)]
        out = assess('Party Rock Anthem', 'LMFAO Featuring Lauren Bennett & GoonRock',
                     '2011-02-05', recs)
        self.assertEqual(out['evidence_level'], 'metadata_corroborated')
        self.assertEqual(out['guest_evidence'], 'artist_credit')

    def test_low_score_results_are_ignored(self):
        recs = [recording('Mood', [('24kGoldn', ' feat. '), ('iann dior', '')],
                          [release('2020-07-24', 'Single')], score=40)]
        out = assess('Mood', '24kGoldn Featuring iann dior', '2020-08-01', recs)
        self.assertEqual(out['evidence_level'], 'no_match')

    def test_recording_verified_is_never_set_at_any_level(self):
        cases = [
            ('Mood', '24kGoldn Featuring iann dior', '2020-08-01',
             [recording('Mood', [('24kGoldn', ' feat. '), ('iann dior', '')],
                        [release('2020-07-24', 'Single')])]),
            ('Party Rock Anthem', 'LMFAO Featuring Lauren Bennett', '2011-02-05',
             [recording('Party Rock Anthem', [('LMFAO', '')], [release('2011-01-25')])]),
            ('Impossible Song', 'Someone Featuring Guest Name', '2013-03-02',
             [recording('Impossible Song', [('Someone', ' feat. '), ('Guest Name', '')],
                        [release('2016-05-01', 'Single')])]),
            ('Baby', 'LL Cool J Featuring The-Dream', '2008-09-20', []),
        ]
        seen = set()
        for title, artist, chart, recs in cases:
            out = assess(title, artist, chart, recs)
            self.assertIn(out['evidence_level'], EVIDENCE_LEVELS)
            self.assertIs(out['recording_verified'], False)
            seen.add(out['evidence_level'])
        self.assertEqual(seen, set(EVIDENCE_LEVELS))


if __name__ == '__main__':
    unittest.main()

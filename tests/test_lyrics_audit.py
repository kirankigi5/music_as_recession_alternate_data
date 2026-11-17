"""Synthetic original text only; no copyrighted lyric fixtures."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from music_recession.lyrics_audit import (audit, candidate_key, eligible_observation,
    file_hash, identity, load_supplement, lyric_flags, normalize, sha, write_csv)

TEXT = 'Synthetic test words created solely to exercise a local source record safely.'


class MatchingTests(unittest.TestCase):
    def test_punctuation_unicode(self):
        self.assertEqual(normalize('It’s   A SONG!'), normalize("It's a song"))
        self.assertNotEqual(normalize('Canción'), normalize('Cancion'))

    def test_features_preserved(self):
        self.assertEqual(candidate_key('A', 'X feat. Y'), candidate_key('A', 'X Featuring Y'))
        self.assertNotEqual(candidate_key('A', 'X feat. Y'), candidate_key('A', 'X'))

    def test_versions_covers_and_collaborators_not_collapsed(self):
        for a,b in [('Song (Remix)', 'Song'), ('Song Live', 'Song'), ('Song Part II', 'Song')]:
            self.assertNotEqual(candidate_key(a, 'X'), candidate_key(b, 'X'))
        self.assertNotEqual(candidate_key('Song', 'X'), candidate_key('Song', 'Y'))
        self.assertNotEqual(candidate_key('Song', 'X & Y'), candidate_key('Song', 'X'))

    def test_missing_instrumental_and_bad_text(self):
        self.assertIn('missing_text', lyric_flags('NaN'))
        self.assertIn('instrumental_review', lyric_flags('[Instrumental]'))
        self.assertIn('error_or_html', lyric_flags('<html>404 not found</html>'))
        self.assertIn('non_latin_script_review', lyric_flags(TEXT+' 漢字'))
        self.assertIn('scraper_boilerplate_review', lyric_flags(TEXT+' 10Embed'))

    def test_cutoff_month_membership_and_delay(self):
        self.assertTrue(eligible_observation('2023-12-30', '2023-12', '2024-01-15'))
        self.assertFalse(eligible_observation('2023-12-30', '2023-12', '2024-01-16'))
        self.assertFalse(eligible_observation('2024-01-06', '2023-12'))
        self.assertFalse(eligible_observation('2023-11-30', '2023-12'))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for p in ['data/raw/lyrics/kaggle', 'data/raw/lyrics/lydia', 'data/raw/lyrics/tmelo_uea', 'data/metadata', 'config']:
            (self.root/p).mkdir(parents=True, exist_ok=True)
        self.chart = [dict(date='1973-01-06', song='Test', artist='Artist', this_week='1'),
                      dict(date='1973-01-13', song='Test', artist='Artist', this_week='2'),
                      dict(date='1973-01-06', song='Test (Remix)', artist='Artist', this_week='3')]
        write_csv(self.root/'billboard_hot100_all.csv', self.chart)
        write_csv(self.root/'data/metadata/song_lyrics_inventory.csv', [dict(song_id='id',song='Test',artist='Artist',lyrics_status='matched_osf_unverified',lyrics_source='OSF:2k7ut')])
        self.osf = [dict(Date=r['date'], Rank=r['this_week'], **{'Song Title':r['song'], 'Artist':r['artist'], 'lyrics_raw':TEXT}) for r in self.chart[:2]]
        write_csv(self.root/'data/raw/lyrics/osf_lyrics.csv', self.osf)
        with zipfile.ZipFile(self.root/'data/raw/lyrics/kaggle/top100.zip', 'w') as z:
            z.writestr('all_songs_data.csv', 'Song Title,Artist,Lyrics,Year,Song URL,Release Date\n')
        with zipfile.ZipFile(self.root/'data/raw/lyrics/lydia/lydia.zip', 'w'):
            pass

    def run_audit(self, name='out'):
        return audit(self.root, self.root/name)

    def test_offline_lineage_no_lyric_export_and_weighting(self):
        original = file_hash(self.root/'data/metadata/song_lyrics_inventory.csv')
        s=self.run_audit()
        self.assertEqual(s['songs'],2)
        self.assertEqual(s['provisional_songs'],1)
        self.assertEqual(s['verified_songs'],0)
        self.assertEqual(s['covered_weight'],199)
        self.assertEqual(s['popularity_weight'],297)
        with (self.root/'out/source_references.csv').open() as f:
            refs=list(csv.DictReader(f))
        self.assertEqual(len(refs),1)
        self.assertEqual(refs[0]['source_occurrences'],'2')
        self.assertEqual(refs[0]['record_number'],'1')
        self.assertEqual(refs[0]['content_sha256'],sha(TEXT.encode()))
        for p in (self.root/'out').iterdir():
            self.assertNotIn(TEXT,p.read_text())
        self.assertEqual(original,file_hash(self.root/'data/metadata/song_lyrics_inventory.csv'))
        with self.assertRaises(ValueError): self.run_audit()

    def test_deterministic(self):
        self.run_audit('a');self.run_audit('b')
        for p in (self.root/'a').iterdir():
            self.assertEqual(p.read_bytes(),(self.root/'b'/p.name).read_bytes())

    def test_duplicate_slots_rejected(self):
        write_csv(self.root/'billboard_hot100_all.csv',self.chart+[self.chart[0]])
        with self.assertRaises(ValueError): self.run_audit()

    def test_concurrent_recordings_preserved(self):
        write_csv(self.root/'billboard_hot100_all.csv',self.chart+[dict(date='1973-01-06',song='Test',artist='Artist',this_week='4')])
        s=self.run_audit()
        self.assertEqual(s['chart_entries'],4)
        self.assertEqual(s['concurrent_identity_count'],1)
        self.assertEqual(s['song_issue_counts']['concurrent_recordings_same_identity'],1)

    def test_future_source_records_ignored(self):
        write_csv(self.root/'data/raw/lyrics/osf_lyrics.csv',[dict(self.osf[0], Date='2024-01-06')])
        s=self.run_audit()
        self.assertEqual(s['provisional_songs'],0)
        self.assertEqual(s['prior_candidates_without_usable_reference'],1)

    def test_wrong_chart_slot_is_not_coverage(self):
        write_csv(self.root/'data/raw/lyrics/osf_lyrics.csv',[dict(self.osf[0], Rank='3')])
        self.assertEqual(self.run_audit()['provisional_songs'],0)

    def test_broken_mapping_flagged(self):
        write_csv(self.root/'config/lydia_matched_song_sources.csv',[dict(song='Test',billboard_artist='Artist',lydia_file='absent.json',lydia_year='1973')])
        self.assertEqual(self.run_audit()['mapping_issue_count'],1)
        self.assertIn('source_record_not_resolved',(self.root/'out/mapping_issues.csv').read_text())

    def test_mapped_hash_mismatch(self):
        p=self.root/'data/raw/lyrics/tmelo_uea/billboard-year-end-1973-lyrics.csv'
        write_csv(p,[dict(Rank='1',Musica='Test',Artista='Artist',Letra=TEXT)])
        write_csv(self.root/'config/tmelo_matched_song_sources.csv',[dict(song='Test (Remix)',billboard_artist='Artist',source_csv=p.name,source_record_number=1,source_title='Test',source_artist='Artist',lyrics_sha256='wrong',source_year='1973')])
        s=self.run_audit()
        self.assertEqual(s['provisional_songs'],1)
        self.assertIn('content_hash_mismatch',(self.root/'out/mapping_issues.csv').read_text())

    def test_trimmed_hash_convention_is_explicit(self):
        p=self.root/'data/raw/lyrics/tmelo_uea/billboard-year-end-1973-lyrics.csv'
        write_csv(p,[dict(Rank='1',Musica='Test',Artista='Artist',Letra='  '+TEXT+'\n')])
        write_csv(self.root/'config/tmelo_matched_song_sources.csv',[dict(song='Test',billboard_artist='Artist',source_csv=p.name,source_record_number=1,source_title='Test',source_artist='Artist',lyrics_sha256=sha(TEXT.encode()),source_year='1973')])
        self.run_audit()
        self.assertIn('hash_matches_trimmed_text_only',(self.root/'out/mapping_issues.csv').read_text())
        self.assertIn(sha(('  '+TEXT+'\n').encode()),(self.root/'out/source_references.csv').read_text())

    def test_malformed_file_quarantined_without_exporting_fragments(self):
        p=self.root/'data/raw/lyrics/tmelo_uea/billboard-year-end-1973-lyrics.csv'
        p.write_text('Rank,Musica,Artista,Letra\n1,Test,Artist,placeholder\nBROKEN_FRAGMENT\n')
        s=self.run_audit()
        self.assertEqual(s['source_issue_count'],2)
        self.assertIn('quarantined_malformed_csv',(self.root/'out/source_issues.csv').read_text())
        for p in (self.root/'out').iterdir():
            self.assertNotIn('BROKEN_FRAGMENT',p.read_text())

    def test_large_fields_are_reviewed_not_parser_failures(self):
        write_csv(self.root/'data/raw/lyrics/osf_lyrics.csv',[dict(self.osf[0],lyrics_raw='Original fixture '*10000)])
        s=self.run_audit()
        self.assertEqual(s['provisional_songs'],0)
        self.assertEqual(s['song_issue_counts']['unusually_long_text_review'],1)

    def test_compound_components_are_never_verified(self):
        write_csv(self.root/'billboard_hot100_all.csv',[dict(date='1973-01-06',song='First/Second',artist='Artist',this_week='1')])
        with zipfile.ZipFile(self.root/'data/raw/lyrics/lydia/lydia.zip','w') as z:
            member='Lyrics Project Data/Lyrics/1973/first.json'
            z.writestr(member,json.dumps(dict(title='First',artist='Artist',lyrics=TEXT)))
        write_csv(self.root/'config/lydia_combined_singles_sources.csv',[dict(song='First/Second',billboard_artist='Artist',lydia_file=member,lydia_year='1973',component_title='First',component_weight='0.5')])
        s=self.run_audit()
        self.assertEqual(s['verified_songs'],0)
        self.assertEqual(s['song_issue_counts']['compound_title_review'],1)

    def test_supplement_permissions_hash_and_path(self):
        p=self.root/'data/raw/lyrics/songdata.csv'
        write_csv(p,[dict(artist='Artist',song='Test',text=TEXT)])
        manifest=self.root/'supplement.json'
        m=dict(path='data/raw/lyrics/songdata.csv',sha256=file_hash(p))
        manifest.write_text(json.dumps(m))
        with self.assertRaises(ValueError): list(load_supplement(self.root,manifest))
        m.update(authorized_local_analysis=True,permission_evidence='Original synthetic fixture authored for tests',reviewer='test')
        manifest.write_text(json.dumps(m))
        self.assertEqual(len(list(load_supplement(self.root,manifest))),1)
        m['sha256']='wrong';manifest.write_text(json.dumps(m))
        with self.assertRaises(ValueError): list(load_supplement(self.root,manifest))
        m['path']='../outside.csv';manifest.write_text(json.dumps(m))
        with self.assertRaises(ValueError): list(load_supplement(self.root,manifest))


if __name__=='__main__': unittest.main()

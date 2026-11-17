"""Synthetic evidence screening tests; no lyric fixture or external model required."""
import unittest

from scripts.audit_osf_evidence import instrumental_status, screen_language, version_terms


class OSFEvidenceTests(unittest.TestCase):
    def test_automated_english_is_not_verification(self):
        result=screen_language('synthetic English fixture '*50,lambda text:('en',.999))
        self.assertEqual(result['language_status'],'english_automated_consistent')
        self.assertFalse(result['language_verified'])

    def test_non_english_block_is_not_hidden_by_whole_text(self):
        # Full document classified English, but its second complete block is not.
        text=' '.join(['english']*100+['spanish']*100)
        result=screen_language(text,lambda t:('es',.99) if t.startswith('spanish') else ('en',.99))
        self.assertEqual(result['language_status'],'mixed_or_non_english_block_review')
        self.assertEqual(result['non_english_high_confidence_words'],100)

    def test_short_tail_is_retained_and_short_text_is_uncertain(self):
        calls=[]
        def detect(text):calls.append(len(text.split()));return 'en',.99
        result=screen_language('fixture '*120,detect)
        self.assertEqual(calls,[120,120])
        self.assertEqual(result['language_word_count'],120)
        self.assertEqual(screen_language('fixture '*10,detect)['language_status'],'too_short_for_language_screen')

    def test_instrumental_markers_are_not_audio_verification(self):
        self.assertEqual(instrumental_status('[Instrumental]'),'instrumental_marker_only')
        self.assertEqual(instrumental_status(''),'no_text_cannot_determine_instrumental')
        self.assertEqual(instrumental_status('word '*100),'very_low_lexical_diversity_review')
        text='This synthetic text contains enough separate vocabulary to avoid the low diversity rule.'
        self.assertEqual(instrumental_status(text),'no_instrumental_marker_detected')

    def test_versions_preserved(self):
        self.assertEqual(version_terms('Synthetic Song (Live Remix)'), 'live;remix')
        self.assertEqual(version_terms('Synthetic Song'),'')
        self.assertEqual(version_terms('Live Your Life'),'')
        self.assertEqual(version_terms('Part of Me'),'')
        self.assertEqual(version_terms('Synthetic Song Part Two'),'part')


if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import patch

from speechprefix.govinfo import TextExtractor, _in_trump_period, _looks_like_speech, clean_transcript


class GovInfoImportTests(unittest.TestCase):
    def test_trump_presidential_period_filter(self):
        self.assertTrue(_in_trump_period("2017-01-20"))
        self.assertTrue(_in_trump_period("2020-11-01"))
        self.assertTrue(_in_trump_period("2025-01-20"))
        self.assertTrue(_in_trump_period("2018-01-20", "2017"))
        self.assertFalse(_in_trump_period("2018-01-20", "2025"))
        self.assertFalse(_in_trump_period("2022-06-01"))
        self.assertFalse(_in_trump_period("not-a-date"))

    def test_title_filter_selects_speech_like_documents(self):
        self.assertTrue(_looks_like_speech("Remarks at a National Celebration"))
        self.assertTrue(_looks_like_speech("The President's News Conference"))
        self.assertFalse(_looks_like_speech("Executive Order 14300"))

    def test_html_extractor_skips_head_and_scripts(self):
        parser = TextExtractor()
        parser.feed("<html><head><title>ignored</title></head><body><p>Welcome</p><script>secret()</script><p>friends</p></body></html>")
        self.assertEqual(" ".join("".join(parser.parts).split()), "Welcome friends")

    def test_clean_transcript_removes_only_known_headers(self):
        cleaned = clean_transcript("Administration of Donald J. Trump\nRemarks on the economy\nDCPD Number: 2026-123\nWe are here.")
        self.assertEqual(cleaned, "Remarks on the economy\nWe are here.")

    def test_env_key_not_required_at_import_time(self):
        with patch.dict("os.environ", {}, clear=True):
            from speechprefix.govinfo import _in_trump_period as importable
            self.assertTrue(importable("2018-01-01"))


if __name__ == "__main__":
    unittest.main()

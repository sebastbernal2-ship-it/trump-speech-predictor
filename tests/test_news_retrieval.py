import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from speechprefix.news_retrieval import collect_speaker_news_snapshots


RSS = b'''<?xml version="1.0"?><rss><channel>
<item><title>Trump discusses tax policy</title><link>https://news.test/tax</link><pubDate>Sun, 20 Oct 2024 12:00:00 GMT</pubDate><source>Example News</source></item>
<item><title>Same-day result</title><link>https://news.test/same-day</link><pubDate>Mon, 21 Oct 2024 08:00:00 GMT</pubDate><source>Example News</source></item>
<item><title>Outside lookback</title><link>https://news.test/old</link><pubDate>Fri, 20 Sep 2024 08:00:00 GMT</pubDate><source>Example News</source></item>
</channel></rss>'''


class NewsRetrievalTests(unittest.TestCase):
    def test_collects_only_pre_cutoff_results_and_records_query_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.jsonl"
            output = root / "snapshots.jsonl"
            source.write_text(json.dumps({"event_id": "event-1", "event_date": "2024-10-21"}) + "\n")
            with patch("speechprefix.news_retrieval.urlopen", return_value=BytesIO(RSS)) as mocked:
                result = collect_speaker_news_snapshots(
                    source, output, speaker="Donald Trump", lookback_days=30, delay_seconds=0,
                )
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["candidate_source_count"], 1)
            request = mocked.call_args.args[0]
            query = parse_qs(urlparse(request.full_url).query)["q"][0]
            self.assertEqual(query, '"Donald Trump" after:2024-09-21 before:2024-10-21')
            row = json.loads(output.read_text().strip())
            self.assertEqual(row["candidate_sources"][0]["title"], "Trump discusses tax policy")
            self.assertEqual(row["candidate_sources"][0]["published_at"], "2024-10-20T12:00:00+00:00")
            self.assertEqual(len(row["query_response_sha256"]), 64)
            self.assertTrue(row["exploratory_only"])
            self.assertNotIn("target_ideas", row)

    def test_refuses_to_overwrite_without_making_a_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.jsonl"
            output = root / "snapshots.jsonl"
            source.write_text(json.dumps({"event_id": "event-1", "event_date": "2024-10-21"}) + "\n")
            output.write_text("prior artifact\n")
            with patch("speechprefix.news_retrieval.urlopen") as mocked:
                with self.assertRaises(FileExistsError):
                    collect_speaker_news_snapshots(source, output, delay_seconds=0)
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()

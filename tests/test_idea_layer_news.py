import base64
import hashlib
import unittest
from datetime import datetime, timezone

from speechprefix.idea_layer.contracts import validate_source_evidence
from speechprefix.idea_layer.news import retrieve_google_news_evidence


class GoogleNewsEvidenceTests(unittest.TestCase):
    def _rss(self):
        return b'''<?xml version="1.0" encoding="UTF-8"?>
        <rss><channel>
          <item><title>Oil supply disruption raises prices</title><link>https://news.example/oil</link>
            <pubDate>Mon, 20 Jan 2030 14:00:00 GMT</pubDate><source>Example News</source></item>
          <item><title>Gaza and Pakistan talks</title><link>https://news.example/gaza</link>
            <pubDate>Mon, 20 Jan 2030 16:05:00 GMT</pubDate><source>Example News</source></item>
          <item><title>Old oil story</title><link>https://news.example/old</link>
            <pubDate>Mon, 01 Dec 2029 10:00:00 GMT</pubDate><source>Example News</source></item>
        </channel></rss>'''

    def test_captures_only_time_eligible_headlines_and_merges_query_hits(self):
        times = iter([
            datetime(2030, 1, 20, 15, 0, tzinfo=timezone.utc),
            datetime(2030, 1, 20, 15, 10, tzinfo=timezone.utc),
            datetime(2030, 1, 20, 15, 20, tzinfo=timezone.utc),
            datetime(2030, 1, 20, 15, 21, tzinfo=timezone.utc),
        ])
        calls = []
        def fetcher(url, timeout):
            calls.append(url)
            return self._rss()
        report = retrieve_google_news_evidence(
            ["oil", "Gaza OR Pakistan"], "2030-01-20T16:00:00Z", lookback_days=30,
            delay_seconds=0, fetcher=fetcher, clock=lambda: next(times), sleeper=lambda _: None,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["unique_headline_count"], 1)
        self.assertEqual(report["query_runs"][0]["eligible_result_count"], 1)
        run = report["query_runs"][0]
        response = base64.b64decode(run["response_body_base64"])
        self.assertEqual(hashlib.sha256(response).hexdigest(), run["response_sha256"])
        source = report["sources"][0]
        validate_source_evidence(source)
        self.assertEqual(source["availability_status"], "verified_before_cutoff")
        self.assertEqual(source["query_ids"], ["q01", "q02"])
        self.assertEqual(source["canonical_source_id"], source["source_id"].rsplit("-obs-", 1)[0])
        self.assertEqual(source["publication_timestamp_basis"], "Google News RSS pubDate; feed-reported, not independently verified with publisher")
        self.assertTrue(all(run["retrieved_at"] < "2030-01-20T16:00:00+00:00" for run in report["query_runs"]))

    def test_aborts_if_collection_crosses_cutoff(self):
        times = iter([
            datetime(2030, 1, 20, 15, 55, tzinfo=timezone.utc),
            datetime(2030, 1, 20, 16, 1, tzinfo=timezone.utc),
        ])
        with self.assertRaisesRegex(ValueError, "at or after prediction cutoff"):
            retrieve_google_news_evidence(
                ["oil"], "2030-01-20T16:00:00Z", fetcher=lambda *_: self._rss(),
                clock=lambda: next(times),
            )

    def test_requires_future_cutoff_and_unique_queries(self):
        with self.assertRaisesRegex(ValueError, "begin strictly before"):
            retrieve_google_news_evidence(
                ["oil"], "2030-01-20T16:00:00Z",
                clock=lambda: datetime(2030, 1, 20, 16, tzinfo=timezone.utc),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            retrieve_google_news_evidence(["oil", " oil "], "2030-01-20T16:00:00Z")


if __name__ == "__main__":
    unittest.main()

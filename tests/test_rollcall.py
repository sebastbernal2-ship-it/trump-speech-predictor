import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speechprefix.rollcall import (
    _normalize_url,
    collect_rollcall,
    iter_factbase_records,
    parse_factbase_page,
)


RECORD = {
    "document_id": "fixture-doc",
    "date": "2026-01-01",
    "factbase_url": "https://rollcall.com/factbase/trump/transcript/fixture-doc/",
    "record_title": "Press Gaggle: fixture",
    "source": "Factba.se",
    "record_type": "Press Gaggle",
    "media_type": "Video",
    "document": {
        "speakers": [
            {"speaker": "Donald Trump", "speaker_id": "Q22686"},
            {"speaker": "Question", "speaker_id": "Q189756"},
        ],
        "topics": [{"topic": "Politics"}],
    },
}

PAGE = """<html><body>
<div id="fixture-doc-1" class="mb-4 border-b mx-6 my-4">
  <h2>Donald Trump</h2>
  <span>00:00:00-00:00:02 (2 sec)</span>
  <div class="text-center border rounded-md">No Signal (0.034)</div>
  <div class="flex-auto text-md leading-loose">Hello &amp; welcome.</div>
</div>
<div id="fixture-doc-2" class="mb-4 border-b mx-6 my-4">
  <h2>Question</h2>
  <span>00:00:02-00:00:03 (1 sec)</span>
  <div class="text-center border rounded-md">No StressLens</div>
  <div class="flex-auto text-md leading-loose">What is your plan?</div>
</div>
</body></html>"""


class RollCallImportTests(unittest.TestCase):
    def test_page_parser_preserves_timed_speakers_source_annotations_and_metadata(self):
        event = parse_factbase_page(PAGE, RECORD, reuse_terms="permissioned use")
        self.assertEqual(event["event_id"], "fixture-doc")
        self.assertEqual(event["source_name"], "Roll Call Factba.se")
        self.assertEqual(event["reuse_terms"], "permissioned use")
        self.assertEqual(event["page_metadata"]["factbase_record"], RECORD)
        self.assertEqual(event["turns"][0]["speaker_role"], "target_speaker")
        self.assertEqual(event["turns"][0]["speaker_id"], "Q22686")
        self.assertEqual(event["turns"][0]["source_turn_id"], "fixture-doc-1")
        self.assertEqual(event["turns"][0]["text"], "Hello & welcome.")
        self.assertEqual(event["turns"][0]["annotations"][0]["score"], 0.034)
        self.assertEqual(event["turns"][1]["speaker_role"], "questioner")
        self.assertEqual(event["turns"][1]["annotations"][0]["status"], "unavailable")

    def test_page_without_timestamped_turns_is_skipped(self):
        self.assertIsNone(parse_factbase_page("<html><body>No Transcript</body></html>", RECORD,
                                             reuse_terms="permissioned use"))

    def test_untranscribed_record_is_skipped_instead_of_aborting_collection(self):
        record = {**RECORD, "record_title": "Interview: No Transcript - fixture"}
        self.assertIsNone(parse_factbase_page(PAGE, record, reuse_terms="permissioned use"))

    def test_url_spaces_are_encoded_for_fetch_and_source_attribution(self):
        url = "https://rollcall.com/factbase/trump/transcript/bitcoin 2025 conference/"
        self.assertEqual(_normalize_url(url),
                         "https://rollcall.com/factbase/trump/transcript/bitcoin%202025%20conference/")

    def test_rejects_non_rollcall_transcript_url(self):
        record = {**RECORD, "factbase_url": "https://example.org/transcript"}
        with self.assertRaisesRegex(ValueError, "unexpected Factba.se"):
            parse_factbase_page(PAGE, record, reuse_terms="permissioned use")

    def test_index_pagination_deduplicates_document_ids(self):
        payloads = [
            json.dumps({"meta": {"total_pages": 2}, "data": [RECORD]}).encode(),
            json.dumps({"meta": {"total_pages": 2}, "data": [RECORD, {**RECORD, "document_id": "second"}]}).encode(),
        ]
        with patch("speechprefix.rollcall._request", side_effect=payloads):
            rows = list(iter_factbase_records(delay=0))
        self.assertEqual([row["document_id"] for row in rows], ["fixture-doc", "second"])

    def test_collection_excludes_timestamp_anomaly_events(self):
        anomalous_page = PAGE.replace(
            "00:00:02-00:00:03 (1 sec)", "00:00:01-00:00:03 (2 sec)"
        ).replace(
            "00:00:00-00:00:02 (2 sec)", "00:00:02-00:00:03 (1 sec)"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            with patch("speechprefix.rollcall.iter_factbase_records",
                       side_effect=lambda **kwargs: iter([RECORD])), \
                 patch("speechprefix.rollcall._request", return_value=anomalous_page.encode()):
                result = collect_rollcall(output, delay=0)
            self.assertEqual(result["timestamp_anomaly_events"], 1)
            self.assertEqual(result["events_added"], 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_collection_records_permanently_unavailable_pages_and_continues(self):
        unavailable = {
            **RECORD,
            "document_id": "unavailable-doc",
            "factbase_url": "https://rollcall.com/factbase/trump/transcript/unavailable/",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            with patch("speechprefix.rollcall.iter_factbase_records",
                       side_effect=lambda **kwargs: iter([unavailable, RECORD])), \
                 patch("speechprefix.rollcall._request",
                       side_effect=[RuntimeError("HTTP 500 from fixture after retries"), PAGE.encode()]):
                result = collect_rollcall(output, delay=0)
            self.assertEqual(result["unavailable_source_page"], 1)
            self.assertEqual(result["unavailable_source_page_ids"][0]["event_id"], "unavailable-doc")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event_id"] for row in rows], ["fixture-doc"])

    def test_collection_appends_clean_events_and_resumes_without_refetching(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            with patch("speechprefix.rollcall.iter_factbase_records", side_effect=lambda **kwargs: iter([RECORD])), \
                 patch("speechprefix.rollcall._request", return_value=PAGE.encode()) as fetch:
                first = collect_rollcall(output, delay=0)
                second = collect_rollcall(output, delay=0)
            self.assertEqual(first["events_added"], 1)
            self.assertEqual(second["already_present"], 1)
            self.assertEqual(fetch.call_count, 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "fixture-doc")


if __name__ == "__main__":
    unittest.main()

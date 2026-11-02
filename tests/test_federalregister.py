import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speechprefix.federalregister import append_context, search_documents


class FederalRegisterTests(unittest.TestCase):
    def test_search_is_keyless_and_drops_future_dated_results(self):
        payload = {
            "results": [
                {"document_number": "2025-001", "publication_date": "2025-04-10", "title": "Earlier",
                 "abstract": "Relevant abstract", "html_url": "https://example.test/1", "type": "RULE",
                 "agencies": [{"name": "Example Agency"}]},
                {"document_number": "2025-002", "publication_date": "2025-04-12", "title": "Future",
                 "abstract": "Not available yet", "html_url": "https://example.test/2", "type": "RULE", "agencies": []},
            ]
        }
        seen = []

        def fake_urlopen(request, timeout):
            seen.append(request.full_url)
            return io.BytesIO(json.dumps(payload).encode())

        with patch("speechprefix.federalregister.urlopen", side_effect=fake_urlopen):
            records = search_documents("2025-04-10", "energy", limit=2, lookback_days=30)
        self.assertEqual([record["context_id"] for record in records], ["2025-001"])
        self.assertNotIn("api_key", seen[0])
        self.assertIn("conditions%5Bpublication_date%5D%5Blte%5D=2025-04-10", seen[0])

    def test_append_is_idempotent_by_document_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.jsonl"
            record = {"context_id": "2025-001", "title": "one"}
            self.assertEqual(append_context(path, [record, record]), 1)
            self.assertEqual(append_context(path, [record]), 0)
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_rejects_invalid_date(self):
        with self.assertRaisesRegex(ValueError, "ISO date"):
            search_documents("not-a-date", "topic")


if __name__ == "__main__":
    unittest.main()

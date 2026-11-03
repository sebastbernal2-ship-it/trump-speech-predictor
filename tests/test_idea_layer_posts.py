import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.idea_layer.contracts import validate_source_evidence
from speechprefix.idea_layer.posts import load_trump_post_export, normalize_trump_post_export


class TrumpPostImportTests(unittest.TestCase):
    def _post(self):
        return {
            "schema_version": 1,
            "record_type": "idea-layer-trump-post-export",
            "platform": "truth_social",
            "account": "realDonaldTrump",
            "post_id": "post-123",
            "permalink": "https://truthsocial.com/@realDonaldTrump/posts/post-123",
            "text": "Exact post text.  ",
            "published_at": "2030-01-20T12:30:00Z",
            "observed_at": "2030-01-20T12:35:00Z",
            "availability_status": "verified_before_cutoff",
            "availability_evidence": "timestamped archive capture sha256:abc123",
            "timestamp_basis": "platform created_at field",
            "reuse_terms": "archive provider terms recorded separately",
        }

    def test_normalizes_with_identity_timestamp_and_exact_text(self):
        source = normalize_trump_post_export([self._post()])[0]
        validate_source_evidence(source)
        self.assertEqual(source["source_type"], "trump_post")
        self.assertEqual(source["account"], "realDonaldTrump")
        self.assertEqual(source["post_id"], "post-123")
        self.assertEqual(source["content"], "Exact post text.  ")
        self.assertEqual(source["canonical_source_id"], "trump-post:truth_social:realDonaldTrump:post-123")

    def test_rejects_unknown_fields_and_untrustworthy_times(self):
        row = self._post()
        row["transcript_excerpt"] = "leaked outcome"
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            normalize_trump_post_export([row])
        row = self._post()
        row["observed_at"] = "2030-01-20T12:00:00Z"
        with self.assertRaisesRegex(ValueError, "precedes published_at"):
            normalize_trump_post_export([row])
        row = self._post()
        row["availability_status"] = "verified_before_cutoff"
        row["availability_evidence"] = ""
        with self.assertRaisesRegex(ValueError, "availability_evidence"):
            normalize_trump_post_export([row])

    def test_loads_jsonl_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posts.jsonl"
            path.write_text(json.dumps(self._post()) + "\n", encoding="utf-8")
            loaded = load_trump_post_export(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["source_type"], "trump_post")
        self.assertEqual(len(loaded[0]["archive_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

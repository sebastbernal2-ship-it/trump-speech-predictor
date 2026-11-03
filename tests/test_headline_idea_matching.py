import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.headline_idea_matching import (
    evaluate_headline_idea_overlap,
    extract_headline_ideas,
    extract_transcript_ideas,
    match_idea_to_turns,
)


class HeadlineIdeaMatchingTests(unittest.TestCase):
    def test_extracts_open_vocabulary_ukraine_war_idea_and_matches_paraphrased_order(self):
        ideas = extract_headline_ideas([{
            "title": "Ukrainian war escalates before peace summit",
            "published_at": "2025-05-12",
            "source_url": "https://example.test/ukraine",
            "publisher": "Example News",
        }], "2025-05-14")
        idea = next(item for item in ideas if item["idea_id"] == "ukraine war")
        turns = [{
            "turn_id": "turn-1",
            "text": "We will end the war in Ukraine."
        }]
        matches = match_idea_to_turns(idea, turns)
        self.assertEqual(matches[0]["turn_id"], "turn-1")
        self.assertIn("war in Ukraine", matches[0]["matched_text"])

    def test_mines_ukraine_war_as_a_cited_transcript_idea(self):
        ideas = extract_transcript_ideas([{
            "turn_id": "turn-ukraine",
            "text": "I will end the war in Ukraine."
        }])
        idea = next(item for item in ideas if item["idea_id"] == "ukraine war")
        self.assertEqual(idea["candidate_idea"], "war Ukraine")
        self.assertEqual(idea["supporting_turn_ids"], ["turn-ukraine"])

    def test_strips_bracketed_audience_cues_from_target_ideas(self):
        ideas = extract_transcript_ideas([{
            "turn_id": "turn-stage-cue",
            "text": "Hillary Clinton [Audience shouts Boo] will end the war in Ukraine."
        }])
        self.assertTrue(any(idea["idea_id"] == "ukraine war" for idea in ideas))
        self.assertFalse(any("audience" in idea["idea_id"] or "boo" in idea["idea_id"] for idea in ideas))

    def test_drops_generic_crowd_fillers_and_country_references(self):
        ideas = extract_transcript_ideas([{
            "turn_id": "turn-generic",
            "text": "So Karen is here. We love the United States."
        }])
        idea_ids = {idea["idea_id"] for idea in ideas}
        self.assertNotIn("karen so", idea_ids)
        self.assertNotIn("state united", idea_ids)

    def test_does_not_combine_terms_across_separate_turns(self):
        idea = {"normalized_terms": ["ukraine", "war"]}
        turns = [
            {"turn_id": "turn-1", "text": "Ukraine is a beautiful country."},
            {"turn_id": "turn-2", "text": "We must prevent another war."},
        ]
        self.assertEqual(match_idea_to_turns(idea, turns), [])

    def test_overlap_audit_joins_snapshots_and_transcripts_without_weak_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots.jsonl"
            events = root / "events.jsonl"
            output = root / "overlap.json"
            source = {
                "title": "Ukraine war threatens regional peace",
                "published_at": "2025-05-12",
                "source_url": "https://example.test/ukraine",
                "publisher": "Example News",
            }
            snapshots.write_text(json.dumps({
                "event_id": "e1", "event_date": "2025-05-14", "exploratory_only": True,
                "candidate_sources": [source],
            }) + "\n", encoding="utf-8")
            events.write_text(json.dumps({
                "event_id": "e1", "event_date": "2025-05-14", "target_speaker": "Donald Trump",
                "turns": [{"turn_id": "t1", "speaker_name": "Donald Trump", "speaker_role": "target_speaker", "text": "I will end the war in Ukraine."}],
            }) + "\n", encoding="utf-8")
            report = evaluate_headline_idea_overlap(snapshots, events, output_path=output, top_k=(50,))
            self.assertEqual(report["summary_by_top_k"]["top_50"]["events"], 1)
            self.assertTrue(any(item["idea_id"] == "ukraine war" and item["mentioned_by_target"] for item in report["events"][0]["top_ideas"]))
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                evaluate_headline_idea_overlap(snapshots, events, output_path=output, top_k=(50,))

    def test_rejects_same_day_or_later_news(self):
        with self.assertRaisesRegex(ValueError, "same-day or later"):
            extract_headline_ideas([{
                "title": "Ukraine war update", "published_at": "2025-05-14", "url": "x",
            }], "2025-05-14")


if __name__ == "__main__":
    unittest.main()

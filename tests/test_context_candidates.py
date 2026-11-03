import unittest

from speechprefix.context_candidates import extract_context_candidates


class ContextCandidateTests(unittest.TestCase):
    def test_groups_headlines_into_source_grounded_idea_candidates(self):
        sources = [
            {"title": "What to expect in Nevada on Election Day", "publisher": "PBS", "published_at": "2024-10-30", "url": "url-election"},
            {"title": "Early vote does not reliably predict results", "publisher": "Silver Bulletin", "published_at": "2024-10-29", "url": "url-vote"},
            {"title": "A fight for working-class Nevada voters", "publisher": "BBC", "published_at": "2024-10-28", "url": "url-workers"},
            {"title": "Immigration in Nevada is a nuanced issue", "publisher": "AP", "published_at": "2024-10-27", "url": "url-immigration"},
            {"title": "Biden tries to clarify garbage comment after uproar", "publisher": "BBC", "published_at": "2024-10-30", "url": "url-garbage-1"},
            {"title": "White House tries to clean up Biden's garbage comment", "publisher": "ABC", "published_at": "2024-10-30", "url": "url-garbage-2"},
        ]
        candidates = extract_context_candidates(sources, "2024-10-31")
        by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
        self.assertEqual(set(by_id), {
            "election_turnout", "working_class_economy", "immigration_border", "biden_garbage_controversy",
        })
        self.assertEqual(by_id["election_turnout"]["source_count"], 2)
        self.assertEqual(by_id["biden_garbage_controversy"]["publisher_count"], 2)
        self.assertIn("not_probability", by_id["election_turnout"]["score_type"])

    def test_generates_diplomacy_hostage_and_commerce_candidates(self):
        sources = [
            {"title": "Gulf diplomacy and Qatar partnership", "published_at": "2025-05-12", "url": "gulf"},
            {"title": "Hamas to release hostage as Gaza ceasefire talks resume", "published_at": "2025-05-12", "url": "hostage"},
            {"title": "Iran and US conclude another round of nuclear talks", "published_at": "2025-05-11", "url": "iran"},
            {"title": "Commercial investment and business agreements", "published_at": "2025-05-10", "url": "business"},
        ]
        candidates = extract_context_candidates(sources, "2025-05-14")
        self.assertTrue({
            "gulf_partnership_diplomacy", "gaza_hostages_ceasefire", "foreign_security", "business_investment",
        }.issubset({candidate["candidate_id"] for candidate in candidates}))

    def test_same_day_and_later_sources_are_rejected(self):
        for published in ("2024-10-31", "2024-11-01"):
            with self.subTest(published=published):
                with self.assertRaisesRegex(ValueError, "same-day or later"):
                    extract_context_candidates([{
                        "title": "Election turnout update", "published_at": published,
                        "url": "url",
                    }], "2024-10-31")

    def test_irrelevant_garbage_headline_does_not_create_political_candidate(self):
        candidates = extract_context_candidates([{
            "title": "City expands garbage pickup schedule", "published_at": "2024-10-30", "url": "url",
        }], "2024-10-31")
        self.assertEqual(candidates, [])

    def test_duplicate_source_is_counted_once(self):
        source = {"title": "Nevada election turnout update", "published_at": "2024-10-30", "url": "same-url"}
        candidate = extract_context_candidates([source, source], "2024-10-31")[0]
        self.assertEqual(candidate["source_count"], 1)


if __name__ == "__main__":
    unittest.main()

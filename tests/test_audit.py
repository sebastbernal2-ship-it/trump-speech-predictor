import unittest

from speechprefix.audit import classify_title, summarize_records


class AuditTests(unittest.TestCase):
    def test_classifies_title_shapes(self):
        self.assertEqual(classify_title("Inaugural Address"), "formal_address")
        self.assertEqual(classify_title("Remarks at the National Prayer Breakfast"), "event_remarks")
        self.assertEqual(classify_title("Remarks at a ceremony and an Exchange With Reporters"), "remarks_plus_press")
        self.assertEqual(classify_title("The President's News Conference"), "press_interaction")
        self.assertEqual(classify_title("Statement on a nomination"), "written_statement")

    def test_summary_counts_provisional_categories(self):
        records = [{"event": "Inaugural Address"}, {"event": "Statement on a nomination"}]
        result = summarize_records(records)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["candidates_for_manual_review"], 1)


if __name__ == "__main__":
    unittest.main()

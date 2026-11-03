import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("optional PyTorch dependency is not installed")

from speechprefix.hybrid_train import generate_hybrid, load_hybrid, train_hybrid
from speechprefix.segments import segment_record


class HybridTrainingTests(unittest.TestCase):
    def test_tiny_chronological_training_saves_loadable_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speeches.jsonl"
            text = (
                "The President. We believe in our country and our future.\n\n"
                "Q. What is the plan?\n\n"
                "The President. We will protect workers and build strong communities while we create jobs "
                "and keep our promises to the American people every day."
            )
            records = [
                {"speech_id": f"speech-{day}", "date": f"2020-01-0{day}",
                 "event": "remarks", "source": "fixture", "text": text}
                for day in (1, 2, 3)
            ]
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            checkpoint = root / "speechplan.pt"
            registry = root / "registry.jsonl"
            question = next(
                segment for segment in segment_record(records[0])
                if segment["segment_type"] == "question"
            )
            evidence = {
                "context_id": "FR-1", "published_at": "2019-12-31",
                "title": "Fixture policy", "abstract": "A dated training document.",
                "source_url": "https://example.test/FR-1",
            }
            evidence_path = root / "evidence.jsonl"
            evidence_path.write_text(json.dumps({
                "question_segment_id": question["segment_id"], "documents": [evidence]
            }) + "\n")
            result = train_hybrid(
                source, checkpoint, epochs=1, batch_size=2, stride=8,
                min_frequency=1, max_vocabulary=64, width=32, layers=2,
                heads=4, codebook_size=8, context_length=8, target_length=8,
                question_length=8, evidence_length=8, evidence_pairs_path=evidence_path,
                device_name="cpu", registry_path=registry,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(registry.is_file())
            self.assertIn("speechplan-h/format-2/", result["model_version"])
            registered = json.loads(registry.read_text(encoding="utf-8").strip())
            self.assertEqual(registered["model_version"], result["model_version"])
            self.assertEqual(len(registered["artifact"]["sha256"]), 64)
            self.assertEqual(registered["dataset"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(
                set(result["test_qa"]), {
                    "combined", "combined_without_evidence", "transcript_only", "question_only"
                }
            )
            self.assertIsInstance(
                generate_hybrid(
                    checkpoint, "We believe", question_text="What is the plan?",
                    max_new_tokens=2, device_name="cpu",
                ),
                str,
            )
            self.assertIsInstance(
                generate_hybrid(
                    checkpoint, "We believe", question_text="What is the plan?",
                    evidence_records=[evidence], as_of="2020-01-01",
                    max_new_tokens=2, device_name="cpu",
                ),
                str,
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            no_evidence = dict(saved)
            no_evidence["metadata"] = dict(saved["metadata"])
            no_evidence["metadata"]["question_answer_chunks"] = dict(
                saved["metadata"]["question_answer_chunks"]
            )
            no_evidence["metadata"]["question_answer_chunks"]["evidence_pairs_train"] = 0
            no_evidence_path = root / "no-evidence.pt"
            torch.save(no_evidence, no_evidence_path)
            with self.assertRaisesRegex(ValueError, "not trained with dated evidence"):
                generate_hybrid(
                    no_evidence_path, "We believe", question_text="What is the plan?",
                    evidence_records=[evidence], as_of="2020-01-01", max_new_tokens=2, device_name="cpu",
                )
            with self.assertRaisesRegex(ValueError, "not known before"):
                generate_hybrid(
                    checkpoint, "We believe", question_text="What is the plan?",
                    evidence_records=[dict(evidence, published_at="2020-01-02")],
                    as_of="2020-01-01", max_new_tokens=2, device_name="cpu",
                )

            legacy = dict(saved)
            legacy["format"] = 1
            legacy["config"] = dict(saved["config"])
            for key in ("question_length", "evidence_length", "question_empty_id", "evidence_empty_id"):
                legacy["config"].pop(key)
            legacy["state_dict"] = {
                key: value for key, value in saved["state_dict"].items()
                if not key.startswith((
                    "question_position.", "evidence_position.", "channel_embeddings.",
                    "question_summary_projection.", "evidence_summary_projection.",
                ))
            }
            legacy_path = root / "legacy.pt"
            torch.save(legacy, legacy_path)
            old_model, _, _, _ = load_hybrid(legacy_path, "cpu")
            self.assertFalse(old_model.question_channel_enabled)
            with self.assertRaisesRegex(FileExistsError, "versioned path"):
                train_hybrid(source, checkpoint, registry_path=registry)



if __name__ == "__main__":
    unittest.main()

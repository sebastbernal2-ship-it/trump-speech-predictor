import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("optional PyTorch dependency is not installed")

from speechprefix.explicit_plan import (
    ExplicitPlanConfig,
    ExplicitPlanPathModel,
    parse_plan_path,
    train_explicit_plan_model,
)
from speechprefix.model import tokenize
from speechprefix.speech_plans import PLAN_MARKERS, plan_path_examples
from speechprefix.hybrid_train import build_vocabulary


def fixture_event(event_id, event_date):
    return {
        "event_id": event_id,
        "event_date": event_date,
        "timestamp_anomalies": [],
        "turns": [
            {"turn_id": f"{event_id}:0", "sequence": 0, "start_seconds": 0, "end_seconds": 10,
             "speaker_name": "Trump", "speaker_role": "target_speaker", "timestamp_valid": True,
             "text": "Opening remarks."},
            {"turn_id": f"{event_id}:1", "sequence": 1, "start_seconds": 10, "end_seconds": 20,
             "speaker_name": "Trump", "speaker_role": "target_speaker", "timestamp_valid": True,
             "text": "The next idea."},
        ],
    }


def fixture_plan(event_id):
    first, second = f"{event_id}:0", f"{event_id}:1"
    return {
        "schema_version": 1,
        "event_id": event_id,
        "review_status": "reviewed",
        "reviewed_by": "fixture-reviewer",
        "themes": [{
            "id": f"{event_id}-theme",
            "summary": "economic growth",
            "turn_ids": [first, second],
            "sections": [{
                "id": f"{event_id}-section",
                "summary": "tax policy",
                "turn_ids": [first, second],
                "talking_points": [
                    {"id": f"{event_id}-point-1", "summary": "tax relief", "turn_ids": [first]},
                    {"id": f"{event_id}-point-2", "summary": "new jobs", "turn_ids": [second]},
                ],
            }],
        }],
    }


class ExplicitPlanTests(unittest.TestCase):
    def test_training_examples_use_only_turns_observed_before_the_point(self):
        event = fixture_event("event-1", "2020-01-01")
        examples = plan_path_examples(event, fixture_plan("event-1"))
        self.assertEqual(len(examples), 2)
        first, second = examples
        self.assertEqual(first["visible_turn_ids"], [])
        self.assertEqual(first["target_turn_ids"], ["event-1:0"])
        self.assertEqual(second["visible_turn_ids"], ["event-1:0"])
        self.assertEqual(second["target_turn_ids"], ["event-1:1"])
        self.assertNotIn("The next idea", second["prefix_text"])
        self.assertEqual(
            tokenize(second["target"]["text"])[0], "__theme__"
        )
        truncated = plan_path_examples(event, fixture_plan("event-1"), max_context_tokens=2)
        self.assertEqual(len(tokenize(truncated[1]["prefix_text"])), 2)

    def test_draft_annotations_cannot_become_model_training_examples(self):
        event = fixture_event("event-1", "2020-01-01")
        plan = fixture_plan("event-1")
        plan["review_status"] = "draft"
        plan["reviewed_by"] = None
        with self.assertRaisesRegex(ValueError, "only reviewed"):
            plan_path_examples(event, plan)

    def test_model_forward_and_generation_use_structured_plan_tokens(self):
        vocabulary = build_vocabulary(
            ["prefix economic growth tax policy tax relief"], 1, 64,
            required_tokens=PLAN_MARKERS,
        )
        lookup = {token: index for index, token in enumerate(vocabulary)}
        config = ExplicitPlanConfig(
            vocab_size=len(vocabulary), pad_id=0, bos_id=2, eos_id=3,
            plan_start_id=lookup["__plan__"], end_plan_id=lookup["__endplan__"],
            theme_id=lookup["__theme__"], section_id=lookup["__section__"],
            point_id=lookup["__point__"],
            context_length=8, plan_length=12, width=16, layers=1, heads=2,
        )
        model = ExplicitPlanPathModel(config)
        context = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 2]])
        target = torch.tensor([[
            lookup["__theme__"], lookup["economic"], lookup["__section__"],
            lookup["tax"], lookup["policy"], lookup["__point__"],
            lookup["tax"], lookup["relief"], lookup["__endplan__"], 0, 0, 0,
        ]])
        result = model(context, target)
        self.assertTrue(math.isfinite(float(result["loss"].detach())))
        self.assertEqual(tuple(result["logits"].shape), (1, 12, len(vocabulary)))
        generated = model.generate(context, max_new_tokens=config.plan_length)
        self.assertLessEqual(len(generated), config.plan_length)
        self.assertTrue(parse_plan_path([vocabulary[index] for index in generated])["point"])

    def test_trains_and_registers_a_separate_version_on_reviewed_fixture_plans(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.jsonl"
            plans_path = root / "plans.jsonl"
            events = [fixture_event(f"event-{day}", f"2020-01-0{day}") for day in (1, 2, 3)]
            events_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            plans_path.write_text(
                "".join(json.dumps(fixture_plan(event["event_id"])) + "\n" for event in events),
                encoding="utf-8",
            )
            output = root / "explicit-plan.pt"
            registry = root / "registry.jsonl"
            result = train_explicit_plan_model(
                events_path, plans_path, output,
                registry_path=registry, epochs=1, batch_size=2,
                context_length=16, plan_length=24, width=16, layers=1, heads=2,
                min_frequency=1, max_vocabulary=64, device_name="cpu",
            )
            self.assertTrue(output.is_file())
            self.assertIn("speechplan-explicit/plan-path-v2/", result["model_version"])
            registered = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(registered["model_version"], result["model_version"])
            self.assertEqual(registered["training"]["label_policy"], "reviewed")
            self.assertGreaterEqual(registered["metrics"]["test"]["examples"], 1)
            from speechprefix.explicit_plan import predict_plan_path

            prediction = predict_plan_path(output, "Observed speech prefix", device_name="cpu")
            self.assertEqual(set(prediction), {"model_version", "theme", "section", "point"})
            with self.assertRaisesRegex(FileExistsError, "versioned path"):
                train_explicit_plan_model(events_path, plans_path, output, registry_path=registry)

    def test_trains_separate_weak_label_version_from_real_transcript_fixture(self):
        from speechprefix.weak_plan_labels import create_weak_plan_labels

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [fixture_event(f"event-{day}", f"2020-01-0{day}") for day in (1, 2, 3)]
            events_path = root / "events.jsonl"
            events_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            records_path = root / "speeches.jsonl"
            records_path.write_text("".join(json.dumps({
                "speech_id": event["event_id"], "date": event["event_date"],
            }) + "\n" for event in events), encoding="utf-8")
            plans_path = root / "weak-plans.jsonl"
            create_weak_plan_labels(
                events_path, records_path, plans_path, section_turns=2, point_turns=1,
            )
            plans = [json.loads(line) for line in plans_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(plans[0]["review_status"], "draft")
            self.assertEqual(plans[0]["weak_label_config"]["idf_training_date_range"], ["2020-01-01", "2020-01-01"])
            self.assertEqual(len(plan_path_examples(events[0], plans[0], label_policy="weak-extractive")), 2)
            with self.assertRaisesRegex(ValueError, "only reviewed"):
                plan_path_examples(events[0], plans[0])
            output = root / "weak-plan.pt"
            result = train_explicit_plan_model(
                events_path, plans_path, output, registry_path=root / "registry.jsonl",
                epochs=1, batch_size=2, context_length=16, plan_length=24,
                width=16, layers=1, heads=2, min_frequency=1,
                max_vocabulary=64, device_name="cpu", label_policy="weak-extractive",
            )
            registered = json.loads((root / "registry.jsonl").read_text(encoding="utf-8"))
            from speechprefix.model_versions import sha256_file

            self.assertEqual(registered["training"]["label_policy"], "weak-extractive")
            self.assertEqual(registered["additional_data"]["sha256"], sha256_file(plans_path))
            self.assertTrue(output.is_file())
            self.assertEqual(result["model_version"], registered["model_version"])

    def test_parser_reads_ordered_hierarchy_and_rejects_malformed_paths(self):
        self.assertEqual(
            parse_plan_path([
                "__theme__", "economic", "growth", "__section__", "tax", "policy",
                "__point__", "tax", "relief", "__endplan__",
            ]),
            {"theme": "economic growth", "section": "tax policy", "point": "tax relief"},
        )
        with self.assertRaisesRegex(ValueError, "complete ordered"):
            parse_plan_path(["__section__", "tax", "__theme__", "economy"])


if __name__ == "__main__":
    unittest.main()

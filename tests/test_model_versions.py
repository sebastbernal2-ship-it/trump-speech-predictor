import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from speechprefix.model_versions import (
    assert_new_model_artifact,
    create_model_version,
    register_model_version,
)


class ModelVersionTests(unittest.TestCase):
    def test_registers_dataset_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text('{"text":"fixture"}\n', encoding="utf-8")
            artifact = root / "model.pt"
            artifact.write_bytes(b"fixture model bytes")
            registry = root / "registry.jsonl"
            version = create_model_version(
                model_family="fixture",
                architecture="v1",
                task="continuation",
                dataset_path=dataset,
                config={"width": 8},
                training={"seed": 17},
            )
            row = register_model_version(artifact, version, registry, metrics={"loss": 1.2})
            saved = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(saved, row)
            self.assertEqual(row["dataset"]["sha256"], hashlib.sha256(dataset.read_bytes()).hexdigest())
            self.assertEqual(row["artifact"]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
            self.assertEqual(row["metrics"]["loss"], 1.2)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_model_version(artifact, version, registry)

    def test_refuses_existing_or_previously_registered_artifact_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.pt"
            artifact.write_bytes(b"model")
            with self.assertRaisesRegex(FileExistsError, "versioned path"):
                assert_new_model_artifact(artifact, root / "registry.jsonl")
            artifact.unlink()
            registry = root / "registry.jsonl"
            registry.write_text(json.dumps({"artifact": {"path": str(artifact.resolve())}}) + "\n")
            with self.assertRaisesRegex(ValueError, "already used"):
                assert_new_model_artifact(artifact, registry)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("optional PyTorch dependency is not installed")

from speechprefix.hybrid_train import generate_hybrid, train_hybrid


class HybridTrainingTests(unittest.TestCase):
    def test_tiny_chronological_training_saves_loadable_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "speeches.jsonl"
            text = "we believe in our country we believe in the future " * 3
            records = [
                {"speech_id": f"speech-{day}", "date": f"2020-01-0{day}",
                 "event": "remarks", "source": "fixture", "text": text}
                for day in (1, 2, 3)
            ]
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            checkpoint = root / "speechplan.pt"
            train_hybrid(
                source, checkpoint, epochs=1, batch_size=2, stride=8,
                min_frequency=1, max_vocabulary=64, width=32, layers=2,
                heads=4, codebook_size=8, context_length=8, target_length=8,
                device_name="cpu",
            )
            self.assertTrue(checkpoint.is_file())
            self.assertIsInstance(
                generate_hybrid(checkpoint, "we believe", max_new_tokens=2, device_name="cpu"),
                str,
            )


if __name__ == "__main__":
    unittest.main()

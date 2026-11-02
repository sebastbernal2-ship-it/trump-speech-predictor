import sys
import types
import unittest
from unittest.mock import Mock, patch

from speechprefix.hf_download import download_snapshot


def fake_hub(download):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    utils = types.ModuleType("huggingface_hub.utils")
    utils.enable_progress_bars = Mock()
    return {"huggingface_hub": hub, "huggingface_hub.utils": utils}


class DownloadTests(unittest.TestCase):
    def test_retries_transient_failure_then_succeeds(self):
        download = Mock(side_effect=[TimeoutError(), "/cached/model"])
        with patch.dict(sys.modules, fake_hub(download)), patch("time.sleep") as sleep:
            result = download_snapshot("example/model", retries=3, timeout_seconds=2)
        self.assertEqual(result, "/cached/model")
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_stops_after_bounded_transient_failures(self):
        download = Mock(side_effect=TimeoutError())
        with patch.dict(sys.modules, fake_hub(download)), patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "Partial files remain cached"):
                download_snapshot("example/model", retries=2, timeout_seconds=1)
        self.assertEqual(download.call_count, 2)

    def test_does_not_retry_nontransient_errors(self):
        download = Mock(side_effect=ValueError("bad model id"))
        with patch.dict(sys.modules, fake_hub(download)), patch("time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                download_snapshot("bad/model", retries=3, timeout_seconds=1)
        self.assertEqual(download.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Bounded, resumable Hugging Face snapshot download."""

from __future__ import annotations

import os
import sys
import time


def _is_transient(error: Exception) -> bool:
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        HfHubHTTPError = ()
    if HfHubHTTPError and isinstance(error, HfHubHTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", 0)
        return status in (408, 429) or status >= 500
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    name = type(error).__name__.lower()
    return any(marker in name for marker in ("timeout", "connection", "network", "offline", "transport"))


def download_snapshot(
    model_id: str,
    retries: int = 3,
    timeout_seconds: int = 20,
    allow_patterns: list[str] | None = None,
    revision: str | None = None,
) -> str:
    """Download with visible progress, per-request timeouts, and bounded retry/resume."""
    if retries < 1 or timeout_seconds < 1:
        raise ValueError("retries and timeout_seconds must be positive")

    # These are read when huggingface_hub is imported, so set them first.
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_seconds)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout_seconds)
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
    for attempt in range(1, retries + 1):
        try:
            options = {
                "repo_id": model_id,
                "max_workers": 1,
                "etag_timeout": timeout_seconds,
                "allow_patterns": allow_patterns or [
                    "*.safetensors", "config.json", "generation_config.json",
                    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "added_tokens.json", "vocab.json", "merges.txt", "*.model",
                ],
            }
            if revision is not None:
                options["revision"] = revision
            return snapshot_download(**options)
        except Exception as error:
            if not _is_transient(error) or attempt == retries:
                raise RuntimeError(
                    f"Model download failed after {attempt} attempt(s) ({type(error).__name__}). "
                    "Partial files remain cached; rerun the command when internet is stable to resume."
                ) from None
            wait = min(2 ** attempt, 8)
            print(
                f"Temporary download failure ({type(error).__name__}); "
                f"retrying {attempt + 1}/{retries} in {wait}s. Cached chunks will be reused.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")

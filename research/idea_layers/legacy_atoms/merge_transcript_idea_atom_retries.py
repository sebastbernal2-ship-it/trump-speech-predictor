"""Create a new immutable atom artifact by replacing failed rows with validated retries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BASE_DEFAULT = ROOT / "data/source-archives/2019-weak-empty-semantic-idea-atoms-v1.jsonl"
RETRY_DEFAULT = ROOT / "data/source-archives/2019-idea-atoms-retry-fe02b997-v1.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/legacy_atoms/2019-weak-empty-semantic-idea-atoms-v1-complete.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    data = path.read_bytes()
    rows = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    ids = [row.get("event_id") for row in rows]
    if any(not isinstance(event_id, str) or not event_id for event_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: event IDs must be unique nonempty strings")
    return rows, data


def _manifest_for(path: Path, raw: bytes) -> tuple[dict[str, Any], str]:
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("output_sha256") != _sha256(raw):
        raise ValueError(f"run manifest does not verify output bytes: {path}")
    return manifest, _sha256(manifest_bytes)


def merge(base_path: Path, retry_paths: list[Path], output_path: Path) -> dict[str, Any]:
    base_rows, base_bytes = _read_rows(base_path)
    base_manifest, base_manifest_hash = _manifest_for(base_path, base_bytes)
    base_by_id = {row["event_id"]: row for row in base_rows}
    replacements: dict[str, tuple[dict[str, Any], str]] = {}
    retry_manifest_hashes = {}
    for retry_path in retry_paths:
        retry_rows, retry_bytes = _read_rows(retry_path)
        retry_manifest, retry_manifest_hash = _manifest_for(retry_path, retry_bytes)
        retry_manifest_hashes[str(retry_path)] = retry_manifest_hash
        for retry in retry_rows:
            event_id = retry["event_id"]
            original = base_by_id.get(event_id)
            if original is None:
                raise ValueError(f"retry event {event_id} is absent from base artifact")
            if original.get("extraction_status") != "model_error":
                raise ValueError(f"base event {event_id} is not a failed extraction")
            if retry.get("extraction_status") not in {"atoms_found", "no_supported_idea", "insufficient_transcript"}:
                raise ValueError(f"retry event {event_id} is not a successful validated result")
            for field in ("event_date", "source_record_sha256", "target_transcript_sha256"):
                if retry.get(field) != original.get(field):
                    raise ValueError(f"retry event {event_id} has mismatched {field}")
            if event_id in replacements:
                raise ValueError(f"multiple retries supplied for event {event_id}")
            replacements[event_id] = (retry, _sha256(retry_bytes))
    if not replacements:
        raise ValueError("no retry replacements were supplied")

    merged_rows = []
    for original in base_rows:
        replacement = replacements.get(original["event_id"])
        if replacement is None:
            merged_rows.append(original)
            continue
        retry, retry_file_hash = replacement
        merged_rows.append({
            **retry,
            "retry_provenance": {
                "replaces_base_output_sha256": _sha256(base_bytes),
                "retry_output_file_sha256": retry_file_hash,
            },
        })
    output_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged_rows
    ).encode("utf-8")
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite merged atom artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {
        status: sum(row.get("extraction_status") == status for row in merged_rows)
        for status in ("atoms_found", "no_supported_idea", "insufficient_transcript", "model_error")
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "transcript-semantic-idea-atom-retry-merge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "base_output_sha256": _sha256(base_bytes),
        "base_manifest_sha256": base_manifest_hash,
        "retry_output_sha256": {str(path): _sha256(path.read_bytes()) for path in retry_paths},
        "retry_manifest_sha256": retry_manifest_hashes,
        "retry_replacements": sorted(replacements),
        "selected_event_count": len(merged_rows),
        "status_counts": status_counts,
        "atom_count": sum(len(row.get("atoms", [])) for row in merged_rows),
        "forecastable_atom_count": sum(
            atom.get("forecastable") is True
            for row in merged_rows for atom in row.get("atoms", [])
        ),
        "output_sha256": _sha256(output_bytes),
        "outputs_local_only": True,
        "limitations": base_manifest.get("limitations", []) + [
            "A retry replaces only an initial model-error row; it does not provide an independent semantic adjudication."
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return {"output": str(output_path), "manifest": str(manifest_path), **status_counts, "atom_count": manifest["atom_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--retry", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    retry_paths = args.retry or [RETRY_DEFAULT]
    print(json.dumps(merge(args.base, retry_paths, args.output), indent=2))


if __name__ == "__main__":
    main()

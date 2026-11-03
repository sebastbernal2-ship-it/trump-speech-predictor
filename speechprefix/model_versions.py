"""Append-only provenance and artifact registry for trained model versions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "registry.jsonl"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _source_fingerprint() -> str:
    files = sorted(Path(__file__).parent.rglob("*.py"))
    files.extend(path for path in (PROJECT_ROOT / "pyproject.toml",) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_provenance() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout
        return {"revision": revision, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def create_model_version(
    *,
    model_family: str,
    architecture: str,
    task: str,
    dataset_path: str | Path,
    config: dict,
    training: dict,
    additional_data_path: str | Path | None = None,
    parent_model_version: str | None = None,
) -> dict:
    """Capture immutable training inputs and code identity before saving weights."""
    dataset = Path(dataset_path)
    timestamp = datetime.now(timezone.utc)
    run_suffix = uuid.uuid4().hex[:8]
    model_version = f"{model_family}/{architecture}/{timestamp:%Y%m%dT%H%M%SZ}-{run_suffix}"
    data = {"path": _display_path(dataset), "sha256": sha256_file(dataset)}
    additional_data = None
    if additional_data_path is not None:
        extra = Path(additional_data_path)
        additional_data = {"path": _display_path(extra), "sha256": sha256_file(extra)}

    return {
        "registry_schema_version": 1,
        "model_version": model_version,
        "model_family": model_family,
        "architecture": architecture,
        "task": task,
        "created_at": timestamp.isoformat(),
        "parent_model_version": parent_model_version,
        "dataset": data,
        "additional_data": additional_data,
        "config": config,
        "training": training,
        "source": {**_git_provenance(), "package_sha256": _source_fingerprint()},
        "environment": {
            "python": platform.python_version(),
            "speechprefix": _installed_version("speechprefix"),
            "torch": _installed_version("torch"),
            "transformers": _installed_version("transformers"),
        },
    }


def assert_new_model_artifact(artifact_path: str | Path, registry_path: str | Path = DEFAULT_REGISTRY) -> None:
    """Prevent overwriting an artifact or reusing a path already in the registry."""
    artifact = Path(artifact_path)
    if artifact.exists():
        raise FileExistsError(f"model artifact already exists; choose a new versioned path: {artifact}")
    registry = Path(registry_path)
    if registry.exists():
        with registry.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("artifact", {}).get("path") == _display_path(artifact):
                    raise ValueError(f"artifact path was already used by a model version: {_display_path(artifact)}")


def register_model_version(
    artifact_path: str | Path,
    version: dict,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    metrics: dict | None = None,
    provenance_status: str = "complete",
) -> dict:
    """Append an artifact hash and metrics to the model registry exactly once."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {artifact}")
    registry = Path(registry_path)
    artifact_record = {"path": _display_path(artifact), "sha256": sha256_file(artifact)}
    row = {
        **version,
        "artifact": artifact_record,
        "metrics": metrics or {},
        "provenance_status": provenance_status,
    }
    existing = []
    if registry.exists():
        with registry.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing.append(json.loads(line))
    if any(item.get("model_version") == version.get("model_version") for item in existing):
        raise ValueError(f"model version is already registered: {version.get('model_version')}")
    if any(item.get("artifact", {}).get("path") == artifact_record["path"] for item in existing):
        raise ValueError(f"artifact path is already registered: {artifact_record['path']}")

    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    return row

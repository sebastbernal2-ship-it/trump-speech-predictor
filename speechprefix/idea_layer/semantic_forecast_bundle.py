"""Bind frozen semantic forecasts into the assistant-review scoring format."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified_forecast(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    forecast = json.loads(raw)
    if not isinstance(forecast, dict) or forecast.get("record_type") != "idea-layer-semantic-forecast":
        raise ValueError(f"{source}: expected an idea-layer semantic forecast")
    expected = forecast.get("forecast_sha256")
    payload = dict(forecast)
    payload.pop("forecast_sha256", None)
    if expected != _sha256(_canonical(payload).encode("utf-8")):
        raise ValueError(f"{source}: forecast_sha256 does not validate")
    return forecast, _sha256(raw)


def build_semantic_forecast_evaluation_bundle(
    forecast_paths: Iterable[str | Path],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Make a frozen, multi-event report suitable for the existing review scorer."""
    paths = list(forecast_paths)
    if not paths:
        raise ValueError("at least one semantic forecast is required")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite semantic forecast bundle: {output_path}")
    forecasts = []
    member_hashes: dict[str, str] = {}
    pinned_config: tuple[str, str, str, int] | None = None
    for path in paths:
        forecast, file_hash = _read_verified_forecast(path)
        event_id = forecast.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in member_hashes:
            raise ValueError(f"duplicate or invalid semantic forecast event ID: {event_id!r}")
        config = (
            forecast.get("provider"), forecast.get("model"), forecast.get("ranker_version"), forecast.get("top_k"),
        )
        if any(not isinstance(item, str) or not item for item in config[:3]) or type(config[3]) is not int or config[3] < 1:
            raise ValueError(f"{event_id}: missing pinned provider/model/ranker/top_k")
        if pinned_config is None:
            pinned_config = config
        elif config != pinned_config:
            raise ValueError("all bundled forecasts must use the same provider, model, ranker, and top_k")
        event_start_at = forecast.get("event_start_at")
        event_date = forecast.get("event_date")
        if not isinstance(event_start_at, str) or not isinstance(event_date, str) or event_date != event_start_at[:10]:
            raise ValueError(f"{event_id}: event_date must match event_start_at")
        candidates = forecast.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{event_id}: candidates must be a list")
        top_k = config[3]
        selected = candidates[:top_k]
        if [candidate.get("rank") for candidate in selected] != list(range(1, len(selected) + 1)):
            raise ValueError(f"{event_id}: candidate ranks must be consecutive from 1")
        predictions = []
        for candidate in selected:
            if not isinstance(candidate.get("idea_id"), str) or not isinstance(candidate.get("idea"), str):
                raise ValueError(f"{event_id}: candidate needs idea_id and idea")
            predictions.append({
                "rank": candidate["rank"],
                "idea_id": candidate["idea_id"],
                "idea": candidate["idea"],
                "score": candidate.get("score"),
                "score_kind": candidate.get("score_kind"),
                "evidence_source_ids": candidate.get("evidence_source_ids", []),
                "uncertainty": candidate.get("uncertainty"),
            })
        forecasts.append({
            "event_id": event_id,
            "event_date": event_date,
            "prediction_cutoff_at": forecast.get("prediction_cutoff_at"),
            "semantic_forecast_sha256": forecast["forecast_sha256"],
            "predictions": predictions,
            "abstained": not predictions,
        })
        member_hashes[event_id] = file_hash
    forecasts.sort(key=lambda row: (row["event_date"], row["event_id"]))
    assert pinned_config is not None
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "semantic-idea-forecast-evaluation-bundle",
        "evaluation_mode": "frozen semantic forecasts for post-hoc assistant review",
        "provider": pinned_config[0],
        "model": pinned_config[1],
        "ranker_version": pinned_config[2],
        "top_k": pinned_config[3],
        "member_forecast_file_sha256": member_hashes,
        "forecasts": forecasts,
        "limitations": [
            "Bundle contains only frozen candidate rankings; it does not verify that evidence citations semantically entail an idea.",
            "All scores and model ratings are uncalibrated and must not be interpreted as probabilities.",
            "Evaluation labels remain assistant-reviewed references, not objective semantic ground truth.",
        ],
    }
    bundle["report_content_sha256"] = _sha256(_canonical(bundle).encode("utf-8"))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as stream:
                json.dump(bundle, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite semantic forecast bundle: {output}") from None
    return bundle

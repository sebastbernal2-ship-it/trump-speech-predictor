"""Add a non-destructive topic layer above transcript-grounded idea points.

Topics group existing point IDs; they never replace, merge, or rewrite points.
The output is an AI-authored organizational candidate, not semantic gold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_VERSION = "speech-idea-topic-grouping-v1.1"
PROMPT_VERSION = "non-destructive-issue-umbrella-topic-partition-v1.1"
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_MODEL = "gpt-6-luna"
_TOPIC_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

_SPEC = importlib.util.spec_from_file_location(
    "layered_outcome_single_pass",
    ROOT / "research/idea_layers/outcomes/extract_layered_transcript_outcomes.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load transcript outcome helpers")
_SINGLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SINGLE)


class TopicLayerError(ValueError):
    """Raised when the proposed topic partition violates its structural contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_prompt(row: dict[str, Any]) -> str:
    ideas = [
        {
            "idea_id": idea["idea_id"],
            "concept": idea["concept"],
            "summary": idea["summary"],
            "kind": idea["kind"],
            "prominence": idea["prominence"],
        }
        for idea in row["ideas"]
    ]
    return """Organize the supplied transcript-grounded conversation points into a separate, broader TOPIC layer. This is a grouping task, not a point-extraction or point-rewriting task.

Definitions and constraints:
- A point is an independently meaningful claim, position, proposal, appeal, or event-specific focus. A topic is a broader subject/thread that helps readers navigate one or more points.
- Preserve every supplied point as an independent node with its existing ID. Do not merge, split, rename, omit, add, or rewrite point nodes.
- Group points under a coherent, useful issue/thread. A topic may be a broad umbrella for distinct claims, policies, or events that share a real subject; it does not imply the member points are the same story or that one caused/supports another. Similar names or political framing alone are insufficient.
- Keep every claim and event as a separate member point. Topic grouping is not point merging: retain all original IDs, summaries, evidence, and boundaries unchanged.
- A topic label should be a concise neutral subject phrase, not a new claim. Its summary may describe the shared scope but must not assert facts beyond the member points.
- Prefer a coherent multi-point topic over a singleton when a reasonable umbrella exists. Use a singleton only when no non-generic shared subject is warranted. Do not impose a fixed number of topics.
- Every point ID must belong to exactly one topic group. Topic membership is organizational only; it does not mean the points are paraphrases, the same story, or interchangeable forecast targets.
- Do not merge the shooting event, the speaker's account of the suspect's entry/status, broader immigration claims, and policy proposals into one point. A carefully named umbrella topic may group the immigration-related points while leaving the shooting response and each immigration claim as separate members.

Return only JSON:
{
  "topics": [
    {
      "topic_id": "lowercase_snake_case_id",
      "label": "short neutral topic name",
      "summary": "one sentence describing the shared subject/scope",
      "member_idea_ids": ["existing idea IDs"]
    }
  ]
}

POINT INVENTORY:
""" + json.dumps(ideas, ensure_ascii=False)


def validate(value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    topics = value.get("topics")
    if not isinstance(topics, list) or not topics:
        raise TopicLayerError("topics must be a nonempty list")
    expected = {idea["idea_id"] for idea in row["ideas"]}
    topic_ids: set[str] = set()
    membership: dict[str, str] = {}
    normalized = []
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise TopicLayerError(f"topics[{index}] must be an object")
        topic_id = topic.get("topic_id")
        if not isinstance(topic_id, str) or not _TOPIC_ID.fullmatch(topic_id) or topic_id in topic_ids:
            raise TopicLayerError(f"topics[{index}].topic_id must be unique lowercase snake_case")
        topic_ids.add(topic_id)
        label, summary = topic.get("label"), topic.get("summary")
        if not isinstance(label, str) or not label.strip() or not isinstance(summary, str) or not summary.strip():
            raise TopicLayerError(f"topics[{index}] label and summary must be nonempty")
        members = topic.get("member_idea_ids")
        if not isinstance(members, list) or not members or any(not isinstance(item, str) for item in members):
            raise TopicLayerError(f"topics[{index}].member_idea_ids must be a nonempty list")
        if len(set(members)) != len(members) or set(members) - expected:
            raise TopicLayerError(f"topics[{index}] has duplicate or unknown point IDs")
        for idea_id in members:
            if idea_id in membership:
                raise TopicLayerError(f"point {idea_id} belongs to multiple primary topics")
            membership[idea_id] = topic_id
        normalized.append({
            "topic_id": topic_id,
            "label": label.strip(),
            "summary": summary.strip(),
            "member_idea_ids": sorted(members),
        })
    if set(membership) != expected:
        raise TopicLayerError(f"topic layer omitted point IDs: {sorted(expected - set(membership))}")
    normalized.sort(key=lambda topic: topic["topic_id"])
    return {
        "topics": normalized,
        "topic_memberships": [
            {"idea_id": idea_id, "topic_id": membership[idea_id], "relation": "organized_under"}
            for idea_id in sorted(membership)
        ],
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return rows


def run(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists() or Path(str(output_path) + ".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite topic-layer output: {output_path}")
    raw_input = input_path.read_bytes()
    rows = _read_rows(input_path)
    output_rows = []
    for row in rows:
        prompt = build_prompt(row)
        try:
            response = _SINGLE._call_pi(
                prompt, provider=args.provider, model=args.model, timeout=args.timeout,
            )
            result = validate(_SINGLE._parse_object(response), row)
            output_rows.append({
                "artifact_type": "speech_idea_topic_layer",
                "status": "validated_candidate",
                "event_id": row.get("event_id"),
                "source_point_inventory_sha256": _sha256(
                    json.dumps(row["ideas"], sort_keys=True, ensure_ascii=False).encode("utf-8")
                ),
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "response_sha256": _sha256(response.encode("utf-8")),
                "topic_layer_version": SCRIPT_VERSION,
                "prompt_version": PROMPT_VERSION,
                **result,
                "counts": {
                    "input_points": len(row["ideas"]),
                    "topics": len(result["topics"]),
                    "singleton_topics": sum(len(topic["member_idea_ids"]) == 1 for topic in result["topics"]),
                },
                "structural_validation": True,
                "semantic_validation": False,
            })
        except Exception as exc:
            output_rows.append({
                "artifact_type": "speech_idea_topic_layer",
                "status": "model_error",
                "event_id": row.get("event_id"),
                "topic_layer_version": SCRIPT_VERSION,
                "prompt_version": PROMPT_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output_rows
    ).encode("utf-8")
    output_path.write_bytes(output_bytes)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "input_path": str(input_path),
        "input_sha256": _sha256(raw_input),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_bytes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(output_rows),
        "success_count": sum(row["status"] == "validated_candidate" for row in output_rows),
        "error_count": sum(row["status"] != "validated_candidate" for row in output_rows),
        "semantic_validation": False,
        "limitations": [
            "Topics organize existing point nodes; they do not repair omissions or point-boundary errors.",
            "A valid partition proves ID coverage only, not that semantic topic groupings are correct.",
            "Topic nodes are not forecast targets unless separately defined and evaluated.",
        ],
    }
    manifest_path = Path(str(output_path) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in output_rows:
        if row["status"] == "validated_candidate":
            print(row["event_id"], row["counts"])
        else:
            print(row["event_id"], row["status"], row.get("error", ""))
    return 0 if manifest["error_count"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="layered outcome JSONL")
    parser.add_argument("--output", type=Path, required=True, help="local topic-layer JSONL output")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    try:
        return run(parser.parse_args())
    except (OSError, ValueError) as exc:
        print(f"topic-layer generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

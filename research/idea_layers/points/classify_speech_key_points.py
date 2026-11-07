"""Diagnostic second pass: classify candidate ideas as key points or subordinate details.

This preserves idea IDs and does not merge/delete candidates. It optionally adds
KPH-inspired specific-to and evidence/support hierarchy edges. Model labels are
not semantic gold and should be reviewed before any forecasting use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_VERSION = "speech-key-point-classification-v1.0"
PROMPT_VERSION = "key-point-salience-and-nonmerging-hierarchy-v1"
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_MODEL = "gpt-6-luna"

_SPEC = importlib.util.spec_from_file_location(
    "layered_outcome_single_pass",
    ROOT / "research/idea_layers/outcomes/extract_layered_transcript_outcomes.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load transcript outcome helpers")
_SINGLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SINGLE)

STATUSES = {"key_point", "supporting_detail", "aside_or_ceremonial", "uncertain"}
KEY_POINT_RELATIONS = {"specific_to"}
DETAIL_RELATIONS = {"evidence_for", "illustrates", "qualifies", "implements"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_prompt(row: dict[str, Any]) -> str:
    mentions = {item["mention_id"]: item for item in row["mentions"]}
    ideas = []
    for idea in row["ideas"]:
        linked = []
        for edge in row["links"]:
            if edge["idea_id"] == idea["idea_id"]:
                mention = mentions[edge["mention_id"]]
                linked.append({
                    "relation": edge["relation"],
                    "mention": mention["text"],
                    "evidence": [quote["exact_quote"] for quote in mention["evidence"]],
                })
        ideas.append({**idea, "linked_mentions": linked})
    return """You are auditing a candidate inventory of conversation points from ONE speech. The input points and quotes are transcript-derived outcomes. Do not use outside knowledge. Classify every supplied idea; do not rewrite, merge, split, add, or delete ideas.

Purpose: distinguish independently meaningful KEY POINTS from subordinate supporting material, and add a sparse hierarchy without collapsing distinct propositions.

A `key_point` is a substantive claim, position, proposal, appeal, argument, or event-specific focus that the speaker makes a main point, develops/repeats, or gives clear independent emphasis. A single focused point can qualify even if it occurs once. Preserve distinct claims even when they share a person, event, policy area, rationale, or rhetorical frame.

A `supporting_detail` is a statistic, example, illustration, implementation detail, or subordinate claim used only to develop a different key point. Assign a parent only when the input wording/evidence makes that relationship clear. A topic match is not enough. `aside_or_ceremonial` is routine, incidental, or non-substantive material that is not itself a key point. Use `uncertain` rather than guessing. Do not use forecastability as a proxy for importance.

Hierarchy rules:
- Keep all distinct key points as separate nodes. A parent relation is an edge, never a merge.
- For a key point child, use `parent_relation="specific_to"` only when it is genuinely a narrower point contained by the broader parent proposition. Mere topical relatedness, shared names, evidence, or contradiction does not qualify. Otherwise leave it a root.
- A supporting detail may point to a key-point parent using `evidence_for`, `illustrates`, `qualifies`, or `implements`, only when that textual role is warranted. Otherwise use no parent.
- `aside_or_ceremonial` and `uncertain` have no parent.
- Never connect the Ukraine call / complaint dispute to impeachment as a parent-child pair merely because the speech connects them rhetorically. Preserve the distinction between the legislative impeachment process, the Ukraine conversation/complaint dispute, and separate Biden-family allegations.
- Return one annotation for every input `idea_id`, exactly once.

Return only JSON:
{
  "annotations": [
    {
      "idea_id": "existing ID",
      "status": "key_point" | "supporting_detail" | "aside_or_ceremonial" | "uncertain",
      "parent_idea_id": "existing ID or null",
      "parent_relation": "specific_to" | "evidence_for" | "illustrates" | "qualifies" | "implements" | null,
      "rationale": "brief textual reason"
    }
  ]
}

CANDIDATE INVENTORY:
""" + json.dumps(ideas, ensure_ascii=False)


def validate(value: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    expected = {item["idea_id"] for item in row["ideas"]}
    received: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(annotations):
        if not isinstance(item, dict):
            raise ValueError(f"annotations[{index}] must be an object")
        idea_id = item.get("idea_id")
        if not isinstance(idea_id, str) or idea_id not in expected or idea_id in received:
            raise ValueError(f"annotations[{index}] has unknown or duplicate idea_id")
        status = item.get("status")
        if status not in STATUSES:
            raise ValueError(f"annotations[{index}] has unsupported status")
        parent_id = item.get("parent_idea_id")
        relation = item.get("parent_relation")
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"annotations[{index}].rationale must be nonempty")
        if parent_id is None:
            if relation is not None:
                raise ValueError(f"annotations[{index}] has relation without parent")
        elif not isinstance(parent_id, str) or parent_id not in expected or parent_id == idea_id:
            raise ValueError(f"annotations[{index}] has invalid parent")
        if status == "key_point":
            if parent_id is None and relation is not None:
                raise ValueError(f"annotations[{index}] has invalid root relation")
            if parent_id is not None and relation not in KEY_POINT_RELATIONS:
                raise ValueError(f"annotations[{index}] key-point parent must be specific_to")
        elif status == "supporting_detail":
            if parent_id is not None and relation not in DETAIL_RELATIONS:
                raise ValueError(f"annotations[{index}] has unsupported detail relation")
        elif parent_id is not None or relation is not None:
            raise ValueError(f"annotations[{index}] status cannot have a parent")
        received[idea_id] = {
            "idea_id": idea_id,
            "status": status,
            "parent_idea_id": parent_id,
            "parent_relation": relation,
            "rationale": rationale.strip(),
        }
    if set(received) != expected:
        raise ValueError(f"annotations omitted IDs: {sorted(expected - set(received))}")
    for item in received.values():
        parent_id = item["parent_idea_id"]
        if parent_id is None:
            continue
        parent = received[parent_id]
        if parent["status"] != "key_point":
            raise ValueError(f"parent {parent_id} is not classified as a key point")
    for idea_id in received:
        seen = set()
        current = idea_id
        while received[current]["parent_idea_id"] is not None:
            if current in seen:
                raise ValueError(f"hierarchy contains a cycle involving {current}")
            seen.add(current)
            current = received[current]["parent_idea_id"]
    return {"annotations": [received[key] for key in sorted(received)]}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return rows


def run(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    raw_input = input_path.read_bytes()
    rows = _read_rows(input_path)
    output_rows = []
    for row in rows:
        prompt = build_prompt(row)
        prompt_hash = sha256(prompt.encode("utf-8"))
        try:
            response = _SINGLE._call_pi(
                prompt, provider=args.provider, model=args.model, timeout=args.timeout,
            )
            response_hash = sha256(response.encode("utf-8"))
            result = validate(_SINGLE._parse_object(response), row)
            output_rows.append({
                "prompt_sha256": prompt_hash,
                "response_sha256": response_hash,
                "artifact_type": "speech_key_point_classification",
                "status": "validated_candidate",
                "event_id": row.get("event_id"),
                "input_idea_inventory_sha256": sha256(json.dumps(row["ideas"], sort_keys=True).encode()),
                "classifier": SCRIPT_VERSION,
                "prompt_version": PROMPT_VERSION,
                **result,
                "counts": {
                    status: sum(item["status"] == status for item in result["annotations"])
                    for status in sorted(STATUSES)
                },
                "key_point_hierarchy_edges": sum(item["parent_relation"] == "specific_to" for item in result["annotations"]),
                "support_relation_edges": sum(item["parent_relation"] in DETAIL_RELATIONS for item in result["annotations"]),
                "structural_validation": True,
                "semantic_validation": False,
            })
        except Exception as exc:
            output_rows.append({
                "artifact_type": "speech_key_point_classification",
                "status": "model_error",
                "event_id": row.get("event_id"),
                "prompt_sha256": prompt_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "classifier": SCRIPT_VERSION,
                "prompt_version": PROMPT_VERSION,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output_rows).encode()
    output_path.write_bytes(payload)
    manifest = {
        "script_version": SCRIPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "input_path": str(input_path),
        "input_sha256": sha256(raw_input),
        "output_path": str(output_path),
        "output_sha256": sha256(payload),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(output_rows),
        "success_count": sum(row["status"] == "validated_candidate" for row in output_rows),
        "error_count": sum(row["status"] != "validated_candidate" for row in output_rows),
        "semantic_validation": False,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for row in output_rows:
        if row["status"] == "validated_candidate":
            print(row["event_id"], row["counts"], "edges=", row["key_point_hierarchy_edges"])
        else:
            print(row["event_id"], row["status"], row.get("error", ""))
    return 0 if manifest["error_count"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="local layered-outcomes JSONL")
    parser.add_argument("--output", type=Path, required=True, help="local diagnostic JSONL output")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

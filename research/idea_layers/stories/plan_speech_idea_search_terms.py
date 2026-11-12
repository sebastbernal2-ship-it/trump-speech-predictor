"""Use Codex to prepare auditable narrow search phrases for a backwards case study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from speechprefix.context_theme_ranker import read_jsonl
from speechprefix.idea_layer.transcript_outcomes import validate_layered_outcome
from generate_open_vocabulary_context_ideas import _call_pi
from speech_news_circulation_case_study import validate_term_plan

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES_DEFAULT = ROOT / "data/source-archives/2019-11-04-lexington-rally-outcomes-codex-v1.jsonl"
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/stories/2019-11-04-lexington-search-terms-codex-v1.json"
MODEL = "gpt-6.1-sol"
PROMPT_VERSION = "transcript-idea-news-search-phrase-expansion-v2-neutral-topic"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse(text: str, idea_ids: set[str]) -> dict[str, list[str]]:
    decoder = json.JSONDecoder()
    payload = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload = value
            break
    return validate_term_plan(payload, idea_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES_DEFAULT)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--event-id", default="dt45-19-8a126a9f")
    parser.add_argument("--provider", default="openai-codex")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    outcomes = [json.loads(line) for line in args.outcomes.read_text(encoding="utf-8").splitlines() if line.strip()]
    outcome = next((row for row in outcomes if row.get("event_id") == args.event_id), None)
    if outcome is None:
        raise ValueError("no outcome for selected event")
    events = {event["event_id"]: event for event in read_jsonl(args.events)}
    if args.event_id not in events:
        raise ValueError("no source event for selected event ID")
    turns_by_id = {
        turn["turn_id"]: turn["text"]
        for turn in events[args.event_id].get("turns", [])
        if turn.get("speaker_role") == "target_speaker"
    }
    validate_layered_outcome(outcome, turns_by_id)
    ideas = outcome["ideas"]
    idea_ids = {idea["idea_id"] for idea in ideas}
    prompt = f"""Create a high-recall but narrow search phrase plan for matching the following speech-idea labels to historical news headlines and one speaker's post archive.

Only use each idea's concept and summary to produce literal phrases, close synonyms, named entities, or common alternate wording that directly refer to that same issue. For each idea, include neutral phrases for the underlying topic/story (for example, ‘impeachment inquiry’ when the idea is an attack on impeachment) as well as claim-framing phrases when they help retrieve that perspective. Do not introduce other events, later knowledge, generic political vocabulary, the speaker's name alone, or broad terms like news, politics, America, or economy. Keep each phrase short (one to four words), specific, case-insensitive searchable, and at most eight per idea. Search phrases are only retrieval aids, not claims that a pre-speech article/post exists.

Return exactly {{"terms_by_idea": [{{"idea_id": "...", "terms": ["phrase", "phrase"]}}]}}. Include every input idea ID exactly once. No extra keys, prose, markdown, or model judgments.

IDEAS (transcript-derived retrospective labels; don't rewrite their meanings):
{json.dumps([{'idea_id': idea['idea_id'], 'concept': idea.get('concept'), 'summary': idea.get('summary')} for idea in ideas], ensure_ascii=False)}"""
    response = _call_pi(prompt, provider=args.provider, model=args.model, timeout=900)
    terms = _parse(response, idea_ids)
    output = args.output.resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite term plan: {output}")
    payload = {
        "schema_version": 1,
        "artifact_type": "speech-idea-search-term-plan",
        "event_id": args.event_id,
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "terms_by_idea": [{"idea_id": key, "terms": values} for key, values in terms.items()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    bytes_out = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(bytes_out)
    manifest = {
        "schema_version": 1,
        "artifact_type": "speech-idea-search-term-plan-manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_id": args.event_id,
        "provider": args.provider,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "outcome_sha256": _hash(args.outcomes.read_bytes()),
        "source_event_corpus_sha256": _hash(args.events.read_bytes()),
        "response_sha256": _hash(response.encode()),
        "output_sha256": _hash(bytes_out),
        "terms_are_retrieval_aids_not_labels": True,
        "output_local_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "ideas": len(terms)}, indent=2))


if __name__ == "__main__":
    main()

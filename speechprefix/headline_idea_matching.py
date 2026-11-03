"""Open-vocabulary headline ideas and transcript phrase-overlap diagnostics.

Headlines generate candidate ideas; target-speaker transcript turns are used
only after the fact to mark literal/near-literal mention evidence.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .context_theme_ranker import read_jsonl

WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
STOPWORDS = frozenset("""a an and are as at be been being by for from had has have he her hers him his i if in into is it its me my of on or our ours she that the their theirs them there these they this those to was we were what when where which who will with you your about after all also amid among around because before between both but can could during each even every first following further how however including just more most new not now only other over said says some such than then through today under until very while why would year years breaking latest live news report update watch video photos trump donald president campaign administration us""".split())
ALIASES = {
    "ukrainian": "ukraine", "ukrainians": "ukraine",
    "russian": "russia", "russians": "russia",
    "american": "america", "americans": "america",
}
KNOWN_ENTITIES = frozenset("ukraine russia china iran israel gaza north korea nato biden clinton putin trump mexico america taiwan japan iraq afghanistan syria saudi arabia hamas".split())
NON_ENTITY_CAPS = frozenset("thank thanks welcome hello good great well now so yes no let ladies gentlemen everybody folks listen look remember first second third fourth fifth mr mrs senator president governor".split())


def _canonical(token: str) -> str:
    word = token.casefold().replace("’", "'").strip("'")
    if word in ALIASES:
        return ALIASES[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list[tuple[str, str, int, int]]:
    return [(_canonical(match.group()), match.group(), match.start(), match.end()) for match in WORD.finditer(text)]


def extract_headline_ideas(sources: list[dict[str, Any]], as_of: str, *, max_terms: int = 3) -> list[dict[str, Any]]:
    """Extract scored 1-3 word candidate ideas from pre-cutoff headlines."""
    try:
        cutoff = date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise ValueError("as_of must be an ISO date (YYYY-MM-DD)") from None
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    earliest = cutoff
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each context source must be an object")
        title = source.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("each context source needs a nonempty title")
        raw_date = source.get("published_at") or source.get("published_date")
        try:
            published = date.fromisoformat(str(raw_date)[:10])
        except (TypeError, ValueError):
            raise ValueError(f"invalid publication date for context source: {title}") from None
        if published >= cutoff:
            raise ValueError(f"same-day or later source is ineligible: {title}")
        earliest = min(earliest, published)
        url = source.get("source_url") or source.get("url") or source.get("source_id")
        if not isinstance(url, str) or not url:
            raise ValueError("each context source needs a URL or source ID")
        publisher = str(source.get("publisher", "")).strip()
        content = [token for token, _, _, _ in _tokens(title) if token not in STOPWORDS and len(token) > 1]
        surfaces = [surface for _, surface, _, _ in _tokens(title) if _canonical(surface) not in STOPWORDS and len(_canonical(surface)) > 1]
        for size in range(1, min(max_terms, len(content)) + 1):
            for start in range(len(content) - size + 1):
                phrase = tuple(content[start:start + size])
                if size == 1 and len(phrase[0]) < 5:
                    continue
                bucket = grouped.setdefault(phrase, {"sources": {}, "surfaces": defaultdict(int)})
                bucket["sources"].setdefault(url, {
                    "title": title.strip(),
                    "publisher": publisher,
                    "published_at": published.isoformat(),
                    "url": url,
                })
                bucket["surfaces"][" ".join(surfaces[start:start + size])] += 1
    lookback = max(1, (cutoff - earliest).days)
    ideas = []
    for phrase, bucket in grouped.items():
        sources_for_idea = sorted(
            bucket["sources"].values(),
            key=lambda item: (item["published_at"], item["title"]), reverse=True,
        )
        publishers = {item["publisher"].casefold() for item in sources_for_idea if item["publisher"]}
        freshness = sum(
            1 - (cutoff - date.fromisoformat(item["published_at"])).days / lookback
            for item in sources_for_idea
        ) / len(sources_for_idea)
        ideas.append({
            "idea_id": " ".join(phrase),
            "candidate_idea": max(bucket["surfaces"].items(), key=lambda item: (item[1], item[0]))[0],
            "normalized_terms": list(phrase),
            "term_count": len(phrase),
            "evidence_score": round(
                math.log1p(len(sources_for_idea)) + 0.25 * math.log1p(len(publishers)) + 0.25 * freshness, 6,
            ),
            "score_type": "source_support_heuristic_not_probability",
            "source_count": len(sources_for_idea),
            "publisher_count": len(publishers),
            "supporting_sources": sources_for_idea,
        })
    return sorted(ideas, key=lambda idea: (-idea["evidence_score"], -idea["term_count"], idea["idea_id"]))


def _prepare_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for turn in turns:
        text = turn.get("text") or turn.get("raw_text")
        turn_id = turn.get("turn_id")
        if not isinstance(text, str) or not isinstance(turn_id, str):
            continue
        tokens = _tokens(text)
        indexes: dict[str, list[int]] = defaultdict(list)
        for index, (canonical, _, _, _) in enumerate(tokens):
            indexes[canonical].append(index)
        prepared.append({"text": text, "turn_id": turn_id, "tokens": tokens, "indexes": indexes})
    return prepared


def _match_prepared_idea(idea: dict[str, Any], prepared_turns: list[dict[str, Any]], max_span_tokens: int) -> list[dict[str, str]]:
    terms = idea.get("normalized_terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("idea needs normalized_terms")
    unique_terms = set(terms)
    matches = []
    for turn in prepared_turns:
        indexes, tokens, text = turn["indexes"], turn["tokens"], turn["text"]
        if any(term not in indexes for term in unique_terms):
            continue
        term_positions = sorted({position for term in unique_terms for position in indexes[term]})
        for start in term_positions:
            limit = start + max_span_tokens
            if all(any(start <= occurrence < limit for occurrence in indexes[term]) for term in unique_terms):
                positions = [
                    occurrence for term in unique_terms for occurrence in indexes[term]
                    if start <= occurrence < limit
                ]
                end = max(positions)
                start_char, end_char = tokens[start][2], tokens[end][3]
                snippet = text[max(0, start_char - 45):min(len(text), end_char + 65)].strip()
                matches.append({"turn_id": turn["turn_id"], "matched_text": snippet})
                break
        if len(matches) >= 3:
            break
    return matches


def match_idea_to_turns(idea: dict[str, Any], turns: list[dict[str, Any]], *, max_span_tokens: int = 8) -> list[dict[str, str]]:
    """Find target-speaker turns containing every idea term within one short span."""
    return _match_prepared_idea(idea, _prepare_turns(turns), max_span_tokens)


def extract_transcript_ideas(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mine repeated, cited content-word phrases from target-speaker turns."""
    speech_fillers = {
        "people", "person", "country", "countries", "thing", "things", "time", "times",
        "know", "look", "think", "believe", "want", "going", "go", "got", "get",
        "make", "made", "come", "came", "say", "said", "see", "really", "very",
        "great", "good", "bad", "big", "much", "many", "lot", "kind", "way",
    }
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for turn in turns:
        text = turn.get("text") or turn.get("raw_text")
        turn_id = turn.get("turn_id")
        if not isinstance(text, str) or not isinstance(turn_id, str):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            words = [item for item in _tokens(sentence) if item[0] not in STOPWORDS and item[0] not in speech_fillers and len(item[0]) > 1]
            for size in range(2, min(4, len(words)) + 1):
                for start in range(len(words) - size + 1):
                    span = words[start:start + size]
                    if len({item[0] for item in span}) != size:
                        continue
                    has_entity = any(
                        item[0] in KNOWN_ENTITIES
                        or (item[1][:1].isupper() and item[2] > 0 and item[0] not in NON_ENTITY_CAPS)
                        for item in span
                    )
                    if not has_entity:
                        continue
                    phrase = tuple(sorted(item[0] for item in span))
                    surface = " ".join(item[1] for item in span)
                    bucket = grouped.setdefault(phrase, {"count": 0, "turn_ids": [], "surfaces": defaultdict(int)})
                    bucket["count"] += 1
                    if turn_id not in bucket["turn_ids"]:
                        bucket["turn_ids"].append(turn_id)
                    bucket["surfaces"][surface] += 1
    return [
        {
            "idea_id": " ".join(phrase),
            "candidate_idea": max(bucket["surfaces"].items(), key=lambda item: (item[1], item[0]))[0],
            "normalized_terms": list(phrase),
            "term_count": len(phrase),
            "occurrence_count": bucket["count"],
            "supporting_turn_ids": bucket["turn_ids"],
        }
        for phrase, bucket in sorted(grouped.items())
    ]


def _target_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    target = str(record.get("target_speaker", "")).casefold()
    return [
        turn for turn in record.get("turns", [])
        if isinstance(turn, dict)
        and (turn.get("speaker_role") == "target_speaker" or (
            target and str(turn.get("speaker_name", "")).casefold() == target
        ))
        and isinstance(turn.get("text") or turn.get("raw_text"), str)
        and (turn.get("text") or turn.get("raw_text", "")).strip()
    ]


def evaluate_headline_idea_overlap(
    snapshots_path: str | Path,
    event_records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    top_k: tuple[int, ...] = (10, 25, 50),
) -> dict[str, Any]:
    """Compare headline-derived issue phrases with target-speaker transcript mentions."""
    snapshots = read_jsonl(snapshots_path)
    records = read_jsonl(event_records_path)
    snapshot_by_id = {row.get("event_id"): row for row in snapshots}
    records_by_id = {row.get("event_id"): row for row in records}
    if len(snapshot_by_id) != len(snapshots) or len(records_by_id) != len(records):
        raise ValueError("duplicate event IDs in context or transcript inputs")
    if not snapshots or not set(snapshot_by_id).issubset(records_by_id):
        raise ValueError("every context snapshot needs a matching transcript record")
    if any(k <= 0 for k in top_k):
        raise ValueError("top_k values must be positive")
    event_rows = []
    for event_id, snapshot in snapshot_by_id.items():
        record = records_by_id[event_id]
        if snapshot.get("event_date") != record.get("event_date"):
            raise ValueError(f"event-date mismatch for {event_id}")
        turns = _target_turns(record)
        if not turns:
            raise ValueError(f"no target-speaker turns for {event_id}")
        sources = snapshot.get("candidate_sources", [])
        ideas = extract_headline_ideas(sources, snapshot["event_date"])
        prepared_turns = _prepare_turns(turns)
        for idea in ideas:
            idea["transcript_matches"] = _match_prepared_idea(idea, prepared_turns, 8)
            idea["mentioned_by_target"] = bool(idea["transcript_matches"])
        specific = [idea for idea in ideas if idea["term_count"] >= 2]
        prepared_headlines = _prepare_turns([
            {"turn_id": source.get("source_url") or source.get("url") or f"source-{index}", "text": source["title"]}
            for index, source in enumerate(sources)
        ])
        transcript_ideas = extract_transcript_ideas(turns)
        source_by_url = {
            source.get("source_url") or source.get("url") or f"source-{index}": source
            for index, source in enumerate(sources)
        }
        for idea in transcript_ideas:
            idea["target_salience_score"] = round(
                math.log1p(idea["occurrence_count"]) * (1 + 0.2 * (idea["term_count"] - 2)), 6,
            )
            headline_matches = _match_prepared_idea(idea, prepared_headlines, 8)
            idea["context_matches"] = [
                {"title": source_by_url[match["turn_id"]]["title"],
                 "publisher": source_by_url[match["turn_id"]].get("publisher", ""),
                 "published_at": source_by_url[match["turn_id"]].get("published_at"),
                 "url": match["turn_id"]}
                for match in headline_matches if match["turn_id"] in source_by_url
            ]
            idea["supported_by_pre_event_headline"] = bool(idea["context_matches"])
        transcript_ideas.sort(key=lambda idea: (-idea["target_salience_score"], -idea["term_count"], idea["idea_id"]))
        topical_ideas = [
            idea for idea in transcript_ideas
            if {"ukraine", "war"}.issubset(idea["normalized_terms"])
            or {"russia", "war"}.issubset(idea["normalized_terms"])
            or {"ukraine", "russia"}.issubset(idea["normalized_terms"])
        ]
        event_rows.append({
            "event_id": event_id,
            "event_date": snapshot["event_date"],
            "context_retrieved_at": snapshot.get("retrieved_at"),
            "context_is_exploratory": snapshot.get("exploratory_only") is True,
            "idea_candidate_count": len(ideas),
            "multiword_candidate_count": len(specific),
            "multiword_candidates_mentioned": sum(idea["mentioned_by_target"] for idea in specific),
            "top_ideas": specific[:max(top_k)],
            "all_idea_candidates": ideas,
            "transcript_idea_count": len(transcript_ideas),
            "transcript_top_ideas": transcript_ideas[:max(top_k)],
            "transcript_top_ideas_with_headline_support": sum(idea["supported_by_pre_event_headline"] for idea in transcript_ideas[:max(top_k)]),
            "war_ukraine_russia_transcript_ideas": topical_ideas,
        })
    event_rows.sort(key=lambda row: (row["event_date"], row["event_id"]))
    summaries = {}
    for k in top_k:
        selected = [idea for row in event_rows for idea in row["top_ideas"][:k]]
        target_ideas = [idea for row in event_rows for idea in row["transcript_top_ideas"][:k]]
        summaries[f"top_{k}"] = {
            "events": len(event_rows),
            "headline_candidate_phrases": len(selected),
            "headline_phrases_mentioned_by_target": sum(idea["mentioned_by_target"] for idea in selected),
            "headline_phrase_mention_precision": sum(idea["mentioned_by_target"] for idea in selected) / len(selected) if selected else 0.0,
            "transcript_target_phrases": len(target_ideas),
            "target_phrases_supported_by_headline": sum(idea["supported_by_pre_event_headline"] for idea in target_ideas),
            "target_phrase_headline_coverage": sum(idea["supported_by_pre_event_headline"] for idea in target_ideas) / len(target_ideas) if target_ideas else 0.0,
        }
    report = {
        "schema_version": 1,
        "report_type": "headline-idea-to-transcript-lexical-overlap",
        "task": "measure whether pre-event headline phrases are echoed near-literally in the target speaker transcript",
        "idea_source": "open-vocabulary 1-3 word phrases from dated context headlines; no hand-written topic taxonomy",
        "target_idea_source": "target-speaker transcript-derived 2-4 term phrases anchored on a recognized entity/proper-name token, ranked by within-speech frequency and phrase length",
        "outcome_source": "target-speaker transcript turns; headline candidate mentions require all normalized terms within an eight-token window in one turn; target phrase coverage checks the same window in one headline",
        "normalization": {"stopword_filter": True, "simple_plural_normalization": True, "nationality_aliases": ALIASES, "max_span_tokens": 8},
        "ranking": "source count + publisher diversity + recency heuristic; score is not a probability",
        "context_warning": "Legacy snapshots were retrieved retrospectively; this is an exploratory lexical-coverage audit, not a prospective forecast evaluation.",
        "input_sha256": {
            "snapshots": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
            "event_records": hashlib.sha256(Path(event_records_path).read_bytes()).hexdigest(),
        },
        "summary_by_top_k": summaries,
        "events": event_rows,
        "limitations": [
            "Lexical overlap is not semantic idea equivalence; it can miss paraphrases and count incidental co-mentions.",
            "Phrase candidates are generated from headlines and outcomes from transcripts, with no human labels.",
            "The 20 context snapshots were retrieved after the events and cannot demonstrate prospective lift.",
            "Headline publication dates are day-level and their pre-event public availability is not independently verified.",
        ],
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing overlap report: {output}")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

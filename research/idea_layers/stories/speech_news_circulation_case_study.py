"""Backwards speech-to-news circulation study on local article/post archives.

Requires optional DuckDB (`uv run --with duckdb ...`). Output stays local under
Git-ignored data/source-archives. Headline/post matches are query-based signals,
not semantic labels or social virality measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
OUTCOMES_DEFAULT = ROOT / "data/source-archives/2019-11-04-lexington-rally-outcomes-codex-v1.jsonl"
ARCHIVE_DEFAULT = ROOT / "data/source-archives/all-the-news-2-1-hf-a1f7e19b"
TWEETS_DEFAULT = ROOT / "data/source-archives/markhershey-complete-trump-tweets-8be27c439d0c"
OUTPUT_DEFAULT = ROOT / "data/source-archives/speech-content/stories/2019-11-04-lexington-news-circulation-v1.jsonl"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_term_plan(value: Any, idea_ids: set[str]) -> dict[str, list[str]]:
    allowed_keys = {"terms_by_idea", "schema_version", "artifact_type", "event_id", "provider", "model", "prompt_version", "manual_additions"}
    if not isinstance(value, dict) or set(value) - allowed_keys or not isinstance(value.get("terms_by_idea"), list):
        raise ValueError("term plan must contain a terms_by_idea list and known provenance fields only")
    result: dict[str, list[str]] = {}
    for row in value["terms_by_idea"]:
        if not isinstance(row, dict) or set(row) != {"idea_id", "terms"}:
            raise ValueError("each term-plan row must contain idea_id and terms")
        idea_id, terms = row["idea_id"], row["terms"]
        if idea_id not in idea_ids or idea_id in result:
            raise ValueError(f"unknown or repeated idea ID in term plan: {idea_id}")
        if not isinstance(terms, list) or not 1 <= len(terms) <= 8:
            raise ValueError(f"{idea_id}: expected 1–8 narrow search phrases")
        cleaned = []
        for term in terms:
            if not isinstance(term, str) or not term.strip() or len(term.strip()) > 80:
                raise ValueError(f"{idea_id}: invalid search phrase")
            term = " ".join(term.casefold().split())
            if term not in cleaned:
                cleaned.append(term)
        result[idea_id] = cleaned
    if set(result) != idea_ids:
        raise ValueError(f"term plan missing IDs: {sorted(idea_ids-set(result))}")
    return result


def _regex(terms: list[str]) -> str:
    phrases = sorted({re.escape(term) for term in terms}, key=len, reverse=True)
    return r"\b(?:" + "|".join(phrases) + r")\b"


def _evidence_examples(hits: list[tuple[Any, ...]], stream: str, terms: list[str]) -> list[dict[str, Any]]:
    examples = []
    for item in hits[:5]:
        if stream == "multi_outlet_news_headlines":
            text, publisher, author, url = item[1], item[2], item[3], item[4]
            published_at = item[0]
        else:
            text, publisher, author, url = item[2], "@realDonaldTrump archive account", None, item[3]
            published_at = item[1]
        examples.append({
            "published_at": published_at,
            "text_or_headline": text,
            "publisher_or_account": publisher,
            "byline": author,
            "source_url": url,
            "matched_search_phrases": [term for term in terms if re.search(_regex([term]), text.casefold())],
        })
    return examples


def poisson_burst_states(counts: list[int], exposures: list[int], *, levels: int = 5, rate_multiplier: float = 2.0) -> list[int]:
    """Exposure-adjusted Poisson rate states over the complete available series.

    This detects data-driven rate bursts without imposing a fixed lookback
    window. It is an exploratory signal detector, not a significance test.
    """
    import math

    if len(counts) != len(exposures) or not counts:
        raise ValueError("counts and exposures must have equal nonzero length")
    if any(count < 0 for count in counts) or any(exposure <= 0 for exposure in exposures):
        raise ValueError("counts must be nonnegative and exposures positive")
    base_rate = sum(counts) / sum(exposures)
    if base_rate == 0:
        return [0] * len(counts)
    rates = [base_rate * (rate_multiplier ** level) for level in range(levels)]
    transition_cost = math.log(max(len(counts), 2))
    previous = [
        rates[state] * exposures[0]
        - counts[0] * math.log(max(rates[state] * exposures[0], 1e-12))
        for state in range(levels)
    ]
    backpointers: list[list[int]] = []
    for index in range(1, len(counts)):
        current, back = [], []
        for state, rate in enumerate(rates):
            expected = rate * exposures[index]
            emission = expected - counts[index] * math.log(max(expected, 1e-12))
            options = [previous[prior] + transition_cost * abs(state - prior) for prior in range(levels)]
            prior_state = min(range(levels), key=options.__getitem__)
            current.append(options[prior_state] + emission)
            back.append(prior_state)
        previous = current
        backpointers.append(back)
    state = min(range(levels), key=previous.__getitem__)
    path = [state]
    for back in reversed(backpointers):
        state = back[state]
        path.append(state)
    return list(reversed(path))


def _load_tables(con: Any, archives: Path, tweets_dir: Path, cutoff: str) -> dict[str, int]:
    glob = (archives / "*.parquet").resolve().as_posix().replace("'", "''")
    con.execute(f"""
        CREATE TEMP TABLE news AS
        SELECT try_cast(substr(date, 1, 10) AS DATE) AS day,
               trim(title) AS title,
               coalesce(nullif(trim(publication), ''), 'unknown publisher') AS publisher,
               nullif(trim(author), '') AS author,
               nullif(trim(url), '') AS url
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE try_cast(substr(date, 1, 10) AS DATE) IS NOT NULL
          AND try_cast(substr(date, 1, 10) AS DATE) < DATE '{cutoff}'
          AND nullif(trim(title), '') IS NOT NULL
        QUALIFY row_number() OVER (
          PARTITION BY coalesce(nullif(trim(url), ''), sha256(date || '|' || title || '|' || publication))
          ORDER BY date DESC, title, publication
        ) = 1
    """)
    before = str((tweets_dir / "before.csv").resolve()).replace("'", "''")
    office = str((tweets_dir / "in-office.csv").resolve()).replace("'", "''")
    con.execute(f"""
        CREATE TEMP TABLE trump_posts AS
        SELECT try_cast(substr(trim("Time"), 1, 10) AS DATE) AS day,
               trim("Time") AS published_at,
               trim("Tweet Text") AS text,
               nullif(trim("Tweet URL"), '') AS url
        FROM read_csv_auto(['{before}', '{office}'], all_varchar=true, union_by_name=true)
        WHERE try_cast(substr(trim("Time"), 1, 10) AS DATE) < DATE '{cutoff}'
          AND nullif(trim("Tweet Text"), '') IS NOT NULL
        QUALIFY row_number() OVER (
          PARTITION BY nullif(trim("Tweet URL"), '')
          ORDER BY trim("Time") DESC, trim("Tweet Text")
        ) = 1 OR nullif(trim("Tweet URL"), '') IS NULL
    """)
    return {
        "news_rows": con.execute("SELECT count(*) FROM news").fetchone()[0],
        "tweet_rows": con.execute("SELECT count(*) FROM trump_posts").fetchone()[0],
    }


def run(
    *, event_id: str, events_path: Path, outcomes_path: Path,
    archives_dir: Path, tweets_dir: Path, terms_path: Path, output_path: Path,
) -> dict[str, Any]:
    import duckdb  # optional runtime dependency, intentionally not in project lock

    events = {row["event_id"]: row for row in _jsonl(events_path)}
    outcomes = {row["event_id"]: row for row in _jsonl(outcomes_path)}
    if event_id not in events or event_id not in outcomes:
        raise ValueError("event must exist in both transcript corpus and outcomes file")
    event, outcome = events[event_id], outcomes[event_id]
    event_day = date.fromisoformat(event["event_date"])
    if outcome.get("status") != "validated_candidate":
        raise ValueError("target outcome row must be structurally validated")
    ideas = outcome.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("validated outcome must contain ideas")
    term_plan = validate_term_plan(json.loads(terms_path.read_text(encoding="utf-8")), {i["idea_id"] for i in ideas})
    output_path = output_path.resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite case-study output {output_path}")

    con = duckdb.connect(":memory:")
    try:
        corpus_counts = _load_tables(con, archives_dir, tweets_dir, event_day.isoformat())
        news_calendar = {
            row[0].isoformat(): row[1]
            for row in con.execute("SELECT day, count(*) FROM news GROUP BY day").fetchall()
        }
        post_calendar = {
            row[0].isoformat(): row[1]
            for row in con.execute("SELECT day, count(*) FROM trump_posts GROUP BY day").fetchall()
        }
        rows = []
        signal_summaries = []
        for idea in ideas:
            idea_id = idea["idea_id"]
            pattern = _regex(term_plan[idea_id])
            news_records = con.execute("""
                SELECT CAST(day AS VARCHAR), title, publisher, author, url
                FROM news WHERE regexp_matches(lower(title), ?)
            """, [pattern]).fetchall()
            post_records = con.execute("""
                SELECT CAST(day AS VARCHAR), published_at, text, url
                FROM trump_posts WHERE regexp_matches(lower(text), ?)
            """, [pattern]).fetchall()
            news_by_day: dict[str, list[tuple[Any, ...]]] = {}
            for record in news_records:
                news_by_day.setdefault(record[0], []).append(record)
            posts_by_day: dict[str, list[tuple[Any, ...]]] = {}
            for record in post_records:
                posts_by_day.setdefault(record[0], []).append(record)
            for stream, grouped, calendar in (
                ("multi_outlet_news_headlines", news_by_day, news_calendar),
                ("donald_trump_twitter_posts", posts_by_day, post_calendar),
            ):
                for day, total in calendar.items():
                    hits = grouped.get(day, [])
                    count = len(hits)
                    if not count:
                        continue
                    if stream == "multi_outlet_news_headlines":
                        publishers = {item[2] for item in hits if item[2]}
                        authors = {item[3] for item in hits if item[3]}
                        urls = {item[4] for item in hits if item[4]}
                    else:
                        publishers, authors, urls = {"@realDonaldTrump archive account"}, set(), {item[3] for item in hits if item[3]}
                    rows.append({
                        "event_id": event_id,
                        "idea_id": idea_id,
                        "idea_concept": idea.get("concept"),
                        "idea_prominence": idea.get("prominence"),
                        "stream": stream,
                        "day": day,
                        "matching_items": count,
                        "corpus_items_that_day": total,
                        "attention_share": count / total if total else None,
                        "distinct_publishers": len(publishers),
                        "distinct_bylines": len(authors),
                        "distinct_urls": len(urls),
                        "search_phrases": term_plan[idea_id],
                        "example_titles": [item[1] if stream == "multi_outlet_news_headlines" else item[2] for item in hits[:5]],
                        "evidence_examples": _evidence_examples(hits, stream, term_plan[idea_id]),
                        "example_bylines": sorted(authors)[:10],
                        "example_publishers": sorted(publishers)[:10],
                        "visible_retweet_handles": sorted({
                            handle for item in hits if stream == "donald_trump_twitter_posts"
                            for handle in re.findall(r"\bRT\s+@([A-Za-z0-9_]+)", item[2])
                        })[:10],
                    })
                ordered_days = sorted(calendar)
                daily_counts = [len(grouped.get(day, [])) for day in ordered_days]
                exposures = [calendar[day] for day in ordered_days]
                states = poisson_burst_states(daily_counts, exposures)
                episodes = []
                start_index = None
                for index, state in enumerate(states + [0]):
                    if state > 0 and start_index is None:
                        start_index = index
                    if state == 0 and start_index is not None:
                        end_index = index - 1
                        episode_days = ordered_days[start_index:end_index + 1]
                        episode_hits = [item for day in episode_days for item in grouped.get(day, [])]
                        if episode_hits:
                            if stream == "multi_outlet_news_headlines":
                                episode_publishers = Counter(item[2] for item in episode_hits if item[2])
                                episode_authors = Counter(item[3] for item in episode_hits if item[3])
                                episode_retweet_handles = Counter()
                            else:
                                episode_publishers = Counter({"@realDonaldTrump archive account": len(episode_hits)})
                                episode_authors = Counter()
                                episode_retweet_handles = Counter(
                                    handle for item in episode_hits
                                    for handle in re.findall(r"\bRT\s+@([A-Za-z0-9_]+)", item[2])
                                )
                            peak_index = max(range(start_index, end_index + 1), key=lambda pos: daily_counts[pos] / exposures[pos])
                            episodes.append({
                                "start_day": ordered_days[start_index],
                                "end_day": ordered_days[end_index],
                                "max_burst_state": max(states[start_index:end_index + 1]),
                                "active_day_count": sum(daily_counts[pos] > 0 for pos in range(start_index, end_index + 1)),
                                "matched_item_count": len(episode_hits),
                                "distinct_publishers_or_accounts": len(episode_publishers),
                                "distinct_news_bylines": len(episode_authors),
                                "top_visible_retweet_handles": episode_retweet_handles.most_common(10),
                                "peak_day": ordered_days[peak_index],
                                "peak_day_item_count": daily_counts[peak_index],
                                "peak_day_attention_share": daily_counts[peak_index] / exposures[peak_index],
                                "top_publishers": episode_publishers.most_common(5),
                                "top_bylines": episode_authors.most_common(5),
                            })
                        start_index = None
                episodes.sort(key=lambda item: (item["max_burst_state"], item["matched_item_count"]), reverse=True)
                signal_summaries.append({
                    "idea_id": idea_id,
                    "stream": stream,
                    "search_phrases": term_plan[idea_id],
                    "first_match_day": min(grouped) if grouped else None,
                    "last_match_day": max(grouped) if grouped else None,
                    "matched_item_count": sum(daily_counts),
                    "active_day_count": sum(count > 0 for count in daily_counts),
                    "burst_episodes": episodes[:10],
                    "burst_model": "exposure-adjusted Poisson state path; baseline fit over full archive-to-cutoff series; state multiplier 2; transition penalty log(number of days); five states",
                })
        rows.sort(key=lambda r: (r["idea_id"], r["stream"], r["day"]))
        output_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
        source_files = sorted(archives_dir.glob("*.parquet"))
        tweet_files = [tweets_dir / "before.csv", tweets_dir / "in-office.csv"]
        manifest = {
            "schema_version": 1,
            "artifact_type": "backwards-speech-news-circulation-case-study",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "event_date": event_day.isoformat(),
            "speech_day_sources_included": False,
            "reason_speech_time_unavailable": "Speech date is known but a verifiable local start time is unavailable; all same-day input records are excluded.",
            "idea_count": len(ideas),
            "mention_count": len(outcome.get("mentions", [])),
            "forecastable_idea_count": sum(item.get("forecastable") is True for item in ideas),
            "outcome_model": outcome.get("model"),
            "outcome_provider": outcome.get("provider"),
            "outcome_sha256": _sha256(outcomes_path),
            "event_corpus_sha256": _sha256(events_path),
            "query_term_plan_sha256": _sha256(terms_path),
            "corpus_counts_before_cutoff": corpus_counts,
            "news_archive_day_range_before_cutoff": [min(news_calendar), max(news_calendar)] if news_calendar else None,
            "speaker_post_day_range_before_cutoff": [min(post_calendar), max(post_calendar)] if post_calendar else None,
            "news_archive_files": [{"name": p.name, "sha256": _sha256(p)} for p in source_files],
            "tweet_archive_files": [{"name": p.name, "sha256": _sha256(p)} for p in tweet_files],
            "matched_daily_rows": len(rows),
            "signal_summaries": signal_summaries,
            "social_virality_measured": False,
            "news_circulation_proxy": "headline item volume, attention share of all archived items that day, publisher and byline diversity; this is not audience reach or engagement.",
            "tweet_signal_proxy": "historical account-archive post volume and text only; original/retweet type is missing, engagement counts are absent, and timezone is unspecified.",
            "limitations": [
                "Outcome ideas are assistant-authored retrospective interpretations, not semantic gold.",
                "Search-phrase matches in headlines are weak lexical signals; they miss paraphrases and admit false positives.",
                "All the News 2.1 date coverage ends 2020-04-02, and coverage starts in 2016; this is not the full news universe.",
                "Publication dates and a later snapshot do not prove contemporaneous availability.",
                "Bylines identify article authors, not every person quoted or discussing an issue.",
                "The Trump Twitter archive lacks a reliable original/retweet field, historical engagement counts, and timezone; account-post counts are not author-specific activity or virality.",
                "No social-platform post archive with historical engagement metrics is available for this event, so virality cannot be estimated.",
                "The dataset's underlying article reuse rights are unclear; raw records remain local and must not be redistributed.",
            ],
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        }
    finally:
        con.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_bytes)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output_path), "manifest": str(manifest_path), "idea_count": len(ideas), "daily_rows": len(rows), **corpus_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="dt45-19-8a126a9f")
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES_DEFAULT)
    parser.add_argument("--archives", type=Path, default=ARCHIVE_DEFAULT)
    parser.add_argument("--tweets", type=Path, default=TWEETS_DEFAULT)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(
        event_id=args.event_id, events_path=args.events, outcomes_path=args.outcomes,
        archives_dir=args.archives, tweets_dir=args.tweets, terms_path=args.terms,
        output_path=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()

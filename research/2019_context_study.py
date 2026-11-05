"""Build a local 2016–2019 source-window set and a frozen 2019 backtest.

Optional runtime dependency: DuckDB. Raw archives and outputs stay under the
Git-ignored data/source-archives directory. The experiment uses only titles and
post text; article bodies are not model features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb

from speechprefix.headline_idea_matching import (
    ISSUE_PHRASE_EXTRACTOR_VERSION,
    _target_turns,
    extract_transcript_ideas,
    issue_phrase_lexicon_sha256,
)
from speechprefix.stages.ideas.model import _features, _fit, _label_names, _metrics, _predict

EVENTS_DEFAULT = ROOT / "data/rollcall-by-type/trump-speech-candidates.jsonl"
ARCHIVES_DEFAULT = ROOT / "data/source-archives"
OUTPUT_DEFAULT = ARCHIVES_DEFAULT / "2019-full-year-context-study-v1"
TRAIN_YEARS = {"2016", "2017", "2018"}
TARGET_YEAR = "2019"
MODEL_CONFIG = {
    "alpha": 4.0, "min_df": 3, "temperature": 8.0,
    "headlines_per_source": 20, "headline_selection": "latest",
}
TUNE_ALPHAS = (0.5, 4.0)
TUNE_MIN_DFS = (1, 3)
TUNE_TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
PROFILES = {
    "abc_only": ("abc_headlines",),
    "all_the_news_only": ("all_the_news_2_1",),
    "news_two_source": ("abc_headlines", "all_the_news_2_1"),
    "trump_twitter_only": ("trump_twitter",),
    "news_plus_trump_twitter": ("abc_headlines", "all_the_news_2_1", "trump_twitter"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _query_rows(con: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    result = con.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def _load_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected = []
    all_by_id = {}
    for event in _jsonl(path):
        event_id = event.get("event_id")
        event_date = event.get("event_date")
        if not isinstance(event_id, str) or not isinstance(event_date, str):
            continue
        if event_date[:4] not in TRAIN_YEARS | {TARGET_YEAR} or event_date < "2016-01-08":
            continue
        if event_id in all_by_id:
            raise ValueError(f"duplicate event ID: {event_id}")
        date.fromisoformat(event_date)
        all_by_id[event_id] = event
        selected.append({"event_id": event_id, "event_date": event_date})
    selected.sort(key=lambda row: (row["event_date"], row["event_id"]))
    return selected, all_by_id


def _create_source_tables(con: duckdb.DuckDBPyConnection, archives: Path) -> dict[str, dict[str, Any]]:
    abc_dir = archives / "abc-million-headlines-2003-2021"
    allnews_dir = archives / "all-the-news-2-1-hf-a1f7e19b"
    cc_dir = archives / "cc-news-hf-81eb2ce0"
    tweet_dir = archives / "markhershey-complete-trump-tweets-8be27c439d0c"
    for required in (abc_dir, allnews_dir, cc_dir, tweet_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)

    abc_csv = _sql_path(abc_dir / "abcnews-date-text_train.csv")
    con.execute(f"""
        CREATE TEMP TABLE abc_headlines AS
        SELECT
            try_strptime(trim(publish_date), '%Y%m%d')::DATE AS published_day,
            strftime(try_strptime(trim(publish_date), '%Y%m%d')::DATE, '%Y-%m-%d') AS published_at,
            trim(headline_text) AS title,
            'ABC News Australia' AS publisher,
            'abc://million-headlines/' || sha256(coalesce(trim(publish_date), '') || '|' || coalesce(trim(headline_text), '')) AS source_id,
            NULL::VARCHAR AS source_url
        FROM read_csv_auto('{abc_csv}', all_varchar=true)
        WHERE try_strptime(trim(publish_date), '%Y%m%d') IS NOT NULL
          AND nullif(trim(headline_text), '') IS NOT NULL
    """)

    allnews_glob = _sql_path(allnews_dir / "*.parquet")
    con.execute(f"""
        CREATE TEMP TABLE all_the_news_2_1 AS
        SELECT
            try_cast(substr(date, 1, 10) AS DATE) AS published_day,
            date AS published_at,
            trim(title) AS title,
            coalesce(nullif(trim(publication), ''), 'Unknown publisher') AS publisher,
            nullif(trim(url), '') AS source_url,
            coalesce(nullif(trim(url), ''), 'dataset://all-the-news-2-1/' || sha256(
                coalesce(date, '') || '|' || coalesce(title, '') || '|' || coalesce(publication, '')
            )) AS source_id
        FROM read_parquet('{allnews_glob}', union_by_name=true)
        WHERE try_cast(substr(date, 1, 10) AS DATE) IS NOT NULL
          AND nullif(trim(title), '') IS NOT NULL
    """)

    tweet_paths = [_sql_path(tweet_dir / "before.csv"), _sql_path(tweet_dir / "in-office.csv")]
    tweet_union = ", ".join(f"'{path}'" for path in tweet_paths)
    con.execute(f"""
        CREATE TEMP TABLE trump_twitter AS
        SELECT published_day, raw_time AS published_at, title, 'Donald Trump Twitter archive' AS publisher,
               source_url, coalesce(source_url, 'dataset://trump-twitter/' || sha256(raw_time || '|' || title)) AS source_id
        FROM (
            SELECT
                try_cast(substr(trim(\"Time\"), 1, 10) AS DATE) AS published_day,
                trim(\"Time\") AS raw_time,
                trim(\"Tweet Text\") AS title,
                nullif(trim(\"Tweet URL\"), '') AS source_url,
                row_number() OVER (
                    PARTITION BY nullif(trim(\"Tweet URL\"), '')
                    ORDER BY trim(\"Time\") DESC, trim(\"Tweet Text\")
                ) AS duplicate_url_rank
            FROM read_csv_auto([{tweet_union}], all_varchar=true, union_by_name=true)
        )
        WHERE published_day IS NOT NULL AND nullif(title, '') IS NOT NULL
          AND (source_url IS NULL OR duplicate_url_rank = 1)
    """)
    # Measure the retained CC-News snapshot independently. Its actual 2019
    # date coverage is not assumed from the broader dataset-card claim.
    cc_glob = _sql_path(cc_dir / "*.parquet")
    con.execute(f"""
        CREATE TEMP TABLE cc_news AS
        SELECT try_cast(substr(date, 1, 10) AS DATE) AS published_day,
               date AS published_at, trim(title) AS title,
               coalesce(nullif(trim(domain), ''), 'Unknown domain') AS publisher,
               nullif(trim(url), '') AS source_url,
               coalesce(nullif(trim(url), ''), 'dataset://cc-news/' || sha256(
                   coalesce(date, '') || '|' || coalesce(title, '') || '|' || coalesce(domain, '')
               )) AS source_id
        FROM read_parquet('{cc_glob}', union_by_name=true)
        WHERE try_cast(substr(date, 1, 10) AS DATE) IS NOT NULL
          AND nullif(trim(title), '') IS NOT NULL
    """)

    return {
        "abc_headlines": {"table": "abc_headlines", "dataset_id": "ABC News Australia / A Million News Headlines"},
        "all_the_news_2_1": {"table": "all_the_news_2_1", "dataset_id": "rjac/all-the-news-2-1-Component-one"},
        "trump_twitter": {"table": "trump_twitter", "dataset_id": "MarkHershey/CompleteTrumpTweetsArchive"},
        "cc_news": {"table": "cc_news", "dataset_id": "vblagoje/cc_news"},
    }


def _event_table(con: duckdb.DuckDBPyConnection, events: list[dict[str, Any]]) -> None:
    con.execute("CREATE TEMP TABLE target_events(event_id VARCHAR, event_date DATE)")
    con.executemany(
        "INSERT INTO target_events VALUES (?, ?)",
        [(event["event_id"], date.fromisoformat(event["event_date"])) for event in events],
    )


def _source_stats_and_candidates(
    con: duckdb.DuckDBPyConnection, events: list[dict[str, Any]], table: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    stats_rows = _query_rows(con, f"""
        SELECT e.event_id,
               count(s.title)::BIGINT AS source_count,
               count(DISTINCT lower(trim(s.title)))::BIGINT AS distinct_title_count,
               count(DISTINCT nullif(trim(s.source_url), ''))::BIGINT AS distinct_url_count,
               count(DISTINCT nullif(trim(s.publisher), ''))::BIGINT AS publisher_count,
               min(s.published_day)::VARCHAR AS earliest_source_day,
               max(s.published_day)::VARCHAR AS latest_source_day
        FROM target_events e
        LEFT JOIN {table} s
          ON s.published_day >= e.event_date - INTERVAL 7 DAY
         AND s.published_day < e.event_date
        GROUP BY e.event_id
    """)
    stats = {row["event_id"]: {
        key: row[key] for key in (
            "source_count", "distinct_title_count", "distinct_url_count", "publisher_count",
            "earliest_source_day", "latest_source_day",
        )
    } for row in stats_rows}

    candidate_rows = _query_rows(con, f"""
        WITH eligible AS (
            SELECT e.event_id, s.title, s.published_at, s.published_day, s.publisher,
                   s.source_url, s.source_id,
                   row_number() OVER (
                       PARTITION BY e.event_id, lower(trim(s.title))
                       ORDER BY s.published_day DESC, s.published_at DESC, s.title, s.source_url
                   ) AS same_title_rank
            FROM target_events e
            JOIN {table} s
              ON s.published_day >= e.event_date - INTERVAL 7 DAY
             AND s.published_day < e.event_date
        ), deduplicated AS (
            SELECT * FROM eligible WHERE same_title_rank = 1
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY event_id
                ORDER BY published_day DESC, published_at DESC, title, source_url
            ) AS headline_rank
            FROM deduplicated
        )
        SELECT event_id, title, published_at, published_day::VARCHAR AS published_day,
               publisher, source_url, source_id
        FROM ranked WHERE headline_rank <= {MODEL_CONFIG['headlines_per_source']}
        ORDER BY event_id, published_day DESC, published_at DESC, title, source_url
    """)
    candidates: dict[str, list[dict[str, Any]]] = {event["event_id"]: [] for event in events}
    for row in candidate_rows:
        candidates[row["event_id"]].append({
            "title": row["title"],
            "published_at": row["published_at"],
            "publisher": row["publisher"],
            "source_id": row["source_id"],
            "source_url": row["source_url"],
            "date_precision": "calendar-day" if table == "abc_headlines" else "timestamp-string-without-verified-timezone",
            "source_dataset": table,
        })
    for event in events:
        stats.setdefault(event["event_id"], {
            "source_count": 0, "distinct_title_count": 0, "distinct_url_count": 0,
            "publisher_count": 0, "earliest_source_day": None, "latest_source_day": None,
        })
    return stats, candidates


def _features_by_profile(
    event_date: str, source_candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    per_source = {}
    for name in ("abc_headlines", "all_the_news_2_1", "trump_twitter"):
        snapshot = {"event_date": event_date, "candidate_sources": source_candidates.get(name, [])}
        per_source[name] = _features(
            snapshot,
            headline_limit=MODEL_CONFIG["headlines_per_source"],
            selection=MODEL_CONFIG["headline_selection"],
        )
    return {
        "abc_only": per_source["abc_headlines"],
        "all_the_news_only": per_source["all_the_news_2_1"],
        "news_two_source": per_source["abc_headlines"] | per_source["all_the_news_2_1"],
        "trump_twitter_only": per_source["trump_twitter"],
        "news_plus_trump_twitter": per_source["abc_headlines"] | per_source["all_the_news_2_1"] | per_source["trump_twitter"],
    }


def _prevalence_rank(labels: list[set[str]]) -> list[tuple[str, float]]:
    counts = Counter(label for target in labels for label in target)
    total = len(labels)
    return sorted(
        ((label, math.log((count + 1) / (total - count + 1))) for label, count in counts.items()),
        key=lambda row: (-row[1], row[0]),
    )


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("precision_at_4", "recall_at_10", "ndcg_at_10", "binary_log_loss", "candidate_coverage", "unseen_target_ideas")
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys} if rows else {}


def _mean_metrics_by_date(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event_date"], []).append(row[field])
    return _mean_metrics([_mean_metrics(day_rows) for day_rows in grouped.values()])


def run(events_path: Path, archives: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite study output directory: {output}")
    output.mkdir(parents=True)
    event_rows, event_records = _load_events(events_path)
    con = duckdb.connect()
    _create_source_tables(con, archives)
    _event_table(con, event_rows)

    stats_by_source = {}
    candidates_by_source = {}
    for source_name, table in (
        ("abc_headlines", "abc_headlines"),
        ("all_the_news_2_1", "all_the_news_2_1"),
        ("trump_twitter", "trump_twitter"),
        ("cc_news", "cc_news"),
    ):
        stats_by_source[source_name], candidates_by_source[source_name] = _source_stats_and_candidates(
            con, event_rows, table,
        )

    labels_by_id: dict[str, set[str]] = {}
    target_ideas_by_id: dict[str, list[dict[str, Any]]] = {}
    feature_docs: dict[str, dict[str, set[str]]] = {}
    snapshot_rows = []
    for event in event_rows:
        event_id, event_date = event["event_id"], event["event_date"]
        source_candidates = {name: candidates_by_source[name][event_id] for name in candidates_by_source}
        for source_name, sources in source_candidates.items():
            for source in sources:
                if date.fromisoformat(str(source["published_at"])[:10]) >= date.fromisoformat(event_date):
                    raise ValueError(f"same-day/future source in {event_id}: {source_name}")
        feature_docs[event_id] = _features_by_profile(event_date, source_candidates)
        window_start = date.fromordinal(date.fromisoformat(event_date).toordinal() - 7)
        outcome = extract_transcript_ideas(_target_turns(event_records[event_id]))
        if outcome:
            labels_by_id[event_id] = {idea["idea_id"] for idea in outcome}
            target_ideas_by_id[event_id] = outcome
        snapshot_rows.append({
            "schema_version": 1,
            "event_id": event_id,
            "event_date": event_date,
            "window_start_inclusive": window_start.isoformat(),
            "window_end_exclusive": event_date,
            "exploratory_only": True,
            "retrieval_profile": "retrospective-local-archive-date-filter-v1",
            "source_window_stats": {name: stats_by_source[name][event_id] for name in stats_by_source},
            "selected_sources": source_candidates,
        })

    train = [event for event in event_rows if event["event_date"][:4] in TRAIN_YEARS and event["event_id"] in labels_by_id]
    targets = [event for event in event_rows if event["event_date"].startswith(TARGET_YEAR) and event["event_id"] in labels_by_id]
    if not train or not targets:
        raise ValueError("no labeled pre-2019 training events or 2019 target events")
    train_labels = [labels_by_id[event["event_id"]] for event in train]
    if max(event["event_date"] for event in train) >= min(event["event_date"] for event in targets):
        raise ValueError("training cutoff overlaps target year")

    evaluations = {}
    model_hashes = {}
    train_dates = sorted({event["event_date"] for event in train})
    validation_start_index = min(len(train_dates) - 1, max(1, int(len(train_dates) * 0.8)))
    validation_start_date = train_dates[validation_start_index]
    config_train = [event for event in train if event["event_date"] < validation_start_date]
    config_validation = [event for event in train if event["event_date"] >= validation_start_date]
    if not config_train or not config_validation:
        raise ValueError("historical validation split produced an empty partition")
    baseline = _prevalence_rank(train_labels)
    baseline_labels = set(dict(baseline))
    label_names = _label_names([
        {"target_ideas": target_ideas_by_id[event["event_id"]]} for event in train
    ])

    for profile, source_names in PROFILES.items():
        documents = [feature_docs[event["event_id"]][profile] for event in train]
        config_train_documents = [feature_docs[event["event_id"]][profile] for event in config_train]
        config_train_labels = [labels_by_id[event["event_id"]] for event in config_train]
        validation_rows = []
        for alpha in TUNE_ALPHAS:
            for min_df in TUNE_MIN_DFS:
                validation_model = _fit(
                    config_train_documents, config_train_labels, alpha=alpha, min_df=min_df,
                )
                validation_known_labels = set(validation_model["models"])
                validation_rankings = [
                    _predict(validation_model, feature_docs[event["event_id"]][profile])
                    for event in config_validation
                ]
                for temperature in TUNE_TEMPERATURES:
                    validation_metrics = [
                        _metrics(
                            [(label, score / temperature) for label, score in ranking],
                            labels_by_id[event["event_id"]],
                            all_labels=validation_known_labels,
                            k=10,
                        )
                        for event, ranking in zip(config_validation, validation_rankings)
                    ]
                    validation_rows.append({
                        "alpha": alpha,
                        "min_df": min_df,
                        "temperature": temperature,
                        "metrics": _mean_metrics(validation_metrics),
                    })
        selected_config = min(validation_rows, key=lambda row: (
            row["metrics"]["binary_log_loss"],
            -row["metrics"]["ndcg_at_10"],
            row["alpha"], row["min_df"], row["temperature"],
        ))
        model = _fit(
            documents, train_labels,
            alpha=selected_config["alpha"], min_df=selected_config["min_df"],
        )
        model["label_names"] = label_names
        all_labels = set(model["models"])
        legacy_model = _fit(
            documents, train_labels,
            alpha=MODEL_CONFIG["alpha"], min_df=MODEL_CONFIG["min_df"],
        )
        legacy_model["label_names"] = label_names
        event_results = []
        for target in targets:
            event_id = target["event_id"]
            target_labels = labels_by_id[event_id]
            raw_ranking = _predict(model, feature_docs[event_id][profile])
            ranked = [(label, score / selected_config["temperature"]) for label, score in raw_ranking]
            model_metrics = _metrics(ranked, target_labels, all_labels=all_labels, k=10)
            legacy_ranked = [
                (label, score / MODEL_CONFIG["temperature"])
                for label, score in _predict(legacy_model, feature_docs[event_id][profile])
            ]
            legacy_metrics = _metrics(
                legacy_ranked, target_labels,
                all_labels=set(legacy_model["models"]), k=10,
            )
            baseline_metrics = _metrics(baseline, target_labels, all_labels=baseline_labels, k=10)
            event_results.append({
                "event_id": event_id,
                "event_date": target["event_date"],
                "training_events": len(train),
                "headline_or_post_counts": {
                    source_name: stats_by_source[source_name][event_id]["source_count"]
                    for source_name in source_names
                },
                "target_idea_count": len(target_labels),
                "candidate_coverage": model_metrics["candidate_coverage"],
                "unseen_target_ideas": int(model_metrics["unseen_target_ideas"]),
                "model_metrics": model_metrics,
                "legacy_frozen_model_metrics": legacy_metrics,
                "prevalence_metrics": baseline_metrics,
                "target_ideas": sorted(target_labels),
                "top_10_predictions": [
                    {"rank": rank, "idea_id": label, "idea": label_names.get(label, label),
                     "logit_div_temperature": round(score, 6), "matched_target": label in target_labels}
                    for rank, (label, score) in enumerate(ranked[:10], 1)
                ],
            })
        model_hashes[profile] = hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evaluations[profile] = {
            "input_sources": list(source_names),
            "training_event_count": len(train),
            "training_date_range": [min(event["event_date"] for event in train), max(event["event_date"] for event in train)],
            "training_unique_idea_label_count": len(all_labels),
            "training_label_observations": sum(len(target) for target in train_labels),
            "feature_vocabulary_size": len(model["vocabulary"]),
            "2019_labeled_target_count": len(targets),
            "2019_labeled_target_distinct_dates": len({event["event_date"] for event in targets}),
            "historical_validation_split": {
                "training_event_count": len(config_train),
                "validation_event_count": len(config_validation),
                "training_date_range": [min(event["event_date"] for event in config_train), max(event["event_date"] for event in config_train)],
                "validation_date_range": [min(event["event_date"] for event in config_validation), max(event["event_date"] for event in config_validation)],
                "selection_rule": "minimum mean validation binary log loss, then higher nDCG@10; validation uses pre-2019 events only",
                "selected_config": selected_config,
                "validation_grid": validation_rows,
            },
            "legacy_frozen_config": MODEL_CONFIG,
            "aggregation_by_event": _mean_metrics([row["model_metrics"] for row in event_results]),
            "aggregation_by_distinct_event_date": _mean_metrics_by_date(event_results, "model_metrics"),
            "legacy_frozen_model_aggregation_by_event": _mean_metrics([row["legacy_frozen_model_metrics"] for row in event_results]),
            "legacy_frozen_model_aggregation_by_distinct_event_date": _mean_metrics_by_date(event_results, "legacy_frozen_model_metrics"),
            "prevalence_reference_aggregation_by_event": _mean_metrics([row["prevalence_metrics"] for row in event_results]),
            "prevalence_reference_aggregation_by_distinct_event_date": _mean_metrics_by_date(event_results, "prevalence_metrics"),
            "per_event": event_results,
        }

    target_inventory = []
    for event in event_rows:
        if event["event_date"].startswith(TARGET_YEAR):
            ideas = target_ideas_by_id.get(event["event_id"], [])
            target_inventory.append({
                "event_id": event["event_id"], "event_date": event["event_date"],
                "has_extracted_outcome": bool(ideas), "extracted_idea_count": len(ideas),
                "distinct_idea_count": len({idea["idea_id"] for idea in ideas}),
            })

    window_report = {
        "schema_version": 1,
        "report_type": "2019-full-year-source-window-coverage",
        "event_corpus_sha256": _sha256(events_path),
        "event_window_count_2016_2019": len(event_rows),
        "2019_target_event_count": len(target_inventory),
        "2019_target_distinct_dates": len({row["event_date"] for row in target_inventory}),
        "2019_target_labeled_events": sum(row["has_extracted_outcome"] for row in target_inventory),
        "2019_labeled_target_distinct_dates": len({row["event_date"] for row in target_inventory if row["has_extracted_outcome"]}),
        "2019_target_unlabeled_events": sum(not row["has_extracted_outcome"] for row in target_inventory),
        "source_window_statistics_2019": {
            name: {
                "events_with_any_source": sum(stats_by_source[name][row["event_id"]]["source_count"] > 0 for row in target_inventory),
                "total_event_window_records": sum(stats_by_source[name][row["event_id"]]["source_count"] for row in target_inventory),
                "mean_records_per_event": sum(stats_by_source[name][row["event_id"]]["source_count"] for row in target_inventory) / max(1, len(target_inventory)),
                "min_records_per_event": min(stats_by_source[name][row["event_id"]]["source_count"] for row in target_inventory),
                "max_records_per_event": max(stats_by_source[name][row["event_id"]]["source_count"] for row in target_inventory),
            } for name in stats_by_source
        },
        "target_event_inventory": target_inventory,
        "date_policy": "Use source publication calendar dates in [event_date - 7 days, event_date); exclude same-day and later material. Timestamps without timezone are treated as date-only for eligibility.",
        "availability_warning": "These archives were retrieved retrospectively. A pre-event publication date does not prove the item was visible, indexed, or available before the speech.",
    }
    evaluation_report = {
        "schema_version": 1,
        "report_type": "2019-calendar-year-context-to-idea-preliminary-ablation",
        "task": "Predict automatically extracted transcript issue phrases for 2019 events using only source text dated in the preceding seven days; train on 2016–2018 paired context/label events with complete source-window coverage.",
        "target_year": 2019,
        "target_policy": f"{ISSUE_PHRASE_EXTRACTOR_VERSION}; weak compact lexical matches, not semantic gold or human annotation.",
        "input_policy": "No target transcript, event title, event location, same-day source, or future source is an input. Model feature documents use up to 20 recent distinct items per source. Twitter text is a separate input ablation, not merged into the headline-only baseline.",
        "training_policy": "Fixed year holdout: labeled 2016–2018 events whose full seven-day windows fall within the All the News 2.1 archive date coverage; no 2019 outcomes are used for fitting or configuration selection.",
        "legacy_frozen_config": MODEL_CONFIG,
        "configuration_selection": "Each source profile tunes alpha, minimum feature document frequency, and temperature on a chronological validation segment of 2016–2018 only; the 2019 target year is excluded from tuning. The legacy v7 settings are also reported as a frozen comparison.",
        "profiles": evaluations,
        "limitations": [
            "Retrospective archive dates do not establish historical visibility or point-in-time availability.",
            "Transcript phrase extraction is incomplete and lexical; scores are diagnostics against that extractor, not semantic accuracy.",
            "The Roll Call candidate corpus may not enumerate every Trump speech in 2019; its local 2019 slice is a retrospective transcript subset.",
            "Although configuration selection uses only pre-2019 validation, 2019 metrics were inspected during this exploratory iteration; they are not an independent or prospective accuracy estimate.",
            "ABC News Australia is a single non-U.S. outlet, and All the News 2.1 has missing URLs/publisher fields and source-selection bias.",
            "Trump tweet timestamps have no timezone and the archive may contain later-deleted posts without deletion times; the Twitter profiles are exploratory only.",
            "CC-News dataset card claims coverage through December 2019, but the retained Hugging Face snapshot has only two rows with 2019 publication dates; it is not included in model profiles.",
            "The HuffPost dataset ends in 2018 and the Stiles Truth Social archive begins in 2022, so neither supplies 2019 input.",
        ],
    }
    for profile in evaluations:
        evaluation_report["profiles"][profile]["model_sha256"] = model_hashes[profile]

    _write_jsonl(output / "event-source-windows-v1.jsonl", snapshot_rows)
    _write_json(output / "2019-target-inventory-v1.json", window_report)
    _write_json(output / "2019-preliminary-model-evaluation-v1.json", evaluation_report)
    archive_manifests = {}
    for name in (
        "abc-million-headlines-2003-2021", "all-the-news-2-1-hf-a1f7e19b",
        "cc-news-hf-81eb2ce0", "markhershey-complete-trump-tweets-8be27c439d0c",
    ):
        manifest = archives / name / "SOURCE-MANIFEST.json"
        if manifest.exists():
            archive_manifests[name] = _sha256(manifest)
    run_manifest = {
        "schema_version": 1,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "study_script_sha256": _sha256(Path(__file__).resolve()),
        "event_corpus_sha256": _sha256(events_path),
        "lexicon_sha256": issue_phrase_lexicon_sha256(),
        "source_manifest_sha256": archive_manifests,
        "output_sha256": {
            name: _sha256(output / name)
            for name in ("event-source-windows-v1.jsonl", "2019-target-inventory-v1.json", "2019-preliminary-model-evaluation-v1.json")
        },
        "outputs_local_only": True,
    }
    _write_json(output / "RUN-MANIFEST.json", run_manifest)
    return {"output": str(output), "target_events": len(target_inventory), "labeled_2019_events": window_report["2019_target_labeled_events"], "profiles": list(evaluations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DEFAULT)
    parser.add_argument("--archives", type=Path, default=ARCHIVES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.events, args.archives, args.output), indent=2))


if __name__ == "__main__":
    main()

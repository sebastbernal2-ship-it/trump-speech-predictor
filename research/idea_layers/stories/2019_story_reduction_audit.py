"""Audit conservative exact-identity redundancy in the local 2019 news sources.

This is a source-reduction diagnostic, not a semantic story clusterer or model
input. It hashes normalized article bodies locally and writes aggregate counts
only. Requires DuckDB; raw corpora and outputs stay Git-ignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[3]
ARCHIVES_DEFAULT = ROOT / "data/source-archives"
OUTPUT_DEFAULT = ARCHIVES_DEFAULT / "speech-content/stories/2019-story-reduction-audit-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _fetch_one(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    result = con.execute(query)
    return dict(zip((column[0] for column in result.description), result.fetchone()))


def run(archives: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    allnews_dir = archives / "all-the-news-2-1-hf-a1f7e19b"
    abc_file = archives / "abc-million-headlines-2003-2021" / "abcnews-date-text_train.csv"
    if not allnews_dir.is_dir() or not abc_file.is_file():
        raise FileNotFoundError("expected retained All the News 2.1 and ABC source archives")
    output.mkdir(parents=True)
    con = duckdb.connect()
    allnews_glob = _sql_path(allnews_dir / "*.parquet")
    abc_csv = _sql_path(abc_file)
    con.execute(f"""
        CREATE TEMP TABLE allnews_2019 AS
        SELECT try_cast(substr(date, 1, 10) AS DATE) AS published_day,
               trim(title) AS title,
               lower(regexp_replace(trim(title), '[[:space:]]+', ' ', 'g')) AS normalized_title,
               nullif(trim(publication), '') AS publisher,
               nullif(trim(url), '') AS source_url,
               CASE WHEN nullif(trim(article), '') IS NULL THEN NULL
                    ELSE sha256(lower(regexp_replace(trim(article), '[[:space:]]+', ' ', 'g'))) END AS body_sha256
        FROM read_parquet('{allnews_glob}', union_by_name=true)
        WHERE try_cast(substr(date, 1, 10) AS DATE) >= DATE '2019-01-01'
          AND try_cast(substr(date, 1, 10) AS DATE) < DATE '2020-01-01'
          AND nullif(trim(title), '') IS NOT NULL
    """)
    con.execute(f"""
        CREATE TEMP TABLE abc_2019 AS
        SELECT try_strptime(trim(publish_date), '%Y%m%d')::DATE AS published_day,
               trim(headline_text) AS title,
               lower(regexp_replace(trim(headline_text), '[[:space:]]+', ' ', 'g')) AS normalized_title
        FROM read_csv_auto('{abc_csv}', all_varchar=true)
        WHERE try_strptime(trim(publish_date), '%Y%m%d') >= DATE '2019-01-01'
          AND try_strptime(trim(publish_date), '%Y%m%d') < DATE '2020-01-01'
          AND nullif(trim(headline_text), '') IS NOT NULL
    """)

    allnews_summary = _fetch_one(con, """
        SELECT count(*)::BIGINT AS rows,
               count(DISTINCT nullif(trim(source_url), ''))::BIGINT AS distinct_exact_urls,
               count(*) FILTER (WHERE source_url IS NULL)::BIGINT AS missing_url_rows,
               count(DISTINCT normalized_title)::BIGINT AS distinct_normalized_titles,
               count(*) FILTER (WHERE body_sha256 IS NOT NULL)::BIGINT AS rows_with_body,
               count(DISTINCT body_sha256)::BIGINT AS distinct_nonempty_body_hashes,
               count(*) FILTER (WHERE publisher IS NULL)::BIGINT AS missing_publisher_rows
        FROM allnews_2019
    """)
    body_groups = _fetch_one(con, """
        WITH grouped AS (
            SELECT body_sha256, count(*)::BIGINT AS rows,
                   count(DISTINCT source_url)::BIGINT AS urls,
                   count(DISTINCT publisher)::BIGINT AS publishers
            FROM allnews_2019
            WHERE body_sha256 IS NOT NULL
            GROUP BY body_sha256
        )
        SELECT count(*) FILTER (WHERE rows > 1)::BIGINT AS repeated_body_hash_groups,
               coalesce(sum(rows - 1) FILTER (WHERE rows > 1), 0)::BIGINT AS excess_rows_in_repeated_body_groups,
               count(*) FILTER (WHERE rows > 1 AND urls > 1)::BIGINT AS repeated_body_groups_across_urls,
               count(*) FILTER (WHERE rows > 1 AND publishers > 1)::BIGINT AS repeated_body_groups_across_publishers,
               count(*) FILTER (WHERE rows > 1 AND urls > 1 AND publishers > 1)::BIGINT AS exact_body_hash_groups_spanning_urls_and_publishers
        FROM grouped
    """)
    url_groups = _fetch_one(con, """
        WITH grouped AS (
            SELECT source_url, count(*)::BIGINT AS rows,
                   count(DISTINCT body_sha256)::BIGINT AS body_versions,
                   count(DISTINCT normalized_title)::BIGINT AS title_versions,
                   count(DISTINCT published_day)::BIGINT AS publication_days
            FROM allnews_2019 WHERE source_url IS NOT NULL
            GROUP BY source_url
        )
        SELECT count(*) FILTER (WHERE rows > 1)::BIGINT AS repeated_exact_url_groups,
               coalesce(sum(rows - 1) FILTER (WHERE rows > 1), 0)::BIGINT AS excess_rows_on_repeated_urls,
               count(*) FILTER (WHERE rows > 1 AND (body_versions > 1 OR title_versions > 1))::BIGINT AS multi_version_url_candidates,
               count(*) FILTER (WHERE rows > 1 AND publication_days > 1)::BIGINT AS cross_day_url_groups
        FROM grouped
    """)
    title_groups = _fetch_one(con, """
        WITH grouped AS (
            SELECT normalized_title, count(*)::BIGINT AS rows,
                   count(DISTINCT publisher)::BIGINT AS publishers,
                   count(DISTINCT published_day)::BIGINT AS publication_days
            FROM allnews_2019 GROUP BY normalized_title
        )
        SELECT count(*) FILTER (WHERE rows > 1)::BIGINT AS repeated_normalized_title_groups,
               coalesce(sum(rows - 1) FILTER (WHERE rows > 1), 0)::BIGINT AS excess_rows_with_repeated_titles,
               count(*) FILTER (WHERE rows > 1 AND publishers > 1)::BIGINT AS repeated_titles_across_publishers,
               count(*) FILTER (WHERE rows > 1 AND publishers > 1 AND publication_days = 1)::BIGINT AS same_day_cross_publisher_title_groups
        FROM grouped
    """)
    abc_summary = _fetch_one(con, """
        SELECT count(*)::BIGINT AS rows,
               count(DISTINCT normalized_title)::BIGINT AS distinct_normalized_titles,
               count(*) - count(DISTINCT normalized_title)::BIGINT AS excess_rows_with_repeated_titles
        FROM abc_2019
    """)
    cross_corpus = _fetch_one(con, """
        SELECT count(*)::BIGINT AS same_day_exact_title_pairs,
               count(DISTINCT a.normalized_title)::BIGINT AS shared_titles,
               count(DISTINCT a.published_day)::BIGINT AS shared_title_dates
        FROM abc_2019 a JOIN allnews_2019 n
          ON a.published_day = n.published_day
         AND a.normalized_title = n.normalized_title
    """)
    con.close()

    sources = {
        "all_the_news_2_1": allnews_dir,
        "abc_news_australia": abc_file,
    }
    source_manifests = {}
    for name, path in sources.items():
        manifest = path / "SOURCE-MANIFEST.json" if path.is_dir() else path.parent / "SOURCE-MANIFEST.json"
        if manifest.exists():
            source_manifests[name] = _sha256(manifest)
    report = {
        "schema_version": 1,
        "report_type": "2019-conservative-news-identity-reduction-audit-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalization": {
            "title": "trim, lowercase, collapse whitespace; punctuation retained",
            "body": "trim, lowercase, collapse whitespace, then SHA-256; no article body text is emitted",
            "url": "exact source URL only; tracking-parameter canonicalization not applied",
        },
        "all_the_news_2_1": allnews_summary,
        "exact_body_copy_groups": body_groups,
        "repeated_url_groups_and_version_candidates": url_groups,
        "normalized_title_groups": title_groups,
        "abc_headline_summary": abc_summary,
        "cross_corpus_same_day_exact_title_matches": cross_corpus,
        "interpretation_limits": [
            "Exact body hashes identify identical normalized text only; they do not detect paraphrase or establish syndication provenance.",
            "Repeated URLs with changed title/body hashes are version candidates, not confirmed updates or corrections.",
            "Repeated titles can describe distinct events; title equality is not a story-cluster gold label.",
            "ABC is a single outlet without article URLs; cross-corpus title matches are conservative string matches only.",
            "This audit uses calendar year 2019 as the requested corpus slice, not as a justified per-event context horizon.",
            "No target transcripts, outcomes, or model scores are read or changed.",
        ],
    }
    report_path = output / "2019-story-reduction-audit-v2.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    manifest = {
        "schema_version": 1,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": source_manifests,
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "report_sha256": _sha256(report_path),
        "outputs_local_only": True,
    }
    with (output / "RUN-MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output": str(report_path), "allnews": allnews_summary, "body_groups": body_groups, "url_groups": url_groups, "title_groups": title_groups, "abc": abc_summary, "cross_corpus": cross_corpus}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ARCHIVES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.archives, args.output), indent=2))


if __name__ == "__main__":
    main()

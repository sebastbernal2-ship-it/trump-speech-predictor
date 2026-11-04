"""Command-line interface for local speech experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .model import NGramModel, evaluate_continuation


def load_speeches(path: str) -> list[str]:
    source = Path(path)
    if source.suffix == ".jsonl":
        from .forecast import load_records

        return [record["text"] for record in load_records(source)]
    text = source.read_text(encoding="utf-8")
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speechprefix",
        description="Research utilities. The active idea-prediction workflow is context-idea-classifier-evaluate.",
        epilog="Other forecasting and review commands are legacy experiments, not the project objective.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an n-gram baseline")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--order", type=int, default=5)
    train.add_argument("--registry", default="models/registry.jsonl")

    predict = sub.add_parser("predict", help="rank next words from a prefix")
    predict.add_argument("--model", required=True)
    predict.add_argument("--prefix", default="")
    predict.add_argument("--top", type=int, default=10)

    stream = sub.add_parser("stream", help="interactively update prefix predictions")
    stream.add_argument("--model", required=True)
    stream.add_argument("--top", type=int, default=10)

    evaluate = sub.add_parser("evaluate", help="hold out final speech and replay it")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--order", type=int, default=5)

    forecast = sub.add_parser("forecast", help="legacy speaker-history phrase baseline (not the active context-to-ideas model)")
    forecast.add_argument("--input", required=True, help="chronological JSONL speech records")
    forecast.add_argument("--top", type=int, default=20)

    govinfo = sub.add_parser("import-govinfo", help="download a bounded batch of Trump-era presidential records")
    govinfo.add_argument("--output", default="data/speeches.jsonl")
    govinfo.add_argument("--limit", type=int, default=25)
    govinfo.add_argument("--period", choices=("all", "2017", "2025"), default="all")
    govinfo.add_argument("--delay", type=float, default=0.25)

    audit = sub.add_parser("audit", help="summarize imported record types without reading transcript bodies")
    audit.add_argument("--input", required=True, help="JSONL speech records")

    segment = sub.add_parser("segment", help="split legacy paragraph transcripts and classify explicit speaker turns")
    segment.add_argument("--input", default="data/speeches.jsonl")
    segment.add_argument("--output", default="data/segments.jsonl")

    turn_import = sub.add_parser("turn-import", help="import a time-coded transcript page and its event metadata")
    turn_import.add_argument("--transcript", required=True, help="saved page text with a Full Transcript section")
    turn_import.add_argument("--metadata", required=True, help="JSON event metadata and optional reviewed prediction annotations")
    turn_import.add_argument("--output", default="data/events.jsonl")

    rollcall_import = sub.add_parser(
        "import-rollcall", help="import permitted Roll Call Factba.se timestamped transcripts"
    )
    rollcall_import.add_argument("--output", default="data/rollcall-events.jsonl")
    rollcall_import.add_argument("--limit", type=int, default=0, help="maximum source records; 0 imports all")
    rollcall_import.add_argument("--delay", type=float, default=0.15, help="seconds between requests")
    rollcall_import.add_argument("--timeout", type=int, default=30)
    rollcall_import.add_argument("--start-date", help="optional earliest event date, YYYY-MM-DD")
    rollcall_import.add_argument("--end-date", help="optional latest event date, YYYY-MM-DD")
    rollcall_import.add_argument(
        "--reuse-terms",
        default="User-reported permission for bulk retrieval and ML use; grant details are not stored here.",
        help="permission or license note stored with every event",
    )

    rollcall_split = sub.add_parser(
        "split-rollcall", help="make disjoint Trump/Q&A/speech task files from the Roll Call event corpus"
    )
    rollcall_split.add_argument("--input", default="data/rollcall-events.jsonl")
    rollcall_split.add_argument("--output-dir", default="data/rollcall-by-type")

    speech_prepare = sub.add_parser(
        "prepare-rollcall-speeches", help="export long Trump-only speech text for continuation training"
    )
    speech_prepare.add_argument("--input", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    speech_prepare.add_argument("--output", default="data/rollcall-by-type/trump-long-speeches.jsonl")
    speech_prepare.add_argument("--min-target-words", type=int, default=1000)
    speech_prepare.add_argument("--min-target-share", type=float, default=0.8)

    context = sub.add_parser("context-fetch", help="fetch time-bounded Federal Register context (no API key)")
    context.add_argument("--as-of", required=True, help="prediction date; same-day records are excluded, YYYY-MM-DD")
    context.add_argument("--query", required=True, help="question or topic known at prediction time")
    context.add_argument("--lookback-days", type=int, default=30)
    context.add_argument("--limit", type=int, default=10)
    context.add_argument("--output", default="data/world_context.jsonl")

    context_news_search = sub.add_parser(
        "context-news-search", help="collect retrospective Google News candidates using metadata-only query profiles"
    )
    context_news_search.add_argument("--input", default="data/rollcall-context-snapshots-v1.jsonl")
    context_news_search.add_argument("--speaker", default="Donald Trump")
    context_news_search.add_argument("--lookback-days", type=int, default=30)
    context_news_search.add_argument("--delay", type=float, default=0.4)
    context_news_search.add_argument("--timeout", type=float, default=25)
    context_news_search.add_argument("--profile", choices=("speaker-name", "speaker-topic-families"), default="speaker-name")
    context_news_search.add_argument("--output", default="data/rollcall-context-snapshots-trump-name-v2.jsonl")

    context_news_freeze = sub.add_parser(
        "context-news-freeze", help="collect and seal news context before a scheduled event cutoff"
    )
    context_news_freeze.add_argument("--manifest", required=True, help="event schedule provenance JSON")
    context_news_freeze.add_argument("--output", required=True, help="new immutable prospective snapshot JSON")
    context_news_freeze.add_argument("--profile", choices=("speaker-name", "speaker-topic-families"), default="speaker-topic-families")
    context_news_freeze.add_argument("--lookback-days", type=int, default=30)
    context_news_freeze.add_argument("--delay", type=float, default=0.4)
    context_news_freeze.add_argument("--timeout", type=float, default=25)

    idea_source_import = sub.add_parser(
        "idea-source-import", help="validate and append source items to the standalone idea-layer ledger"
    )
    idea_source_import.add_argument("--input", required=True, help="JSONL source-evidence records")
    idea_source_import.add_argument("--ledger", default="data/idea-layer/source-ledger.jsonl")

    idea_post_import = sub.add_parser(
        "idea-post-import", help="normalize and append permitted Trump-post archive records"
    )
    idea_post_import.add_argument("--input", required=True, help="JSONL v1 Trump-post archive export")
    idea_post_import.add_argument("--ledger", default="data/idea-layer/source-ledger.jsonl")

    idea_news_collect = sub.add_parser(
        "idea-news-collect", help="capture timestamped Google News headlines for an upcoming event"
    )
    idea_news_collect.add_argument("--manifest", required=True, help="event manifest with a future prediction cutoff")
    idea_news_collect.add_argument("--query", action="append", required=True, help="predeclared topic query; repeatable")
    idea_news_collect.add_argument("--ledger", default="data/idea-layer/source-ledger.jsonl")
    idea_news_collect.add_argument("--report", required=True, help="new immutable retrieval report JSON")
    idea_news_collect.add_argument("--lookback-days", type=int, default=30)
    idea_news_collect.add_argument("--delay", type=float, default=0.5)
    idea_news_collect.add_argument("--timeout", type=float, default=20)

    idea_snapshot_freeze = sub.add_parser(
        "idea-snapshot-freeze", help="freeze verified news, posts, and event details before a speech cutoff"
    )
    idea_snapshot_freeze.add_argument("--manifest", required=True, help="pre-event idea-layer event manifest JSON")
    idea_snapshot_freeze.add_argument("--ledger", default="data/idea-layer/source-ledger.jsonl")
    idea_snapshot_freeze.add_argument("--output", required=True, help="new immutable input snapshot JSON")
    idea_snapshot_freeze.add_argument("--lookback-days", type=int, default=30)

    idea_proposals_seal = sub.add_parser(
        "idea-proposals-seal", help="seal provider-neutral idea proposals against one frozen snapshot"
    )
    idea_proposals_seal.add_argument("--snapshot", required=True)
    idea_proposals_seal.add_argument("--input", required=True, help="JSON array of idea_id, idea, evidence_source_ids")
    idea_proposals_seal.add_argument("--generator", required=True, help="human or provider/model identifier")
    idea_proposals_seal.add_argument("--output", required=True, help="new immutable proposal-set JSON")

    idea_headline_propose = sub.add_parser(
        "idea-headline-propose",
        help="extract source-cited issue-phrase proposals from a frozen snapshot (lexical baseline)",
    )
    idea_headline_propose.add_argument("--snapshot", required=True)
    idea_headline_propose.add_argument("--output", required=True, help="new immutable proposal-set JSON")

    idea_semantic_ablate = sub.add_parser(
        "idea-semantic-ablate", help="legacy semantic-forecast prototype ablation; not the active supervised model"
    )
    idea_semantic_ablate.add_argument("--snapshot", required=True)
    idea_semantic_ablate.add_argument(
        "--variant", required=True,
        choices=("combined", "news_only", "event_context_only", "speaker_history_only"),
    )
    idea_semantic_ablate.add_argument("--output", required=True)

    idea_semantic_prompt = sub.add_parser(
        "idea-semantic-prompt",
        help="legacy provider-neutral semantic-forecast prompt prototype; not the active supervised model",
    )
    idea_semantic_prompt.add_argument("--snapshot", required=True)
    idea_semantic_prompt.add_argument("--max-sources", type=int, default=40)
    idea_semantic_prompt.add_argument("--output", required=True, help="new immutable prompt JSON")

    idea_semantic_proposals = sub.add_parser(
        "idea-semantic-proposals",
        help="legacy semantic-forecast prototype response validation",
    )
    idea_semantic_proposals.add_argument("--snapshot", required=True)
    idea_semantic_proposals.add_argument("--prompt", required=True)
    idea_semantic_proposals.add_argument("--response", required=True, help="provider response containing raw JSON")
    idea_semantic_proposals.add_argument("--provider", required=True)
    idea_semantic_proposals.add_argument("--model", required=True)
    idea_semantic_proposals.add_argument("--output", required=True)

    idea_semantic_rank = sub.add_parser(
        "idea-semantic-rank", help="legacy semantic-forecast prototype ranking"
    )
    idea_semantic_rank.add_argument("--snapshot", required=True)
    idea_semantic_rank.add_argument("--proposals", required=True)
    idea_semantic_rank.add_argument("--top-k", type=int, default=10)
    idea_semantic_rank.add_argument("--output", required=True)

    idea_semantic_bundle = sub.add_parser(
        "idea-semantic-bundle", help="legacy semantic-forecast prototype evaluation bundle"
    )
    idea_semantic_bundle.add_argument("--forecast", action="append", required=True, help="semantic forecast JSON; repeat for events")
    idea_semantic_bundle.add_argument("--output", required=True)

    idea_rank = sub.add_parser(
        "idea-rank", help="rank cited proposals with the transparent evidence-weighted baseline"
    )
    idea_rank.add_argument("--snapshot", required=True)
    idea_rank.add_argument("--proposals", required=True, help="sealed proposal-set JSON")
    idea_rank.add_argument("--output", required=True, help="new immutable forecast JSON")
    idea_rank.add_argument("--model-version", default="evidence-weighted-baseline-v1")

    idea_news_backtest = sub.add_parser(
        "idea-news-phrase-backtest",
        help="legacy news-phrase prototype backtest; not the active context-to-ideas classifier",
    )
    idea_news_backtest.add_argument("--snapshots", default="data/rollcall-context-snapshots-topic-families-v1.jsonl")
    idea_news_backtest.add_argument("--events", default="data/rollcall-events.jsonl")
    idea_news_backtest.add_argument("--output", required=True, help="new immutable backtest report JSON")
    idea_news_backtest.add_argument("--train-fraction", type=float, default=0.8)
    idea_news_backtest.add_argument("--validation-fraction", type=float, default=0.1)
    idea_news_backtest.add_argument(
        "--semantic-model", default=None,
        help="optional locally cached Hugging Face encoder for phrase/headline cosine ranking; never downloads weights",
    )

    assistant_review_validate = sub.add_parser(
        "assistant-review-validate",
        help="validate transcript-only assistant review hashes, citations, and declared review ordering",
    )
    assistant_review_validate.add_argument("--pass-a", action="append", required=True, help="frozen Pass-A JSONL; repeatable")
    assistant_review_validate.add_argument("--mapped", action="append", required=True, help="provenance-mapped JSONL; repeatable")
    assistant_review_validate.add_argument("--key", default="data/rollcall-theme-review-key-v2.json")
    assistant_review_validate.add_argument("--events", default="data/rollcall-events.jsonl")
    assistant_review_validate.add_argument("--transcripts", default="data/rollcall-theme-review-annotator-a-v2.jsonl")
    assistant_review_validate.add_argument("--output", help="optional new JSON validation report; existing files are never overwritten")

    assistant_review_forecast = sub.add_parser(
        "assistant-review-history-forecast",
        help="legacy history-only comparator replay (not the active context-to-ideas model)",
    )
    assistant_review_forecast.add_argument("--history", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    assistant_review_forecast.add_argument("--targets", required=True, help="JSONL with only event_id and event_date")
    assistant_review_forecast.add_argument("--top-k", type=int, default=10)
    assistant_review_forecast.add_argument("--output", required=True, help="new immutable forecast replay JSON")

    assistant_news_forecast = sub.add_parser(
        "assistant-review-news-forecast",
        help="replay a retrospective, date-filtered news-to-phrase baseline for assistant-reviewed events",
    )
    assistant_news_forecast.add_argument("--snapshots", required=True, help="historical headline snapshot JSONL")
    assistant_news_forecast.add_argument("--top-k", type=int, default=10)
    assistant_news_forecast.add_argument("--output", required=True, help="new immutable retrospective forecast report")

    assistant_news_audit = sub.add_parser(
        "assistant-review-news-audit",
        help="diagnose lexical evidence, phrase-generation, and top-k bottlenecks in a frozen retrospective news forecast",
    )
    assistant_news_audit.add_argument("--snapshots", required=True)
    assistant_news_audit.add_argument("--forecast", required=True)
    assistant_news_audit.add_argument("--agreement", required=True)
    assistant_news_audit.add_argument("--reference", action="append", required=True)
    assistant_news_audit.add_argument("--top-k", type=int, default=10)
    assistant_news_audit.add_argument("--output", required=True, help="new immutable post-hoc audit report")

    assistant_review_score = sub.add_parser(
        "assistant-review-forecast-score",
        help="aggregate explicit assistant decisions against frozen transcript-only outcomes",
    )
    assistant_review_score.add_argument("--forecast", required=True, help="frozen history forecast replay JSON")
    assistant_review_score.add_argument("--reference", action="append", required=True, help="mapped assistant outcome JSONL; repeatable")
    assistant_review_score.add_argument("--decisions", required=True, help="assistant candidate-match decisions JSONL")
    assistant_review_score.add_argument("--protocol-snapshot", required=True, help="exact AI review policy bytes named by decisions")
    assistant_review_score.add_argument("--output", required=True, help="new immutable agreement report JSON")

    speaker_idea_eval = sub.add_parser(
        "speaker-idea-evaluate", help="legacy speaker-history baseline evaluation; not the active context-to-ideas model"
    )
    speaker_idea_eval.add_argument("--input", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    speaker_idea_eval.add_argument("--output", default="data/rollcall-speaker-idea-evaluation-v1.json")
    speaker_idea_eval.add_argument("--train-fraction", type=float, default=0.8)
    speaker_idea_eval.add_argument("--validation-fraction", type=float, default=0.1)

    context_idea_classifier = sub.add_parser(
        "context-idea-classifier-evaluate", help="train a chronological supervised news-headline-to-speech-idea classifier"
    )
    context_idea_classifier.add_argument("--snapshots", required=True, help="historical pre-event context snapshots JSONL")
    context_idea_classifier.add_argument("--events", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    context_idea_classifier.add_argument("--output", required=True)
    context_idea_classifier.add_argument("--train-fraction", type=float, default=0.8)
    context_idea_classifier.add_argument("--validation-fraction", type=float, default=0.1)

    context_idea_predict = sub.add_parser(
        "context-idea-predict", help="rank ideas for a later event from a fitted context-classifier artifact"
    )
    context_idea_predict.add_argument("--model", required=True, help="evaluation JSON containing the final fitted model")
    context_idea_predict.add_argument("--snapshot", required=True, help="one target event's preceding-week context snapshot JSON")
    context_idea_predict.add_argument("--top-k", type=int, default=10)
    context_idea_predict.add_argument("--output", required=True, help="new immutable prediction JSON")

    selective_eval = sub.add_parser(
        "speaker-idea-selective-evaluate",
        help="legacy speaker-history baseline selection; not the active context-to-ideas model",
    )
    selective_eval.add_argument("--input", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    selective_eval.add_argument("--output", default="data/rollcall-speaker-idea-selective-evaluation-v5.json")
    selective_eval.add_argument("--target-precision", type=float, default=0.8)
    selective_eval.add_argument("--minimum-inner-predictions", type=int, default=50)

    selective_forecast = sub.add_parser(
        "speaker-idea-forecast",
        help="legacy speaker-history phrase forecast; not the active context-to-ideas model",
    )
    selective_forecast.add_argument("--input", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    selective_forecast.add_argument("--as-of", required=True, help="exclusive ISO date cutoff for the forecast")
    selective_forecast.add_argument("--output", required=True, help="new immutable forecast JSON")
    selective_forecast.add_argument("--window-days", type=int, default=90)
    selective_forecast.add_argument("--min-history-events", type=int, default=50)
    selective_forecast.add_argument("--max-predictions", type=int, default=1)

    context_theme_eval = sub.add_parser(
        "context-theme-evaluate", help="legacy context-theme prototype evaluation; not the active supervised model"
    )
    context_theme_eval.add_argument("--snapshots", default="data/rollcall-context-snapshots-v1.jsonl")
    context_theme_eval.add_argument("--labels", default="data/rollcall-theme-family-drafts-v1.jsonl")
    context_theme_eval.add_argument("--output", default="data/rollcall-context-theme-evaluation-v1.json")

    transcript_idea_audit = sub.add_parser(
        "transcript-idea-audit", help="legacy retrospective headline/phrase audit; not the active supervised model"
    )
    transcript_idea_audit.add_argument("--snapshots", default="data/rollcall-context-snapshots-v1.jsonl")
    transcript_idea_audit.add_argument("--events", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    transcript_idea_audit.add_argument("--output", default="data/rollcall-transcript-idea-audit-v1.json")

    context_issue_forecast = sub.add_parser(
        "context-issue-forecast", help="legacy prospective candidate workflow; not the active retrospective supervised model"
    )
    context_issue_forecast.add_argument("--snapshot", required=True, help="sealed prospective context snapshot JSON")
    context_issue_forecast.add_argument("--output", required=True, help="new immutable local forecast JSON")

    context_candidates = sub.add_parser(
        "context-candidates", help="legacy source-grounded candidate extractor; not the active supervised classifier"
    )
    context_candidates.add_argument("--input", required=True, help="JSON context map or JSONL snapshot file")
    context_candidates.add_argument("--event-id", help="select an event from a multi-event JSONL input")

    context_idea_eval = sub.add_parser(
        "context-idea-evaluate", help="legacy context-candidate comparison with a speaker-history baseline; not the active model"
    )
    context_idea_eval.add_argument("--snapshots", default="data/rollcall-context-snapshots-v1.jsonl")
    context_idea_eval.add_argument("--events", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    context_idea_eval.add_argument("--output", default="data/rollcall-context-idea-evaluation-v1.json")
    context_idea_eval.add_argument("--train-fraction", type=float, default=0.8)
    context_idea_eval.add_argument("--validation-fraction", type=float, default=0.1)

    context_issue_forecast_eval = sub.add_parser(
        "context-issue-forecast-evaluate", help="score a saved pre-cutoff issue forecast after its event"
    )
    context_issue_forecast_eval.add_argument("--forecast", required=True, help="immutable forecast JSON")
    context_issue_forecast_eval.add_argument("--snapshot", required=True, help="sealed prospective snapshot JSON")
    context_issue_forecast_eval.add_argument("--events", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    context_issue_forecast_eval.add_argument("--output", required=True, help="new immutable local evaluation report")
    context_issue_forecast_eval.add_argument("--top-k", default="4,10,25", help="comma-separated cutoffs")

    context_candidate_eval = sub.add_parser(
        "context-candidate-evaluate", help="legacy candidate evaluation; not the active supervised classifier"
    )
    context_candidate_eval.add_argument("--snapshots", default="data/rollcall-context-snapshots-v1.jsonl")
    context_candidate_eval.add_argument("--labels", default="data/rollcall-theme-family-drafts-v1.jsonl")
    context_candidate_eval.add_argument("--output", default="data/rollcall-context-candidate-evaluation-v1.json")

    context_snapshot_audit = sub.add_parser(
        "context-snapshot-audit", help="audit exploratory context records for prospective-freeze eligibility"
    )
    context_snapshot_audit.add_argument("--snapshots", default="data/rollcall-context-snapshots-v1.jsonl")
    context_snapshot_audit.add_argument("--output", default="data/rollcall-context-snapshot-audit-v1.json")

    context_snapshot_validate = sub.add_parser(
        "context-snapshot-validate", help="validate a timestamped prospective context snapshot"
    )
    context_snapshot_validate.add_argument("--input", required=True, help="prospective snapshot JSON")

    context_snapshot_seal = sub.add_parser(
        "context-snapshot-seal", help="freeze a future-event snapshot now and write an immutable hashed copy"
    )
    context_snapshot_seal.add_argument("--input", required=True, help="snapshot JSON with event and source provenance")
    context_snapshot_seal.add_argument("--output", required=True, help="new output path; existing files are never overwritten")

    theme_review = sub.add_parser(
        "theme-review-packet", help="legacy human-annotation prototype; not part of the active assistant-only workflow"
    )
    theme_review.add_argument(
        "--selection-manifest", "--snapshots", dest="snapshots",
        default="data/rollcall-context-snapshots-v1.jsonl",
        help="local event-selection JSONL (legacy alias: --snapshots)",
    )
    theme_review.add_argument("--events", default="data/rollcall-by-type/trump-speech-candidates.jsonl")
    theme_review.add_argument("--taxonomy", default="data/rollcall-theme-family-drafts-v1.jsonl")
    theme_review.add_argument("--annotator-a", default="data/rollcall-theme-review-annotator-a-v1.jsonl")
    theme_review.add_argument("--annotator-b", default="data/rollcall-theme-review-annotator-b-v1.jsonl")
    theme_review.add_argument("--key", default="data/rollcall-theme-review-key-v1.json")
    theme_review.add_argument("--seed", type=int, default=17)
    theme_review.add_argument(
        "--selection-description", default="User-provided event selection; see the local selection manifest.",
    )

    theme_review_agreement = sub.add_parser(
        "theme-review-agreement", help="legacy human-annotation prototype; not part of the active assistant-only workflow"
    )
    theme_review_agreement.add_argument("--annotator-a", default="data/rollcall-theme-review-annotator-a-v1.jsonl")
    theme_review_agreement.add_argument("--annotator-b", default="data/rollcall-theme-review-annotator-b-v1.jsonl")
    theme_review_agreement.add_argument("--key", default="data/rollcall-theme-review-key-v1.json")
    theme_review_agreement.add_argument("--output", default="data/rollcall-theme-review-agreement-v1.json")

    theme_review_template = sub.add_parser(
        "theme-review-adjudication-template", help="legacy human-annotation prototype; not part of the active assistant-only workflow"
    )
    theme_review_template.add_argument("--annotator-packet", default="data/rollcall-theme-review-annotator-a-v1.jsonl")
    theme_review_template.add_argument("--key", default="data/rollcall-theme-review-key-v1.json")
    theme_review_template.add_argument("--output", default="data/rollcall-theme-adjudication-template-v1.jsonl")

    theme_review_adjudicate = sub.add_parser(
        "theme-review-adjudicate", help="legacy human-annotation prototype; not part of the active assistant-only workflow"
    )
    theme_review_adjudicate.add_argument("--annotator-a", default="data/rollcall-theme-review-annotator-a-v1.jsonl")
    theme_review_adjudicate.add_argument("--annotator-b", default="data/rollcall-theme-review-annotator-b-v1.jsonl")
    theme_review_adjudicate.add_argument("--key", default="data/rollcall-theme-review-key-v1.json")
    theme_review_adjudicate.add_argument("--decisions", required=True, help="human-written adjudication JSONL")
    theme_review_adjudicate.add_argument("--output", default="data/rollcall-theme-reviewed-v1.jsonl")

    llm = sub.add_parser("llm", help="run a Hugging Face causal-LM continuation baseline")
    llm.add_argument("--model", default="openai-community/gpt2")
    llm.add_argument("--revision", help="Hugging Face model commit; the default GPT-2 revision is pinned")
    llm.add_argument("--prefix", required=True)
    llm.add_argument("--candidates", type=int, default=8)
    llm.add_argument("--tokens", type=int, default=32)
    llm.add_argument("--download-retries", type=int, default=3)
    llm.add_argument("--download-timeout", type=int, default=20, help="per-request timeout in seconds")

    ncm = sub.add_parser("ncm-generate", help="generate coarse-to-fine candidates with official Stanford NCM weights")
    ncm.add_argument("--count", type=int, default=2)
    ncm.add_argument("--download-retries", type=int, default=3)
    ncm.add_argument("--download-timeout", type=int, default=20)

    hybrid_train = sub.add_parser("hybrid-train", help="train the from-scratch SpeechPlan-H model on local records")
    hybrid_train.add_argument("--input", required=True)
    hybrid_train.add_argument("--output", default="checkpoints/speechplan-h.pt")
    hybrid_train.add_argument("--registry", default="models/registry.jsonl")
    hybrid_train.add_argument("--epochs", type=int, default=1)
    hybrid_train.add_argument("--batch-size", type=int, default=32)
    hybrid_train.add_argument("--stride", type=int, default=32)
    hybrid_train.add_argument("--device", default="auto")
    hybrid_train.add_argument("--max-steps", type=int, default=0, help="cap training steps per epoch for smoke tests")
    hybrid_train.add_argument("--evidence-pairs", help="dated evidence mappings keyed by question segment ID")

    weak_labels = sub.add_parser(
        "weak-plan-labels", help="create explicitly weak extractive plan labels from real Roll Call speech transcripts"
    )
    weak_labels.add_argument("--events", default="data/rollcall-events.jsonl")
    weak_labels.add_argument("--speech-records", default="data/rollcall-by-type/trump-long-speeches.jsonl")
    weak_labels.add_argument("--output", default="data/rollcall-weak-speech-plans.jsonl")
    weak_labels.add_argument("--section-turns", type=int, default=15)
    weak_labels.add_argument("--point-turns", type=int, default=3)
    weak_labels.add_argument("--top-terms", type=int, default=4)

    explicit_train = sub.add_parser(
        "explicit-plan-train", help="train the plan-path predictor with reviewed, weak-extractive, or assistant-draft labels"
    )
    explicit_train.add_argument("--events", default="data/rollcall-events.jsonl")
    explicit_train.add_argument("--plans", default="data/rollcall-speech-plans.jsonl")
    explicit_train.add_argument("--output", default="checkpoints/explicit-plan-run-001.pt")
    explicit_train.add_argument("--registry", default="models/registry.jsonl")
    explicit_train.add_argument("--epochs", type=int, default=5)
    explicit_train.add_argument("--batch-size", type=int, default=16)
    explicit_train.add_argument("--context-length", type=int, default=128)
    explicit_train.add_argument("--plan-length", type=int, default=96)
    explicit_train.add_argument("--width", type=int, default=128)
    explicit_train.add_argument("--layers", type=int, default=4)
    explicit_train.add_argument("--heads", type=int, default=4)
    explicit_train.add_argument("--min-frequency", type=int, default=2)
    explicit_train.add_argument("--max-vocabulary", type=int, default=20_000)
    explicit_train.add_argument("--seed", type=int, default=17)
    explicit_train.add_argument("--cpu-threads", type=int, default=0)
    explicit_train.add_argument("--device", default="auto")
    explicit_train.add_argument("--max-steps", type=int, default=0)
    explicit_train.add_argument(
        "--label-policy", choices=("reviewed", "weak-extractive", "assistant-draft"), default="reviewed",
    )

    explicit_predict = sub.add_parser(
        "explicit-plan-predict", help="predict a theme/section/talking-point path from an observed prefix"
    )
    explicit_predict.add_argument("--model", required=True)
    explicit_predict.add_argument("--prefix", required=True)
    explicit_predict.add_argument("--tokens", type=int, default=96)
    explicit_predict.add_argument("--device", default="auto")

    explicit_evaluate = sub.add_parser(
        "explicit-plan-evaluate", help="evaluate a plan-path model against chronological held-out examples and baselines"
    )
    explicit_evaluate.add_argument("--model", required=True)
    explicit_evaluate.add_argument("--events", default="data/rollcall-events.jsonl")
    explicit_evaluate.add_argument("--plans", required=True)
    explicit_evaluate.add_argument("--device", default="auto")
    explicit_evaluate.add_argument("--cpu-threads", type=int, default=0)

    hybrid = sub.add_parser("hybrid", help="generate with a trained from-scratch SpeechPlan-H checkpoint")
    hybrid.add_argument("--model", required=True)
    hybrid.add_argument("--prefix", required=True)
    hybrid.add_argument("--question", default="", help="explicit question to condition the response on")
    hybrid.add_argument("--as-of", help="prediction date for optional Federal Register retrieval; same-day records excluded")
    hybrid.add_argument("--context-query", help="retrieval query; defaults to --question")
    hybrid.add_argument("--tokens", type=int, default=32)
    hybrid.add_argument("--device", default="auto")

    pretrained = sub.add_parser("pretrained-hybrid", help="run the separate GPT-2/NCM candidate-fusion baseline")
    pretrained.add_argument("--prefix", required=True)
    pretrained.add_argument("--llm-revision", help="optional Hugging Face commit for a non-default LLM")
    pretrained.add_argument("--llm-candidates", type=int, default=4)
    pretrained.add_argument("--ncm-candidates", type=int, default=4)
    pretrained.add_argument("--llm-weight", type=float, default=0.5, help="provisional; fit on validation data")
    pretrained.add_argument("--ncm-weight", type=float, default=0.5, help="provisional; fit on validation data")
    pretrained.add_argument("--download-retries", type=int, default=3)
    pretrained.add_argument("--download-timeout", type=int, default=20)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from .model_versions import assert_new_model_artifact, create_model_version, register_model_version

            assert_new_model_artifact(args.output, args.registry)
            speeches = load_speeches(args.input)
            model = NGramModel.train(speeches, order=args.order)
            version = create_model_version(
                model_family="word-ngram",
                architecture="format-1",
                task="continuation",
                dataset_path=args.input,
                config={"order": model.order, "vocabulary_size": len(model.vocabulary)},
                training={"training_records": len(speeches)},
            )
            model.save(args.output, metadata=version)
            register_model_version(args.output, version, args.registry)
            print(f"Saved order-{model.order} model version {version['model_version']} to {args.output}")
        elif args.command == "predict":
            for word, probability in NGramModel.load(args.model).predict(args.prefix, args.top):
                print(f"{word}\t{probability:.4f}")
        elif args.command == "stream":
            model = NGramModel.load(args.model)
            prefix: list[str] = []
            print("Enter transcript chunks; blank line exits.")
            while True:
                try:
                    chunk = input("> ")
                except EOFError:
                    break
                if not chunk.strip():
                    break
                prefix.append(chunk.strip())
                print("  " + ", ".join(f"{w} ({p:.3f})" for w, p in model.predict(" ".join(prefix), args.top)))
        elif args.command == "evaluate":
            result = evaluate_continuation(load_speeches(args.input), order=args.order)
            for key, value in result.items():
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        elif args.command == "forecast":
            from .forecast import forecast_phrases, load_records

            records = load_records(args.input)
            for phrase, probability in forecast_phrases(records, top=args.top):
                print(f"{phrase}\t{probability:.4f}")
        elif args.command == "import-govinfo":
            from .govinfo import collect

            saved, failed = collect(Path(args.output), args.limit, args.delay, args.period)
            print(f"Done: saved {saved}, skipped {failed}.")
        elif args.command == "audit":
            from .audit import classify_title, summarize_records
            from .forecast import load_records

            records = load_records(args.input)
            summary = summarize_records(records)
            print(f"records: {summary['records']}")
            print(f"counts: {summary['counts']}")
            print(f"candidates_for_manual_review: {summary['candidates_for_manual_review']}")
            for record in records:
                print(f"{record['date']}\t{classify_title(record['event'])}\t{record['speech_id']}\t{record['event']}")
        elif args.command == "segment":
            from collections import Counter
            import json
            from .forecast import load_records
            from .segments import segment_records

            records = load_records(args.input)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            counts = Counter()
            with output.open("w", encoding="utf-8") as stream:
                for segment_record in segment_records(records):
                    counts[segment_record["segment_type"]] += 1
                    stream.write(json.dumps(segment_record, ensure_ascii=False) + "\n")
            print(f"Wrote {sum(counts.values())} segments from {len(records)} records to {output}.")
            print(f"segment_types: {dict(sorted(counts.items()))}")
        elif args.command == "turn-import":
            import json
            from .events import add_prediction_examples, append_event, parse_timestamped_transcript

            metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
            event = parse_timestamped_transcript(
                Path(args.transcript).read_text(encoding="utf-8"), metadata
            )
            annotations = metadata.get("prediction_annotations", [])
            event = add_prediction_examples(event, annotations)
            append_event(args.output, event)
            print(f"Appended {len(event['turns'])} timestamped turns for {event['event_id']} to {args.output}.")
        elif args.command == "import-rollcall":
            from .rollcall import collect_rollcall

            def progress(stats):
                print(
                    f"Processed {stats['records_seen']} records; fetched {stats['pages_fetched']} pages; "
                    f"added {stats['events_added']} events.",
                    file=sys.stderr,
                )

            result = collect_rollcall(
                args.output,
                limit=args.limit,
                delay=args.delay,
                timeout=args.timeout,
                start_date=args.start_date,
                end_date=args.end_date,
                reuse_terms=args.reuse_terms,
                progress=progress,
            )
            print(f"Roll Call import complete: {result}")
        elif args.command == "split-rollcall":
            from .rollcall_split import split_rollcall_events

            result = split_rollcall_events(args.input, args.output_dir)
            print(f"Roll Call task split complete: {result}")
        elif args.command == "prepare-rollcall-speeches":
            from .rollcall_split import extract_long_speech_records

            result = extract_long_speech_records(
                args.input,
                args.output,
                min_target_words=args.min_target_words,
                min_target_share=args.min_target_share,
            )
            print(f"Long Trump speech export complete: {result}")
        elif args.command == "context-fetch":
            from .federalregister import append_context, search_documents

            records = search_documents(args.as_of, args.query, args.limit, args.lookback_days)
            added = append_context(args.output, records)
            print(f"Found {len(records)} pre-event documents; appended {added} new records to {args.output}.")
        elif args.command == "context-news-search":
            import json
            from .news_retrieval import collect_speaker_news_snapshots

            result = collect_speaker_news_snapshots(
                args.input, args.output, speaker=args.speaker, lookback_days=args.lookback_days,
                delay_seconds=args.delay, timeout_seconds=args.timeout, profile=args.profile,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "context-news-freeze":
            import json
            from .news_retrieval import collect_prospective_news_snapshot

            result = collect_prospective_news_snapshot(
                args.manifest, args.output, profile=args.profile, lookback_days=args.lookback_days,
                delay_seconds=args.delay, timeout_seconds=args.timeout,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "idea-source-import":
            import json
            from .idea_layer.ledger import append_source_records

            result = append_source_records(args.input, args.ledger)
            print(json.dumps({"ledger": args.ledger, **result}, ensure_ascii=False, indent=2))
        elif args.command == "idea-post-import":
            import json
            from .idea_layer.ledger import append_source_items
            from .idea_layer.posts import load_trump_post_export

            sources = load_trump_post_export(args.input)
            result = append_source_items(sources, args.ledger)
            print(json.dumps({"ledger": args.ledger, "normalized_posts": len(sources), **result}, ensure_ascii=False, indent=2))
        elif args.command == "idea-news-collect":
            import hashlib
            import json
            from .idea_layer.contracts import validate_event_manifest
            from .idea_layer.ledger import append_source_items
            from .idea_layer.news import retrieve_google_news_evidence

            report_path = Path(args.report)
            if report_path.exists():
                raise FileExistsError(f"refusing to overwrite retrieval report: {report_path}")
            if report_path.resolve() == Path(args.ledger).resolve():
                raise ValueError("retrieval report and source ledger paths must differ")
            manifest_path = Path(args.manifest)
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            event = validate_event_manifest(manifest)
            capture = retrieve_google_news_evidence(
                args.query, event["prediction_cutoff_at"], lookback_days=args.lookback_days,
                timeout_seconds=args.timeout, delay_seconds=args.delay,
            )
            ledger_result = append_source_items(capture["sources"], args.ledger)
            report = {
                "schema_version": 1,
                "record_type": "idea-layer-news-capture-run",
                "event_id": event["event_id"],
                "event_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "ledger_path": args.ledger,
                "ledger_append": ledger_result,
                "capture": capture,
            }
            report["report_sha256"] = hashlib.sha256(
                json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "report": args.report,
                "event_id": event["event_id"],
                "captured_headlines": capture["unique_headline_count"],
                "ledger_append": ledger_result,
                "report_sha256": report["report_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-snapshot-freeze":
            import json
            from .idea_layer.ledger import freeze_idea_input_snapshot

            result = freeze_idea_input_snapshot(
                args.manifest, args.ledger, args.output, lookback_days=args.lookback_days,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "idea-proposals-seal":
            import json
            from .idea_layer.ranker import make_idea_proposal_set

            output_path = Path(args.output)
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite proposal set: {output_path}")
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            proposals = json.loads(Path(args.input).read_text(encoding="utf-8"))
            proposal_set = make_idea_proposal_set(snapshot, proposals, generator=args.generator)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(proposal_set, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "output": args.output,
                "proposal_set_id": proposal_set["proposal_set_id"],
                "proposal_count": len(proposal_set["proposals"]),
                "proposal_set_sha256": proposal_set["proposal_set_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-headline-propose":
            import json
            from .idea_layer.headline_proposals import propose_headline_issue_phrases

            output_path = Path(args.output)
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite proposal set: {output_path}")
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            proposal_set = propose_headline_issue_phrases(snapshot)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(proposal_set, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "output": args.output,
                "proposal_set_id": proposal_set["proposal_set_id"],
                "generator": proposal_set["generator"],
                "proposal_count": len(proposal_set["proposals"]),
                "proposal_set_sha256": proposal_set["proposal_set_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-semantic-ablate":
            import json
            from .idea_layer.ablation import derive_source_ablation
            from .idea_layer.semantic_forecast import write_immutable_json

            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            derived = derive_source_ablation(snapshot, args.variant)
            write_immutable_json(args.output, derived)
            print(json.dumps({
                "output": args.output,
                "event_id": derived["event_id"],
                "source_ablation": derived["source_ablation"],
                "source_count": len(derived["sources"]),
                "snapshot_sha256": derived["snapshot_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-semantic-prompt":
            import json
            from .idea_layer.semantic_forecast import build_semantic_prompt, write_immutable_json

            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            prompt = build_semantic_prompt(snapshot, max_sources=args.max_sources)
            write_immutable_json(args.output, prompt)
            print(json.dumps({
                "output": args.output,
                "event_id": prompt["event_id"],
                "prompt_source_count": len(prompt["prompt_source_ids"]),
                "prompt_sha256": prompt["prompt_sha256"],
                "provider_call_made": False,
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-semantic-proposals":
            import json
            from .idea_layer.semantic_forecast import build_semantic_proposal_set, write_immutable_json

            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            prompt = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
            response = Path(args.response).read_text(encoding="utf-8")
            proposals = build_semantic_proposal_set(
                snapshot, prompt, response, provider=args.provider, model=args.model,
            )
            write_immutable_json(args.output, proposals)
            print(json.dumps({
                "output": args.output,
                "event_id": proposals["event_id"],
                "proposal_count": len(proposals["proposals"]),
                "proposal_set_sha256": proposals["proposal_set_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-semantic-rank":
            import json
            from .idea_layer.semantic_forecast import rank_semantic_idea_proposals, write_immutable_json

            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            proposals = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
            forecast = rank_semantic_idea_proposals(snapshot, proposals, top_k=args.top_k)
            write_immutable_json(args.output, forecast)
            print(json.dumps({
                "output": args.output,
                "event_id": forecast["event_id"],
                "candidate_count": forecast["candidate_count"],
                "top_k": forecast["top_k"],
                "abstained": forecast["abstained"],
                "forecast_sha256": forecast["forecast_sha256"],
                "score_kind": "uncalibrated_heuristic_ranking_score_not_probability",
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-semantic-bundle":
            import json
            from .idea_layer.semantic_forecast_bundle import build_semantic_forecast_evaluation_bundle

            bundle = build_semantic_forecast_evaluation_bundle(args.forecast, output_path=args.output)
            print(json.dumps({
                "output": args.output,
                "event_count": len(bundle["forecasts"]),
                "candidate_count": sum(len(row["predictions"]) for row in bundle["forecasts"]),
                "provider": bundle["provider"],
                "model": bundle["model"],
                "report_content_sha256": bundle["report_content_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-rank":
            import json
            from .idea_layer.ranker import rank_pre_speech_ideas

            output_path = Path(args.output)
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite forecast: {output_path}")
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            proposal_set = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
            forecast = rank_pre_speech_ideas(
                snapshot, proposal_set, model_version=args.model_version,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(forecast, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "output": args.output,
                "forecast_id": forecast["forecast_id"],
                "candidate_count": len(forecast["candidates"]),
                "forecast_sha256": forecast["forecast_sha256"],
                "model_version": forecast["model_version"],
                "score_kind": "ranking_score",
            }, ensure_ascii=False, indent=2))
        elif args.command == "idea-news-phrase-backtest":
            import json
            from .idea_layer.news_phrase_backtest import evaluate_retrospective_news_phrase_model

            result = evaluate_retrospective_news_phrase_model(
                args.snapshots, args.events, output_path=args.output,
                train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
                semantic_model=args.semantic_model,
            )
            metrics = result["test_metrics"]
            print(json.dumps({
                "output": args.output,
                "evaluation_mode": result["evaluation_mode"],
                "split_sizes": {key: len(result["split"][key]) for key in (
                    "train_event_ids", "validation_event_ids", "test_event_ids",
                )},
                "test_metrics": metrics,
                "aggregate_error_breakdown_by_model": result["test_diagnostics"]["aggregate_error_breakdown_by_model"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "assistant-review-validate":
            import json
            from .idea_layer.assistant_review_validation import validate_assistant_review_artifacts

            report = validate_assistant_review_artifacts(
                pass_a_paths=args.pass_a,
                mapped_paths=args.mapped,
                key_path=args.key,
                events_path=args.events,
                transcript_packet_path=args.transcripts,
            )
            if args.output:
                output_path = Path(args.output)
                if output_path.exists():
                    raise FileExistsError(f"refusing to overwrite validation report: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                report["output"] = args.output
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "assistant-review-history-forecast":
            import json
            from .idea_layer.review_history_forecast import forecast_history_prevalence_for_review_events

            result = forecast_history_prevalence_for_review_events(
                args.history, args.targets, output_path=args.output, top_k=args.top_k,
            )
            print(json.dumps({
                "output": args.output,
                "evaluation_mode": result["evaluation_mode"],
                "model_version": result["model_version"],
                "target_count": len(result["forecasts"]),
                "abstentions": sum(item["abstained"] for item in result["forecasts"]),
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "assistant-review-news-forecast":
            import json
            from .idea_layer.retrospective_news_backtest import build_retrospective_news_idea_forecasts

            result = build_retrospective_news_idea_forecasts(
                args.snapshots, output_path=args.output, top_k=args.top_k,
            )
            print(json.dumps({
                "output": args.output,
                "evaluation_mode": result["evaluation_mode"],
                "event_count": len(result["forecasts"]),
                "candidate_count": sum(len(row["predictions"]) for row in result["forecasts"]),
                "abstentions": sum(row["abstained"] for row in result["forecasts"]),
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "assistant-review-news-audit":
            import json
            from .idea_layer.retrospective_news_error_audit import audit_retrospective_news_forecast

            result = audit_retrospective_news_forecast(
                snapshots_path=args.snapshots,
                forecast_path=args.forecast,
                agreement_path=args.agreement,
                reference_paths=args.reference,
                output_path=args.output,
                top_k=args.top_k,
            )
            print(json.dumps({
                "output": args.output,
                "audit_status": result["audit_status"],
                "scored_event_count": result["scored_event_count"],
                "forecastable_outcome_miss_count": result["forecastable_outcome_miss_count"],
                "misses_by_lexical_diagnostic_bucket": result["misses_by_lexical_diagnostic_bucket"],
                "interpretation_limits": result["interpretation_limits"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "assistant-review-forecast-score":
            import json
            from .idea_layer.assistant_review_forecast_evaluation import evaluate_assistant_reviewed_forecasts

            result = evaluate_assistant_reviewed_forecasts(
                forecast_path=args.forecast,
                reference_paths=args.reference,
                decisions_path=args.decisions,
                protocol_snapshot_path=args.protocol_snapshot,
                output_path=args.output,
            )
            print(json.dumps({
                "output": args.output,
                "scored_event_count": result["scored_event_count"],
                "candidate_decision_counts": result["candidate_decision_counts"],
                "candidate_agreement": result["candidate_agreement"],
                "outcome_coverage": result["outcome_coverage"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-idea-classifier-evaluate":
            import json
            from .stages.ideas.model import evaluate_context_to_idea_classifier

            result = evaluate_context_to_idea_classifier(
                args.snapshots, args.events, output_path=args.output,
                train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
            )
            print(json.dumps({
                "output": args.output, "example_count": result["example_count"],
                "selected_config": result["selected_config"], "test_metrics": result["test_metrics"],
                "final_fit": result["final_fit"], "report_content_sha256": result["report_content_sha256"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-idea-predict":
            import json
            from .stages.ideas.model import predict_context_ideas

            model_artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
            predictions = predict_context_ideas(snapshot, model_artifact, top_k=args.top_k)
            result = {
                "schema_version": 1,
                "artifact_type": "context-to-idea-prediction",
                "model_report_sha256": model_artifact.get("report_content_sha256"),
                "event_id": snapshot.get("event_id"),
                "event_date": snapshot.get("event_date"),
                "predictions": predictions,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({"output": args.output, "prediction_count": len(predictions)}, indent=2))
        elif args.command == "speaker-idea-evaluate":
            import json
            from .speaker_idea_forecast import evaluate_speaker_idea_forecast

            result = evaluate_speaker_idea_forecast(
                args.input, output_path=args.output, train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )
            split_summary = {
                key: value for key, value in result["split"].items()
                if not key.endswith("_event_ids")
            }
            print(json.dumps({
                "output": args.output,
                "selected_half_life_days": result["selected_half_life_days"],
                "target_extractor": result["target_extractor"],
                "split": split_summary,
                "validation_metrics_by_half_life": result["validation_metrics_by_half_life"],
                "validation_metrics_fixed_day_windows": result["validation_metrics_fixed_day_windows"],
                "test_metrics": result["test_metrics"],
                "test_metric_differences": result["test_metric_differences"],
                "test_error_analysis_at_10": result["test_error_analysis_at_10"],
                "test_probability_diagnostic": result["test_probability_diagnostic"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "speaker-idea-selective-evaluate":
            import json
            from .selective_speaker_idea_forecast import evaluate_selective_speaker_idea_forecast

            result = evaluate_selective_speaker_idea_forecast(
                args.input, output_path=args.output, target_precision=args.target_precision,
                minimum_inner_predictions=args.minimum_inner_predictions,
            )
            config_selection = dict(result["config_selection"])
            config_selection["inner_metrics"] = {
                key: value for key, value in config_selection["inner_metrics"].items()
                if key != "per_date_counts"
            }
            validation_metrics = {
                key: value for key, value in result["validation_metrics"].items()
                if key != "per_date_counts"
            }
            baseline_metrics = {
                key: value for key, value in result["validation_reference_top1_30day"].items()
                if key != "per_date_counts"
            }
            print(json.dumps({
                "output": args.output,
                "model_version": result["model_version"],
                "config_selection": config_selection,
                "split": result["split"],
                "validation_metrics": validation_metrics,
                "validation_pooled_precision_date_cluster_bootstrap": result["validation_pooled_precision_date_cluster_bootstrap"],
                "validation_reference_top1_30day": baseline_metrics,
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "speaker-idea-forecast":
            import json
            from .selective_speaker_idea_forecast import forecast_selective_speaker_ideas

            result = forecast_selective_speaker_ideas(
                args.input, args.as_of, output_path=args.output, window_days=args.window_days,
                min_history_events=args.min_history_events, max_predictions=args.max_predictions,
            )
            print(json.dumps({
                "output": args.output,
                "model_version": result["model_version"],
                "as_of_date": result["as_of_date"],
                "config": result["config"],
                "history_event_count": result["history_event_count"],
                "predictions": result["predictions"],
                "abstained": result["abstained"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-theme-evaluate":
            import json
            from .context_theme_evaluation import evaluate_context_theme_ranker

            result = evaluate_context_theme_ranker(args.snapshots, args.labels, output_path=args.output)
            print(json.dumps({
                "output": args.output,
                "test_metrics": result["test_metrics"],
                "test_predictions": result["test_predictions"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "transcript-idea-audit":
            import json
            from .headline_idea_matching import evaluate_headline_idea_overlap

            result = evaluate_headline_idea_overlap(args.snapshots, args.events, output_path=args.output)
            event_ideas = []
            for event in result["events"]:
                event_ideas.append({
                    "event_id": event["event_id"],
                    "date": event["event_date"],
                    "retrieval_profile": event["retrieval_profile"],
                    "context_retrieved_at": event["context_retrieved_at"],
                    "retrieval_queries": event["retrieval_queries"],
                    "transcript_derived_ideas": [
                        {"idea": idea["candidate_idea"], "occurrences": idea["occurrence_count"],
                         "turn_ids": idea["supporting_turn_ids"],
                         "headline_support": [
                             {"title": source["title"], "publisher": source["publisher"],
                              "published_at": source["published_at"], "url": source["url"],
                              "query_purposes": source["query_purposes"]}
                             for source in idea["context_matches"][:3]
                         ]}
                        for idea in event["transcript_top_ideas"][:10]
                    ],
                    "headline_derived_candidates": [
                        {"idea": idea["candidate_idea"], "source_count": idea["source_count"],
                         "mentioned_by_target": idea["mentioned_by_target"],
                         "transcript_turn_ids": [match["turn_id"] for match in idea["transcript_matches"]],
                         "supporting_sources": [
                             {"title": source["title"], "publisher": source["publisher"],
                              "published_at": source["published_at"], "url": source["url"],
                              "query_purposes": source["query_purposes"]}
                             for source in idea["supporting_sources"][:3]
                         ]}
                        for idea in event["top_ideas"][:10]
                    ],
                })
            print(json.dumps({
                "output": args.output,
                "summary_by_top_k": result["summary_by_top_k"],
                "events": event_ideas,
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-issue-forecast":
            import json
            from .context_issue_forecast import generate_prospective_issue_forecast

            result = generate_prospective_issue_forecast(args.snapshot, args.output)
            print(json.dumps({
                "output": args.output,
                "event_id": result["event_id"],
                "prediction_cutoff_at": result["prediction_cutoff_at"],
                "candidate_extractor": result["candidate_extractor"],
                "candidate_count": result["candidate_count"],
                "candidates": [
                    {"rank": item["rank"], "idea": item["idea"],
                     "source_count": item["source_count"], "supporting_sources": item["supporting_sources"][:3]}
                    for item in result["candidates"][:10]
                ],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-candidates":
            import json
            from .context_candidates import extract_context_candidates, load_context_record

            record = load_context_record(args.input, args.event_id)
            print(json.dumps({
                "event_id": record["event_id"],
                "event_date": record["event_date"],
                "candidate_policy": "heuristic source-derived topics; evidence scores are not probabilities",
                "candidates": extract_context_candidates(record["sources"], record["event_date"]),
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-idea-evaluate":
            import json
            from .context_idea_evaluation import evaluate_context_idea_candidates

            result = evaluate_context_idea_candidates(
                args.snapshots, args.events, output_path=args.output,
                train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
            )
            print(json.dumps({
                "output": args.output,
                "phrase_extractor": result["phrase_extractor"],
                "split": result["split"],
                "test_metrics": result["test_metrics"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-issue-forecast-evaluate":
            import json
            from .context_issue_forecast_evaluation import evaluate_prospective_issue_forecast

            try:
                top_k = tuple(int(value) for value in args.top_k.split(","))
            except ValueError:
                raise ValueError("--top-k must be comma-separated positive integers") from None
            result = evaluate_prospective_issue_forecast(
                args.forecast, args.snapshot, args.events, output_path=args.output, top_k=top_k,
            )
            print(json.dumps({
                "output": args.output,
                "event_id": result["event_id"],
                "target_extractor": result["target_extractor"],
                "forecast_candidate_extractor": result["forecast_candidate_extractor"],
                "metrics": result["metrics"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-candidate-evaluate":
            import json
            from .context_candidate_evaluation import evaluate_context_candidate_ranker

            result = evaluate_context_candidate_ranker(args.snapshots, args.labels, output_path=args.output)
            print(json.dumps({
                "output": args.output,
                "test_metrics": result["test_metrics"],
                "test_predictions": result["test_predictions"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-snapshot-audit":
            import json
            from .context_snapshots import audit_legacy_snapshots

            result = audit_legacy_snapshots(args.snapshots, output_path=args.output)
            print(json.dumps({
                "output": args.output,
                "snapshot_count": result["snapshot_count"],
                "strict_prospective_eligible_count": result["strict_prospective_eligible_count"],
                "issue_event_counts": result["issue_event_counts"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "context-snapshot-validate":
            import json
            from .context_snapshots import validate_prospective_snapshot

            try:
                snapshot = json.loads(Path(args.input).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid snapshot JSON: {exc}") from None
            print(json.dumps(validate_prospective_snapshot(snapshot), ensure_ascii=False, indent=2))
        elif args.command == "context-snapshot-seal":
            import json
            from .context_snapshots import seal_prospective_snapshot

            try:
                snapshot = json.loads(Path(args.input).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid snapshot JSON: {exc}") from None
            sealed = seal_prospective_snapshot(snapshot)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing snapshot: {output}")
            output.write_text(json.dumps(sealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(output), "snapshot_sha256": sealed["snapshot_sha256"]}, indent=2))
        elif args.command == "theme-review-packet":
            import json
            from .theme_review import generate_theme_review_packets

            result = generate_theme_review_packets(
                args.snapshots, args.events, args.taxonomy,
                annotator_a_path=args.annotator_a, annotator_b_path=args.annotator_b,
                key_path=args.key, seed=args.seed, selection_description=args.selection_description,
            )
            print(json.dumps({
                "item_count": result["item_count"],
                "family_count": result["family_count"],
                "annotator_a": args.annotator_a,
                "annotator_b": args.annotator_b,
                "hidden_key": args.key,
                "limitations": "pilot packets only; annotations require independent human review and adjudication",
            }, ensure_ascii=False, indent=2))
        elif args.command == "theme-review-agreement":
            import json
            from .theme_review_evaluation import evaluate_theme_review_agreement

            result = evaluate_theme_review_agreement(
                args.annotator_a, args.annotator_b, args.key, output_path=args.output,
            )
            print(json.dumps({
                "output": args.output,
                "item_count": result["item_count"],
                "overall_family_agreement": result["overall_family_agreement"],
                "per_family": result["per_family"],
                "adjudication_required": result["adjudication_required"],
                "limitations": result["limitations"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "theme-review-adjudication-template":
            import json
            from .theme_review_adjudication import generate_adjudication_template

            result = generate_adjudication_template(
                args.annotator_packet, args.key, output_path=args.output,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "theme-review-adjudicate":
            import json
            from .theme_review_adjudication import export_adjudicated_theme_labels

            result = export_adjudicated_theme_labels(
                args.annotator_a, args.annotator_b, args.key, args.decisions,
                output_path=args.output,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "llm":
            from .hybrid import combine_candidate_mass
            from .llm import CausalLM

            model = CausalLM(
                args.model, download_retries=args.download_retries,
                download_timeout=args.download_timeout, revision=args.revision,
            )
            print(f"Model version: {model.model_id}@{model.model_revision or 'local'}")
            print("Top next-token pieces (probability in the full vocabulary):")
            for piece, probability in model.next_tokens(args.prefix, top=10):
                print(f"{piece!r}\t{probability:.5f}")
            candidates = model.generate_candidates(args.prefix, count=args.candidates, max_new_tokens=args.tokens)
            print("Top next-token candidate mass (normalized only over generated candidates):")
            for token, probability in combine_candidate_mass(candidates, {"llm": 1.0}):
                print(f"{token!r}\t{probability:.5f}")
        elif args.command == "ncm-generate":
            from .ncm import NCMGenerator

            model = NCMGenerator(download_retries=args.download_retries, download_timeout=args.download_timeout)
            for index, text in enumerate(model.generate(count=args.count), 1):
                print(f"candidate {index}: {text}")
        elif args.command == "hybrid-train":
            from .hybrid_train import train_hybrid

            result = train_hybrid(
                args.input,
                args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                stride=args.stride,
                device_name=args.device,
                max_steps=args.max_steps,
                evidence_pairs_path=args.evidence_pairs,
                registry_path=args.registry,
            )
            print(f"Saved SpeechPlan-H version {result['model_version']}: {result['output']}")
        elif args.command == "weak-plan-labels":
            from .weak_plan_labels import create_weak_plan_labels

            print(create_weak_plan_labels(
                args.events, args.speech_records, args.output,
                section_turns=args.section_turns, point_turns=args.point_turns,
                top_terms=args.top_terms,
            ))
        elif args.command == "explicit-plan-train":
            from .explicit_plan import train_explicit_plan_model

            result = train_explicit_plan_model(
                args.events, args.plans, args.output,
                registry_path=args.registry, epochs=args.epochs,
                batch_size=args.batch_size, context_length=args.context_length,
                plan_length=args.plan_length, width=args.width,
                layers=args.layers, heads=args.heads,
                min_frequency=args.min_frequency, max_vocabulary=args.max_vocabulary,
                seed=args.seed, max_steps=args.max_steps,
                device_name=args.device, label_policy=args.label_policy,
                cpu_threads=args.cpu_threads or None,
            )
            print(
                f"Saved explicit plan version {result['model_version']} to {result['output']}; "
                f"held-out loss={result['test']['loss']:.4f}, "
                f"greedy exact path accuracy={result['test']['greedy_exact_path_accuracy']:.4f}"
            )
        elif args.command == "explicit-plan-predict":
            import json
            from .explicit_plan import predict_plan_path

            print(json.dumps(predict_plan_path(
                args.model, args.prefix, device_name=args.device,
                max_new_tokens=args.tokens,
            ), ensure_ascii=False))
        elif args.command == "explicit-plan-evaluate":
            import json
            from .plan_evaluation import evaluate_explicit_plan_model

            print(json.dumps(evaluate_explicit_plan_model(
                args.model, args.events, args.plans, device_name=args.device,
                cpu_threads=args.cpu_threads or None,
            ), ensure_ascii=False, indent=2))
        elif args.command == "hybrid":
            from .hybrid_train import generate_hybrid

            evidence = ()
            if args.as_of or args.context_query:
                if not args.as_of:
                    raise ValueError("--as-of is required for time-filtered context retrieval")
                query = args.context_query or args.question
                if not query:
                    raise ValueError("provide --question or --context-query for context retrieval")
                from .federalregister import search_documents

                evidence = search_documents(args.as_of, query, limit=10, lookback_days=90)
            print(generate_hybrid(
                args.model, args.prefix, question_text=args.question, evidence_records=evidence,
                as_of=args.as_of, max_new_tokens=args.tokens, device_name=args.device,
            ))
        elif args.command == "pretrained-hybrid":
            from .hybrid_model import LLMNCMHybrid
            from .llm import CausalLM
            from .ncm import NCMGenerator

            llm = CausalLM(
                download_retries=args.download_retries,
                download_timeout=args.download_timeout,
                revision=args.llm_revision,
            )
            ncm = NCMGenerator(download_retries=args.download_retries, download_timeout=args.download_timeout)
            model = LLMNCMHybrid(llm, ncm)
            print(
                f"Model versions: LLM={llm.model_id}@{llm.model_revision or 'local'}; "
                f"NCM={ncm.checkpoint_revision}; tokenizer={ncm.tokenizer_revision}"
            )
            candidates, mass = model.predict(
                args.prefix,
                llm_count=args.llm_candidates,
                ncm_count=args.ncm_candidates,
                weights={"llm": args.llm_weight, "ncm": args.ncm_weight},
            )
            print("Combined first-token candidate mass (finite pool; uncalibrated):")
            for token, probability in mass[:10]:
                print(f"{token!r}\t{probability:.5f}")
            print("Candidate scores (mean log-likelihood; incomparable units):")
            for candidate in candidates:
                print(f"LLM={candidate.scores['llm']:.4f} NCM={candidate.scores['ncm']:.4f}\t{candidate.text}")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

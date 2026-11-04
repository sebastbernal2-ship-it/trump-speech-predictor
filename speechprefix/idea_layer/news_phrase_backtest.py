"""Small-data retrospective news-to-issue phrase model.

This is an offline development backtest, not the strict pre-speech pipeline:
its historical Google News snapshots were retrieved after the speeches. The
actual model sees headline text only; transcript-derived labels are read by the
evaluator after predictions and are weak lexical outcomes.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_TOKEN = re.compile(r"[a-z][a-z']{1,}")
_STOPWORDS = frozenset("""a an and are as at be been being by for from had has have he her hers him his i if in into is it its me my of on or our ours she that the their theirs them there these they this those to was we were what when where which who will with you your about after all also amid among around because before between both but can could during each even every first following further how however including just more most new not now only other over said says some such than then through today under until very while why would year years breaking latest live news report update watch video trump donald president campaign administration united states us""".split())
DEFAULT_TOP_K = (4, 10, 25)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from None
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _headline_features(sources: list[dict[str, Any]], event_date: date) -> tuple[set[str], list[dict[str, Any]]]:
    features: set[str] = set()
    eligible_sources = []
    for source in sources:
        title = source.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            published = date.fromisoformat(str(source.get("published_at", ""))[:10])
        except ValueError:
            continue
        # Date-only same-day material is withheld; it cannot be ordered relative to a speech.
        if published >= event_date:
            continue
        tokens = [word for word in _TOKEN.findall(title.casefold()) if word not in _STOPWORDS]
        features.update(f"u:{word}" for word in tokens)
        features.update(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        eligible_sources.append({**source, "_features": set(f"u:{word}" for word in tokens) | set(
            f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )})
    return features, eligible_sources


def _fit_bernoulli_nb(
    train: list[dict[str, Any]], alpha: float = 1.0, min_document_frequency: int = 1
) -> tuple[list[str], dict[str, Any]]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_document_frequency <= 0:
        raise ValueError("min_document_frequency must be positive")
    document_frequency = Counter(feature for event in train for feature in event["features"])
    vocabulary = sorted(feature for feature, count in document_frequency.items() if count >= min_document_frequency)
    labels = sorted(set().union(*(event["target_ids"] for event in train))) if train else []
    document_features = [event["features"] for event in train]
    n_documents = len(train)
    model: dict[str, Any] = {"vocabulary": vocabulary, "labels": {}}
    for label in labels:
        positive_indices = [i for i, event in enumerate(train) if label in event["target_ids"]]
        negative_indices = [i for i, event in enumerate(train) if label not in event["target_ids"]]
        n_positive, n_negative = len(positive_indices), len(negative_indices)
        prior_log_odds = math.log((n_positive + alpha) / (n_negative + alpha))
        base = prior_log_odds
        deltas: dict[str, float] = {}
        for feature in vocabulary:
            positive_count = sum(feature in document_features[i] for i in positive_indices)
            negative_count = sum(feature in document_features[i] for i in negative_indices)
            p_positive = (positive_count + alpha) / (n_positive + 2 * alpha)
            p_negative = (negative_count + alpha) / (n_negative + 2 * alpha)
            absent_delta = math.log1p(-p_positive) - math.log1p(-p_negative)
            present_delta = math.log(p_positive) - math.log(p_negative)
            base += absent_delta
            deltas[feature] = present_delta - absent_delta
        model["labels"][label] = {
            "base_log_odds": base,
            "feature_deltas": deltas,
            "training_event_count": n_positive,
        }
    return vocabulary, model


def _score(model: dict[str, Any], features: set[str]) -> dict[str, float]:
    return {
        label: params["base_log_odds"] + sum(params["feature_deltas"].get(feature, 0.0) for feature in features)
        for label, params in model["labels"].items()
    }


def _fit_multinomial_nb(train: list[dict[str, Any]], alpha: float = 1.0) -> dict[str, Any]:
    """One-vs-rest multinomial NB over per-event headline-document counts."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    vocabulary = sorted(set().union(*(event["feature_counts"] for event in train))) if train else []
    labels = sorted(set().union(*(event["target_ids"] for event in train))) if train else []
    models = {}
    for label in labels:
        positive = [event for event in train if label in event["target_ids"]]
        negative = [event for event in train if label not in event["target_ids"]]
        positive_total = sum(sum(min(count, 3) for count in event["feature_counts"].values()) for event in positive)
        negative_total = sum(sum(min(count, 3) for count in event["feature_counts"].values()) for event in negative)
        weights = {}
        for feature in vocabulary:
            positive_count = sum(min(event["feature_counts"].get(feature, 0), 3) for event in positive)
            negative_count = sum(min(event["feature_counts"].get(feature, 0), 3) for event in negative)
            p_positive = (positive_count + alpha) / (positive_total + alpha * max(1, len(vocabulary)))
            p_negative = (negative_count + alpha) / (negative_total + alpha * max(1, len(vocabulary)))
            weights[feature] = math.log(p_positive / p_negative)
        models[label] = {
            "base_log_odds": math.log((len(positive) + alpha) / (len(negative) + alpha)),
            "feature_weights": weights,
            "training_event_count": len(positive),
        }
    return {"vocabulary": vocabulary, "labels": models}


def _score_multinomial(model: dict[str, Any], feature_counts: Counter[str]) -> dict[str, float]:
    return {
        label: params["base_log_odds"] + sum(
            min(feature_counts.get(feature, 0), 3) * weight
            for feature, weight in params["feature_weights"].items()
        )
        for label, params in model["labels"].items()
    }


def _rank_by_prevalence(train: list[dict[str, Any]]) -> list[str]:
    counts = Counter(label for event in train for label in event["target_ids"])
    return sorted(counts, key=lambda label: (-counts[label], label))


def _training_surface(train: list[dict[str, Any]], label: str) -> str:
    counts = Counter(
        event["label_surfaces"][label]
        for event in train
        if label in event["label_surfaces"]
    )
    return max(counts, key=lambda surface: (counts[surface], surface)) if counts else label


def _headline_proposal_history_hybrid(
    event: dict[str, Any],
    train: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Open-vocabulary headline phrase proposals plus a restrained training-history prior."""
    from ..headline_idea_matching import extract_headline_ideas

    extracted = extract_headline_ideas(event["sources"], event["date"].isoformat())
    extracted_by_id = {row["idea_id"]: row for row in extracted}
    prevalence = Counter(label for row in train for label in row["target_ids"])
    surfaces = {label: _training_surface(train, label) for label in prevalence}
    max_evidence = max((row["evidence_score"] for row in extracted), default=1.0) or 1.0
    candidates = set(prevalence) | set(extracted_by_id)
    ranked = []
    for label in candidates:
        source_candidate = extracted_by_id.get(label)
        evidence_score = float(source_candidate["evidence_score"]) if source_candidate else 0.0
        source_support = evidence_score / max_evidence
        history_prevalence = prevalence[label] / max(1, len(train))
        support_headlines = []
        if source_candidate:
            support_headlines = [
                {
                    "title": source.get("title"),
                    "publisher": source.get("publisher"),
                    "published_at": source.get("published_at"),
                    "source_url": source.get("url"),
                }
                for source in source_candidate["supporting_sources"][:5]
            ]
        ranked.append({
            "idea_id": label,
            "idea": source_candidate["candidate_idea"] if source_candidate else surfaces.get(label, label),
            "ranking_score": round(0.75 * source_support + 0.25 * history_prevalence, 6),
            "headline_evidence_score": round(evidence_score, 6),
            "normalized_headline_support": round(source_support, 6),
            "training_event_prevalence": round(history_prevalence, 6),
            "source_count": source_candidate["source_count"] if source_candidate else 0,
            "publisher_count": source_candidate["publisher_count"] if source_candidate else 0,
            "matched_weak_target": label in event["target_ids"],
            "supporting_headlines": support_headlines,
        })
    ranked.sort(key=lambda row: (-row["ranking_score"], row["idea_id"]))
    return ranked, extracted


def _canonical_issue_term(term: str) -> str:
    aliases = {
        "american": "america", "americans": "america",
        "ukrainian": "ukraine", "ukrainians": "ukraine",
        "russian": "russia", "russians": "russia",
        "chinese": "china", "iranians": "iran", "iranian": "iran",
        "israeli": "israel", "israelis": "israel",
    }
    term = aliases.get(term.casefold(), term.casefold())
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")) and term != "news":
        return term[:-1]
    return term


def _hybrid_lexical_rank(
    event: dict[str, Any], train: list[dict[str, Any]], label_ids: set[str]
) -> list[dict[str, Any]]:
    """Rank train-known ideas with complete headline term support, then a small prevalence tie-break."""
    source_tokens = {
        id(source): {
            _canonical_issue_term(feature[2:])
            for feature in source["_features"]
            if feature.startswith("u:")
        }
        for source in event["sources"]
    }
    prevalence = Counter(label for row in train for label in row["target_ids"])
    ranked = []
    for label in label_ids:
        terms = {_canonical_issue_term(term) for term in label.split()}
        if not terms:
            continue
        term_sources = {
            term: [source for source in event["sources"] if term in source_tokens[id(source)]]
            for term in terms
        }
        # Abstain unless every idea term is grounded in at least one headline.
        if any(not matches for matches in term_sources.values()):
            continue
        cooccurrence = max(
            (len(terms & source_tokens[id(source)]) / len(terms) for source in event["sources"]),
            default=0.0,
        )
        direct_relevance = 0.75 + 0.25 * cooccurrence
        historical_prevalence = prevalence[label] / max(1, len(train))
        score = direct_relevance + 0.1 * historical_prevalence
        supporting = {}
        for matches in term_sources.values():
            for source in matches:
                supporting[id(source)] = source
        citations = [
            {
                "title": source["title"],
                "publisher": source.get("publisher"),
                "published_at": source.get("published_at"),
                "source_url": source.get("source_url"),
            }
            for source in sorted(supporting.values(), key=lambda row: (row.get("published_at") or "", row["title"]))[:5]
        ]
        ranked.append({
            "idea_id": label,
            "idea": _training_surface(train, label),
            "ranking_score": round(score, 6),
            "direct_headline_relevance": round(direct_relevance, 6),
            "training_event_prevalence": round(historical_prevalence, 6),
            "matched_terms": sorted(terms),
            "supporting_headlines": citations,
        })
    return sorted(ranked, key=lambda row: (-row["ranking_score"], row["idea_id"]))


class _LocalContextualEncoder:
    """Optional frozen transformer encoder; loads weights only from local cache/path."""

    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("semantic ranking requires the optional torch and transformers packages") from error
        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)

    def encode(self, texts: list[str]):
        torch = self.torch
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start:start + 16]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                output = self.model(**encoded).last_hidden_state
                attention = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                special = torch.tensor(
                    [self.tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True) for row in encoded["input_ids"]],
                    dtype=output.dtype,
                ).unsqueeze(-1)
                mask = attention * (1 - special)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(vectors, dim=0) if vectors else torch.empty((0, self.model.config.hidden_size))


def _tfidf_knn_rank(
    event: dict[str, Any],
    train: list[dict[str, Any]],
    label_ids: set[str],
    *,
    neighbor_count: int = 5,
    prior_weight: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transfer issue labels from similar earlier headline-context events."""
    document_frequency = Counter(feature for row in train for feature in row["features"])
    idf = {
        feature: math.log((len(train) + 1) / (frequency + 1)) + 1.0
        for feature, frequency in document_frequency.items()
    }
    def norm(features: set[str]) -> float:
        return math.sqrt(sum(idf[feature] ** 2 for feature in features if feature in idf))

    test_features = event["features"] & set(idf)
    test_norm = norm(test_features)
    neighbors = []
    if test_norm:
        for row in train:
            shared = test_features & row["features"]
            denominator = test_norm * norm(row["features"])
            similarity = sum(idf[feature] ** 2 for feature in shared) / denominator if denominator else 0.0
            if similarity > 0:
                neighbors.append({"event": row, "similarity": similarity})
    neighbors.sort(key=lambda item: (-item["similarity"], item["event"]["date"], item["event"]["event_id"]))
    neighbors = neighbors[:neighbor_count]
    similarity_mass = sum(item["similarity"] for item in neighbors)
    prevalence = Counter(label for row in train for label in row["target_ids"])
    ranked = []
    for label in label_ids:
        neighbor_vote = (
            sum(item["similarity"] for item in neighbors if label in item["event"]["target_ids"]) / similarity_mass
            if similarity_mass else prevalence[label] / max(1, len(train))
        )
        historical_prevalence = prevalence[label] / max(1, len(train))
        score = (1 - prior_weight) * neighbor_vote + prior_weight * historical_prevalence
        label_neighbors = [item for item in neighbors if label in item["event"]["target_ids"]]
        supporting_features = set().union(*(item["event"]["features"] for item in label_neighbors)) if label_neighbors else set()
        source_scores = []
        for source in event["sources"]:
            source_features = source["_features"]
            overlap = {feature for feature in source_features & supporting_features if feature in idf}
            evidence_score = sum(idf[feature] for feature in overlap)
            if evidence_score > 0:
                source_scores.append((evidence_score, source))
        source_scores.sort(key=lambda item: (-item[0], item[1].get("published_at") or "", item[1]["title"]))
        ranked.append({
            "idea_id": label,
            "idea": _training_surface(train, label),
            "ranking_score": round(score, 6),
            "similar_event_vote": round(neighbor_vote, 6),
            "training_event_prevalence": round(historical_prevalence, 6),
            "supporting_training_events": [
                {"event_id": item["event"]["event_id"], "event_date": item["event"]["date"].isoformat(),
                 "similarity": round(item["similarity"], 6)}
                for item in label_neighbors[:3]
            ],
            "supporting_headlines": [
                {"title": source["title"], "publisher": source.get("publisher"),
                 "published_at": source.get("published_at"), "source_url": source.get("source_url"),
                 "overlap_weight": round(score, 6)}
                for score, source in source_scores[:3]
            ],
        })
    ranked.sort(key=lambda row: (-row["ranking_score"], row["idea_id"]))
    neighbor_summary = [
        {"event_id": item["event"]["event_id"], "event_date": item["event"]["date"].isoformat(),
         "similarity": round(item["similarity"], 6)}
        for item in neighbors
    ]
    return ranked, neighbor_summary


def _graph_label_propagation_rank(
    event: dict[str, Any],
    train: list[dict[str, Any]],
    label_ids: set[str],
    *,
    neighbor_count: int = 5,
    restart_probability: float = 0.2,
    steps: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Weighted context-event/issue graph with personalized PageRank propagation."""
    document_frequency = Counter(feature for row in train for feature in row["features"])
    idf = {feature: math.log((len(train) + 1) / (frequency + 1)) + 1.0 for feature, frequency in document_frequency.items()}
    all_events = [*train, event]
    query_id = event["event_id"]

    def cosine(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_features = left["features"] & set(idf)
        right_features = right["features"] & set(idf)
        shared = left_features & right_features
        left_norm = math.sqrt(sum(idf[feature] ** 2 for feature in left_features))
        right_norm = math.sqrt(sum(idf[feature] ** 2 for feature in right_features))
        if not shared or not left_norm or not right_norm:
            return 0.0
        return sum(idf[feature] ** 2 for feature in shared) / (left_norm * right_norm)

    adjacency: dict[str, dict[str, float]] = {}
    def add_edge(left: str, right: str, weight: float) -> None:
        if weight <= 0:
            return
        adjacency.setdefault(left, {})[right] = max(adjacency.setdefault(left, {}).get(right, 0.0), weight)
        adjacency.setdefault(right, {})[left] = max(adjacency.setdefault(right, {}).get(left, 0.0), weight)

    query_node = f"event:{query_id}"
    neighbor_rows = []
    for row in all_events:
        node = f"event:{row['event_id']}"
        candidates = [other for other in train if other["event_id"] != row["event_id"]]
        ranked = sorted(
            ((cosine(row, other), other) for other in candidates),
            key=lambda item: (-item[0], item[1]["date"], item[1]["event_id"]),
        )
        selected = [(score, other) for score, other in ranked[:neighbor_count] if score > 0]
        for score, other in selected:
            other_node = f"event:{other['event_id']}"
            add_edge(node, other_node, score)
            if node == query_node:
                neighbor_rows.append({"event_id": other["event_id"], "event_date": other["date"].isoformat(), "similarity": round(score, 6)})
    for row in train:
        event_node = f"event:{row['event_id']}"
        label_weight = 1.0 / max(1, len(row["target_ids"]))
        for label in row["target_ids"]:
            add_edge(event_node, f"idea:{label}", label_weight)

    if not adjacency.get(query_node):
        counts = Counter(label for row in train for label in row["target_ids"])
        fallback = [
            {"idea_id": label, "idea": _training_surface(train, label),
             "ranking_score": round(counts[label] / max(1, len(train)), 6),
             "matched_weak_target": label in event["target_ids"], "supporting_training_events": [],
             "supporting_headlines": []}
            for label in label_ids
        ]
        fallback.sort(key=lambda row: (-row["ranking_score"], row["idea_id"]))
        return fallback, [], "prevalence_fallback_no_context_edges"

    rank = {node: 0.0 for node in adjacency}
    rank[query_node] = 1.0
    for _ in range(steps):
        next_rank = {node: 0.0 for node in adjacency}
        next_rank[query_node] = restart_probability
        for node, mass in rank.items():
            neighbors = adjacency.get(node, {})
            total_weight = sum(neighbors.values())
            if total_weight <= 0:
                next_rank[node] = next_rank.get(node, 0.0) + (1 - restart_probability) * mass
                continue
            for neighbor, weight in neighbors.items():
                next_rank[neighbor] = next_rank.get(neighbor, 0.0) + (1 - restart_probability) * mass * weight / total_weight
        rank = next_rank

    ranked = []
    for label in label_ids:
        label_node = f"idea:{label}"
        supporting_events = []
        for neighbor in neighbor_rows:
            train_row = next(row for row in train if row["event_id"] == neighbor["event_id"])
            if label in train_row["target_ids"]:
                supporting_events.append(neighbor)
        supporting_features = set().union(*(
            next(row for row in train if row["event_id"] == neighbor["event_id"])["features"]
            for neighbor in supporting_events
        )) if supporting_events else set()
        source_rows = []
        for source in event["sources"]:
            overlap = source["_features"] & supporting_features
            weight = sum(idf.get(feature, 0.0) for feature in overlap)
            if weight > 0:
                source_rows.append((weight, source))
        source_rows.sort(key=lambda item: (-item[0], item[1].get("published_at") or "", item[1]["title"]))
        ranked.append({
            "idea_id": label,
            "idea": _training_surface(train, label),
            "ranking_score": round(rank.get(label_node, 0.0), 8),
            "matched_weak_target": label in event["target_ids"],
            "supporting_training_events": supporting_events[:3],
            "supporting_headlines": [
                {"title": source["title"], "publisher": source.get("publisher"),
                 "published_at": source.get("published_at"), "source_url": source.get("source_url"),
                 "overlap_weight": round(weight, 6)}
                for weight, source in source_rows[:3]
            ],
        })
    ranked.sort(key=lambda row: (-row["ranking_score"], row["idea_id"]))
    return ranked, neighbor_rows, "personalized_pagerank"


def _query_conditioned_type_path_rank(
    event: dict[str, Any],
    train: list[dict[str, Any]],
    label_ids: set[str],
    history_ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score candidate-specific event→headline→concept←idea and history paths.

    This borrows query-conditioned path selection and typed relations, not a
    trainable Q-GNN architecture. Candidate-specific attention is normalized
    across the current event's headlines.
    """
    term_event_df = Counter()
    for row in train:
        terms = {
            _canonical_issue_term(feature[2:])
            for feature in row["features"]
            if feature.startswith("u:")
        }
        term_event_df.update(terms)
    source_terms = {
        id(source): {
            _canonical_issue_term(feature[2:])
            for feature in source["_features"]
            if feature.startswith("u:")
        }
        for source in event["sources"]
    }
    history_by_label = {row["idea_id"]: row for row in history_ranking}
    ranked = []
    for label in label_ids:
        terms = {_canonical_issue_term(term) for term in label.split()}
        if not terms:
            continue
        weights = {
            term: math.log((len(train) + 1) / (term_event_df.get(term, 0) + 1)) + 1.0
            for term in terms
        }
        total_weight = sum(weights.values())
        coverage_by_source = {
            id(source): sum(weight for term, weight in weights.items() if term in source_terms[id(source)]) / total_weight
            for source in event["sources"]
        }
        query_terms = set().union(*source_terms.values()) if source_terms else set()
        query_coverage = sum(weight for term, weight in weights.items() if term in query_terms) / total_weight
        best_headline_coverage = max(coverage_by_source.values(), default=0.0)
        # Query-conditioned attention: each candidate attends to the headlines
        # that contain its own concept terms, not one shared event summary.
        if coverage_by_source:
            max_coverage = max(coverage_by_source.values())
            exp_weights = {
                source_id: math.exp(5.0 * (coverage - max_coverage))
                for source_id, coverage in coverage_by_source.items()
            }
            normalizer = sum(exp_weights.values())
            attended_coverage = sum(
                exp_weights[source_id] * coverage_by_source[source_id]
                for source_id in coverage_by_source
            ) / normalizer
        else:
            attended_coverage = 0.0
        direct_path_score = 0.5 * query_coverage + 0.3 * best_headline_coverage + 0.2 * attended_coverage
        history_row = history_by_label.get(label, {})
        history_path_score = float(history_row.get("similar_event_vote", 0.0))
        score = 0.75 * direct_path_score + 0.25 * history_path_score
        supporting_headlines = []
        for source in event["sources"]:
            matched = sorted(terms & source_terms[id(source)])
            if matched:
                supporting_headlines.append({
                    "title": source["title"], "publisher": source.get("publisher"),
                    "published_at": source.get("published_at"), "source_url": source.get("source_url"),
                    "matched_concepts": matched,
                    "candidate_conditioned_attention": round(coverage_by_source[id(source)], 6),
                })
        supporting_headlines.sort(key=lambda row: (-row["candidate_conditioned_attention"], row["published_at"] or "", row["title"]))
        ranked.append({
            "idea_id": label,
            "idea": _training_surface(train, label),
            "ranking_score": round(score, 6),
            "direct_query_path_score": round(direct_path_score, 6),
            "history_path_score": round(history_path_score, 6),
            "idea_concepts": sorted(terms),
            "query_coverage": round(query_coverage, 6),
            "best_headline_coverage": round(best_headline_coverage, 6),
            "history_supporting_events": history_row.get("supporting_training_events", []),
            "supporting_headlines": supporting_headlines[:3],
        })
    return sorted(ranked, key=lambda row: (-row["ranking_score"], row["idea_id"]))


def _semantic_rank(event: dict[str, Any], train: list[dict[str, Any]], label_ids: list[str], encoder: _LocalContextualEncoder, label_vectors):
    titles = [source["title"] for source in event["sources"]]
    if not titles:
        return []
    title_vectors = encoder.encode(titles)
    similarities = label_vectors @ title_vectors.T
    prevalence = Counter(label for row in train for label in row["target_ids"])
    ranked = []
    for index, label in enumerate(label_ids):
        values, indices = similarities[index].sort(descending=True)
        evidence = [
            {
                "title": event["sources"][int(source_index)]["title"],
                "publisher": event["sources"][int(source_index)].get("publisher"),
                "published_at": event["sources"][int(source_index)].get("published_at"),
                "source_url": event["sources"][int(source_index)].get("source_url"),
                "cosine_similarity": round(float(value), 6),
            }
            for value, source_index in zip(values[:3], indices[:3])
        ]
        mean_top_similarity = float(values[: min(3, len(values))].mean())
        historical_prevalence = prevalence[label] / max(1, len(train))
        ranked.append({
            "idea_id": label,
            "idea": _training_surface(train, label),
            "ranking_score": round(mean_top_similarity + 0.05 * historical_prevalence, 6),
            "mean_top_3_headline_cosine": round(mean_top_similarity, 6),
            "training_event_prevalence": round(historical_prevalence, 6),
            "supporting_headlines": evidence,
        })
    return sorted(ranked, key=lambda row: (-row["ranking_score"], row["idea_id"]))


def _event_metrics(ranking: list[str], targets: set[str], top_k: tuple[int, ...]) -> dict[str, float]:
    result = {}
    for k in top_k:
        hits = len(set(ranking[:k]) & targets)
        result[f"precision_at_{k}"] = hits / k
        result[f"recall_at_{k}"] = hits / len(targets) if targets else 0.0
    relevance = [int(label in targets) for label in ranking[:10]]
    dcg = sum(rel / math.log2(position + 2) for position, rel in enumerate(relevance))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(10, len(targets))))
    result["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return result


def _mean_metrics(rows: list[dict[str, float]], top_k: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    keys = [f"{metric}_at_{k}" for k in top_k for metric in ("precision", "recall")] + ["ndcg_at_10"]
    return {"events": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}


def _ranking_error_breakdown(
    ranking: list[str],
    targets: set[str],
    training_label_ids: set[str],
    top_k: tuple[int, ...],
    *,
    train: list[dict[str, Any]],
    target_surfaces: dict[str, str],
    score_map: dict[str, float],
    explanation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positions = {label: rank for rank, label in enumerate(ranking, 1)}
    explanations = {row.get("idea_id"): row for row in explanation_rows}
    def idea(label: str) -> str:
        return target_surfaces.get(label) or _training_surface(train, label)
    def candidate_record(label: str) -> dict[str, Any]:
        row = explanations.get(label, {})
        return {
            "idea_id": label,
            "idea": idea(label),
            "rank": positions.get(label),
            "ranking_score": score_map.get(label),
            "explanation": row,
        }
    target_ranks = []
    for label in sorted(targets):
        rank = positions.get(label)
        if label not in positions and label not in training_label_ids:
            status = "unseen_in_training_and_not_proposed"
        elif label not in positions:
            status = "filtered_or_not_generated_by_model"
        elif rank > max(top_k):
            status = "ranked_beyond_reported_cutoffs"
        else:
            status = "ranked_within_reported_cutoffs"
        target_ranks.append({
            **candidate_record(label),
            "status": status,
            "unseen_in_training_vocabulary": label not in training_label_ids,
        })
    at_k = {}
    for k in top_k:
        predicted = ranking[:k]
        hits = set(predicted) & targets
        false_positives = [label for label in predicted if label not in targets]
        missed = sorted(targets - set(predicted))
        missed_rows = []
        for label in missed:
            if label not in positions and label not in training_label_ids:
                reason = "unseen_in_training_and_not_proposed"
            elif label not in positions:
                reason = "filtered_or_not_generated_by_model"
            else:
                reason = "ranked_below_cutoff"
            missed_rows.append({
                **candidate_record(label),
                "miss_reason": reason,
                "unseen_in_training_vocabulary": label not in training_label_ids,
            })
        at_k[str(k)] = {
            "hit_count": len(hits),
            "matched_weak_targets": [candidate_record(label) for label in sorted(hits, key=lambda value: positions[value])],
            "unmatched_predictions": [candidate_record(label) for label in false_positives],
            "missed_weak_targets": missed_rows,
            "prediction_slots_filled": len(predicted),
            "abstained_slots": max(0, k - len(predicted)),
        }
    return {
        "candidate_count": len(ranking),
        "target_count": len(targets),
        "targets_unseen_in_training": sum(label not in training_label_ids for label in targets),
        "targets_represented_in_candidate_list": sum(label in positions for label in targets),
        "targets_filtered_or_not_generated": sum(label in training_label_ids and label not in positions for label in targets),
        "target_ranks": target_ranks,
        "at_k": at_k,
    }


def _aggregate_error_breakdown(
    per_event: list[dict[str, Any]],
    model_names: list[str],
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    aggregate = {}
    for model_name in model_names:
        by_k = {}
        for k in top_k:
            hits = false_positives = target_count = represented = unseen = filtered = below_cutoff = events_with_hit = 0
            for event in per_event:
                detail = event["models"][model_name]
                item = detail["at_k"][str(k)]
                hits += item["hit_count"]
                false_positives += len(item["unmatched_predictions"])
                target_count += detail["target_count"]
                represented += detail["targets_represented_in_candidate_list"]
                unseen += sum(row["miss_reason"] == "unseen_in_training_and_not_proposed" for row in item["missed_weak_targets"])
                filtered += sum(row["miss_reason"] == "filtered_or_not_generated_by_model" for row in item["missed_weak_targets"])
                below_cutoff += sum(row["miss_reason"] == "ranked_below_cutoff" for row in item["missed_weak_targets"])
                events_with_hit += bool(item["hit_count"])
            by_k[str(k)] = {
                "hits": hits,
                "unmatched_predictions": false_positives,
                "missed_weak_targets": target_count - hits,
                "unseen_target_misses": unseen,
                "evidence_filter_or_generation_misses": filtered,
                "ranked_below_cutoff_misses": below_cutoff,
                "events_with_at_least_one_hit": events_with_hit,
                "event_count": len(per_event),
                "weak_target_count": target_count,
                "candidate_inventory_coverage_count": represented,
                "candidate_inventory_coverage": represented / target_count if target_count else 0.0,
                "precision": hits / (k * len(per_event)) if per_event else 0.0,
                "recall": hits / target_count if target_count else 0.0,
            }
        aggregate[model_name] = {"at_k": by_k}
    return aggregate


def _top_k_jaccard(per_event: list[dict[str, Any]], model_names: list[str], k: int = 10) -> dict[str, Any]:
    pairs = {}
    for left_index, left in enumerate(model_names):
        for right in model_names[left_index + 1:]:
            values = []
            for event in per_event:
                a = set(event["model_rankings"][left][:k])
                b = set(event["model_rankings"][right][:k])
                union = a | b
                values.append(len(a & b) / len(union) if union else 1.0)
            pairs[f"{left}__vs__{right}"] = sum(values) / len(values) if values else 0.0
    return pairs


def evaluate_retrospective_news_phrase_model(
    snapshots_path: str | Path,
    event_records_path: str | Path,
    *,
    output_path: str | Path | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
    semantic_model: str | None = None,
) -> dict[str, Any]:
    """Fit news-token -> weak issue-phrase Bernoulli NB and test chronologically.

    The validation fold is reserved but not used to tune this fixed alpha=1
    baseline. This makes a small, auditable train/test demonstration without
    claiming that retrospective source snapshots were available at the time.
    """
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction and validation_fraction must define nonempty chronological partitions")
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k must contain positive values")
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite backtest report: {output_path}")

    # Imported only by the post-event evaluator. The fitted predictor below sees headlines, not turns.
    from ..headline_idea_matching import (
        ISSUE_PHRASE_EXTRACTOR_VERSION,
        _target_turns,
        extract_transcript_ideas,
        issue_phrase_lexicon_sha256,
    )

    snapshots = _read_jsonl(snapshots_path)
    records = _read_jsonl(event_records_path)
    snapshot_by_id = {row.get("event_id"): row for row in snapshots}
    records_by_id = {row.get("event_id"): row for row in records}
    if len(snapshot_by_id) != len(snapshots) or len(records_by_id) != len(records):
        raise ValueError("duplicate event IDs in snapshots or outcome records")
    if not snapshots or not set(snapshot_by_id) <= set(records_by_id):
        raise ValueError("every snapshot needs a matching event record")

    events = []
    for event_id, snapshot in snapshot_by_id.items():
        record = records_by_id[event_id]
        event_date_text = snapshot.get("event_date")
        if event_date_text != record.get("event_date"):
            raise ValueError(f"event date mismatch for {event_id}")
        try:
            event_date = date.fromisoformat(event_date_text)
        except (TypeError, ValueError):
            raise ValueError(f"invalid event date for {event_id}") from None
        sources = snapshot.get("candidate_sources")
        if not isinstance(sources, list):
            raise ValueError(f"candidate_sources must be a list for {event_id}")
        features, eligible_sources = _headline_features(sources, event_date)
        feature_counts = Counter(feature for source in eligible_sources for feature in source["_features"])
        turns = _target_turns(record)
        if not turns:
            continue
        ideas = extract_transcript_ideas(turns)
        label_surfaces: dict[str, str] = {}
        for idea in ideas:
            label_surfaces[idea["idea_id"]] = idea["candidate_idea"]
        events.append({
            "event_id": event_id,
            "date": event_date,
            "features": features,
            "feature_counts": feature_counts,
            "sources": eligible_sources,
            "target_ids": set(label_surfaces),
            "label_surfaces": label_surfaces,
            "retrieved_at": snapshot.get("retrieved_at"),
        })
    events.sort(key=lambda event: (event["date"], event["event_id"]))
    dates = sorted({event["date"] for event in events})
    if len(dates) < 10:
        raise ValueError("need at least 10 distinct event dates for chronological train/validation/test")
    train_end = min(len(dates) - 2, max(1, int(len(dates) * train_fraction)))
    validation_end = min(len(dates) - 1, max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction))))
    train_dates, validation_dates, test_dates = set(dates[:train_end]), set(dates[train_end:validation_end]), set(dates[validation_end:])
    train = [event for event in events if event["date"] in train_dates]
    validation = [event for event in events if event["date"] in validation_dates]
    test = [event for event in events if event["date"] in test_dates]
    if not train or not validation or not test:
        raise ValueError("chronological split produced an empty partition")

    vocabulary, model = _fit_bernoulli_nb(train, alpha=1.0)
    filtered_vocabulary, filtered_model = _fit_bernoulli_nb(train, alpha=1.0, min_document_frequency=2)
    multinomial_model = _fit_multinomial_nb(train, alpha=1.0)
    training_label_ids = set(model["labels"])
    prevalence = _rank_by_prevalence(train)
    semantic_encoder = _LocalContextualEncoder(semantic_model) if semantic_model else None
    semantic_label_ids = sorted(training_label_ids)
    semantic_label_vectors = (
        semantic_encoder.encode([_training_surface(train, label) for label in semantic_label_ids])
        if semantic_encoder else None
    )
    model_metrics, filtered_model_metrics, multinomial_metrics, baseline_metrics, hybrid_metrics, headline_proposal_metrics, knn_metrics, graph_metrics, query_path_metrics, semantic_metrics, predictions = [], [], [], [], [], [], [], [], [], [], []
    hybrid_covered_events = 0
    diagnostic_events = []
    headline_proposal_target_coverage = 0
    headline_proposal_target_count = 0
    unseen_target_count = 0
    total_target_count = 0
    for event in test:
        scores = _score(model, event["features"] & set(vocabulary))
        ranking = sorted(scores, key=lambda label: (-scores[label], label))
        filtered_scores = _score(filtered_model, event["features"] & set(filtered_vocabulary))
        filtered_ranking = sorted(filtered_scores, key=lambda label: (-filtered_scores[label], label))
        multinomial_scores = _score_multinomial(multinomial_model, event["feature_counts"])
        multinomial_ranking = sorted(multinomial_scores, key=lambda label: (-multinomial_scores[label], label))
        baseline_ranking = list(prevalence)
        model_metric = _event_metrics(ranking, event["target_ids"], top_k)
        filtered_model_metric = _event_metrics(filtered_ranking, event["target_ids"], top_k)
        multinomial_metric = _event_metrics(multinomial_ranking, event["target_ids"], top_k)
        baseline_metric = _event_metrics(baseline_ranking, event["target_ids"], top_k)
        hybrid_ranking = _hybrid_lexical_rank(event, train, training_label_ids)
        hybrid_ids = [row["idea_id"] for row in hybrid_ranking]
        hybrid_metric = _event_metrics(hybrid_ids, event["target_ids"], top_k)
        headline_proposal_ranking, extracted_headline_proposals = _headline_proposal_history_hybrid(event, train)
        headline_proposal_ids = [row["idea_id"] for row in headline_proposal_ranking]
        headline_proposal_metric = _event_metrics(headline_proposal_ids, event["target_ids"], top_k)
        extracted_headline_proposal_ids = {row["idea_id"] for row in extracted_headline_proposals}
        headline_proposal_target_coverage += len(extracted_headline_proposal_ids & event["target_ids"])
        headline_proposal_target_count += len(event["target_ids"])
        knn_ranking, knn_neighbors = _tfidf_knn_rank(event, train, training_label_ids)
        knn_metric = _event_metrics([row["idea_id"] for row in knn_ranking], event["target_ids"], top_k)
        graph_ranking, graph_neighbors, graph_mode = _graph_label_propagation_rank(event, train, training_label_ids)
        graph_metric = _event_metrics([row["idea_id"] for row in graph_ranking], event["target_ids"], top_k)
        query_path_ranking = _query_conditioned_type_path_rank(event, train, training_label_ids, knn_ranking)
        query_path_metric = _event_metrics([row["idea_id"] for row in query_path_ranking], event["target_ids"], top_k)
        for rank_index, row in enumerate(query_path_ranking[:max(top_k)], 1):
            row["rank"] = rank_index
            row["matched_weak_target"] = row["idea_id"] in event["target_ids"]
        semantic_ranking = (
            _semantic_rank(event, train, semantic_label_ids, semantic_encoder, semantic_label_vectors)
            if semantic_encoder else []
        )
        semantic_ids = [row["idea_id"] for row in semantic_ranking]
        semantic_metric = _event_metrics(semantic_ids, event["target_ids"], top_k) if semantic_encoder else None
        model_metrics.append(model_metric)
        filtered_model_metrics.append(filtered_model_metric)
        multinomial_metrics.append(multinomial_metric)
        baseline_metrics.append(baseline_metric)
        hybrid_metrics.append(hybrid_metric)
        headline_proposal_metrics.append(headline_proposal_metric)
        knn_metrics.append(knn_metric)
        graph_metrics.append(graph_metric)
        query_path_metrics.append(query_path_metric)
        if semantic_metric is not None:
            semantic_metrics.append(semantic_metric)
        hybrid_covered_events += bool(hybrid_ranking)
        unseen = sorted(event["target_ids"] - training_label_ids)
        unseen_target_count += len(unseen)
        total_target_count += len(event["target_ids"])
        ranked = []
        for rank, label in enumerate(ranking[:max(top_k)], 1):
            params = model["labels"][label]
            positive_features = sorted(
                (feature for feature in event["features"] if params["feature_deltas"].get(feature, 0.0) > 0),
                key=lambda feature: (-params["feature_deltas"][feature], feature),
            )[:5]
            support_sources = []
            for source in event["sources"]:
                overlap = len(set(positive_features) & source["_features"])
                if overlap:
                    support_sources.append({
                        "title": source["title"],
                        "publisher": source.get("publisher"),
                        "published_at": source.get("published_at"),
                        "source_url": source.get("source_url"),
                        "overlap_with_top_model_features": overlap,
                    })
            support_sources.sort(key=lambda row: (-row["overlap_with_top_model_features"], row["published_at"] or "", row["title"]))
            ranked.append({
                "rank": rank,
                "idea_id": label,
                "idea": _training_surface(train, label),
                "ranking_score_log_odds": round(scores[label], 6),
                "matched_weak_target": label in event["target_ids"],
                "top_headline_features": positive_features,
                "supporting_headlines": support_sources[:3],
            })
        filtered_ranked = []
        for rank, label in enumerate(filtered_ranking[:max(top_k)], 1):
            params = filtered_model["labels"][label]
            positive_features = sorted(
                (feature for feature in event["features"] if params["feature_deltas"].get(feature, 0.0) > 0),
                key=lambda feature: (-params["feature_deltas"][feature], feature),
            )[:5]
            support_sources = []
            for source in event["sources"]:
                overlap = len(set(positive_features) & source["_features"])
                if overlap:
                    support_sources.append({
                        "title": source["title"], "publisher": source.get("publisher"),
                        "published_at": source.get("published_at"), "source_url": source.get("source_url"),
                        "overlap_with_top_model_features": overlap,
                    })
            support_sources.sort(key=lambda row: (-row["overlap_with_top_model_features"], row["published_at"] or "", row["title"]))
            filtered_ranked.append({
                "rank": rank, "idea_id": label, "idea": _training_surface(train, label),
                "ranking_score_log_odds": round(filtered_scores[label], 6),
                "matched_weak_target": label in event["target_ids"],
                "top_headline_features": positive_features,
                "supporting_headlines": support_sources[:3],
            })
        multinomial_top10 = []
        for rank_index, label in enumerate(multinomial_ranking[:max(top_k)], 1):
            params = multinomial_model["labels"][label]
            contributing = sorted(
                ((feature, event["feature_counts"].get(feature, 0) * weight)
                 for feature, weight in params["feature_weights"].items()
                 if event["feature_counts"].get(feature, 0) and weight > 0),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            contributing_features = {feature for feature, _ in contributing}
            support = []
            for source in event["sources"]:
                overlap = len(contributing_features & source["_features"])
                if overlap:
                    support.append({
                        "title": source["title"], "publisher": source.get("publisher"),
                        "published_at": source.get("published_at"), "source_url": source.get("source_url"),
                        "overlap_with_positive_model_features": overlap,
                    })
            support.sort(key=lambda row: (-row["overlap_with_positive_model_features"], row["published_at"] or "", row["title"]))
            multinomial_top10.append({
                "rank": rank_index, "idea_id": label, "idea": _training_surface(train, label),
                "ranking_score_log_odds": round(multinomial_scores[label], 6),
                "matched_weak_target": label in event["target_ids"],
                "top_positive_features": [feature for feature, _ in contributing],
                "supporting_headlines": support[:3],
            })
        model_rankings = {
            "bernoulli_nb": ranking,
            "bernoulli_nb_min_df2": filtered_ranking,
            "multinomial_nb": multinomial_ranking,
            "lexical_hybrid": hybrid_ids,
            "headline_proposals_plus_history": headline_proposal_ids,
            "tfidf_event_knn": [row["idea_id"] for row in knn_ranking],
            "ppr_graph": [row["idea_id"] for row in graph_ranking],
            "query_conditioned_paths": [row["idea_id"] for row in query_path_ranking],
            "training_prevalence": baseline_ranking,
        }
        prevalence_counts = Counter(label for row in train for label in row["target_ids"])
        model_scores = {
            "bernoulli_nb": scores,
            "bernoulli_nb_min_df2": filtered_scores,
            "multinomial_nb": multinomial_scores,
            "lexical_hybrid": {row["idea_id"]: row["ranking_score"] for row in hybrid_ranking},
            "headline_proposals_plus_history": {row["idea_id"]: row["ranking_score"] for row in headline_proposal_ranking},
            "tfidf_event_knn": {row["idea_id"]: row["ranking_score"] for row in knn_ranking},
            "ppr_graph": {row["idea_id"]: row["ranking_score"] for row in graph_ranking},
            "query_conditioned_paths": {row["idea_id"]: row["ranking_score"] for row in query_path_ranking},
            "training_prevalence": {label: prevalence_counts[label] for label in prevalence},
        }
        model_explanations = {
            "bernoulli_nb": ranked,
            "bernoulli_nb_min_df2": filtered_ranked,
            "multinomial_nb": multinomial_top10,
            "lexical_hybrid": hybrid_ranking,
            "headline_proposals_plus_history": headline_proposal_ranking,
            "tfidf_event_knn": knn_ranking,
            "ppr_graph": graph_ranking,
            "query_conditioned_paths": query_path_ranking,
            "training_prevalence": [],
        }
        if semantic_encoder:
            model_rankings["frozen_encoder"] = semantic_ids
            model_scores["frozen_encoder"] = {row["idea_id"]: row["ranking_score"] for row in semantic_ranking}
            model_explanations["frozen_encoder"] = semantic_ranking
        model_names = list(model_rankings)
        diagnostic_event = {
            "event_id": event["event_id"],
            "event_date": event["date"].isoformat(),
            "weak_target_ideas": [
                {"idea_id": label, "idea": event["label_surfaces"].get(label, _training_surface(train, label))}
                for label in sorted(event["target_ids"])
            ],
            "models": {
                name: _ranking_error_breakdown(
                    model_rankings[name], event["target_ids"], training_label_ids, top_k,
                    train=train, target_surfaces=event["label_surfaces"], score_map=model_scores[name],
                    explanation_rows=model_explanations[name],
                )
                for name in model_names
            },
            "model_rankings": {name: ids[:max(top_k)] for name, ids in model_rankings.items()},
        }
        diagnostic_events.append(diagnostic_event)
        predictions.append({
            "event_id": event["event_id"],
            "event_date": event["date"].isoformat(),
            "snapshot_retrieved_at": event["retrieved_at"],
            "target_phrase_count": len(event["target_ids"]),
            "target_phrase_ids": sorted(event["target_ids"]),
            "target_phrases_unseen_in_training": unseen,
            "model_top_10": ranked[:10],
            "model_top_k": ranked,
            "bernoulli_nb_min_df2_top_10": filtered_ranked[:10],
            "bernoulli_nb_min_df2_top_k": filtered_ranked,
            "multinomial_nb_top_10": multinomial_top10[:10],
            "multinomial_nb_top_k": multinomial_top10,
            "hybrid_lexical_top_10": hybrid_ranking[:10],
            "hybrid_supported_candidate_count": len(hybrid_ranking),
            "headline_proposals_plus_history_top_10": headline_proposal_ranking[:10],
            "headline_proposal_candidate_count": len(extracted_headline_proposals),
            "headline_proposal_target_hits_available": sorted(extracted_headline_proposal_ids & event["target_ids"]),
            "hybrid_abstained": not bool(hybrid_ranking),
            "historical_context_knn_top_10": knn_ranking[:10],
            "historical_context_knn_neighbors": knn_neighbors,
            "graph_label_propagation_top_10": graph_ranking[:10],
            "graph_context_neighbors": graph_neighbors,
            "graph_inference_mode": graph_mode,
            "query_conditioned_type_path_top_10": query_path_ranking[:10],
            "query_conditioned_type_path_top_k": query_path_ranking[:max(top_k)],
            "semantic_encoder_top_10": semantic_ranking[:10] if semantic_encoder else None,
            "training_prevalence_top_10": [
                {"rank": rank, "idea_id": label,
                 "idea": _training_surface(train, label),
                 "matched_weak_target": label in event["target_ids"]}
                for rank, label in enumerate(prevalence[:10], 1)
            ],
        })

    test_metrics = {
        "news_phrase_bernoulli_naive_bayes": _mean_metrics(model_metrics, top_k),
        "news_phrase_bernoulli_min_df2": _mean_metrics(filtered_model_metrics, top_k),
        "news_phrase_multinomial_naive_bayes": _mean_metrics(multinomial_metrics, top_k),
        "hybrid_lexical_relevance_plus_prevalence": {
            **_mean_metrics(hybrid_metrics, top_k),
            "events_with_at_least_one_supported_candidate": hybrid_covered_events,
            "event_coverage": hybrid_covered_events / len(test),
            "mean_supported_candidate_count": sum(len(_hybrid_lexical_rank(event, train, training_label_ids)) for event in test) / len(test),
        },
        "headline_proposals_plus_history": {
            **_mean_metrics(headline_proposal_metrics, top_k),
            "test_weak_targets_proposed_from_headlines": headline_proposal_target_coverage,
            "test_weak_target_count": headline_proposal_target_count,
            "headline_proposal_target_coverage": headline_proposal_target_coverage / headline_proposal_target_count if headline_proposal_target_count else 0.0,
        },
        "historical_context_tfidf_knn": _mean_metrics(knn_metrics, top_k),
        "heterogeneous_context_idea_graph_pagerank": _mean_metrics(graph_metrics, top_k),
        "query_conditioned_typed_path_ranker": _mean_metrics(query_path_metrics, top_k),
        "training_phrase_prevalence": _mean_metrics(baseline_metrics, top_k),
        "test_target_label_coverage_in_training_vocabulary": (
            1.0 - unseen_target_count / total_target_count if total_target_count else 0.0
        ),
        "test_unseen_target_phrase_count": unseen_target_count,
        "test_target_phrase_count": total_target_count,
    }
    if semantic_encoder:
        test_metrics["frozen_transformer_similarity_plus_prevalence"] = _mean_metrics(semantic_metrics, top_k)
    diagnostic_model_names = list(diagnostic_events[0]["models"])
    test_diagnostics = {
        "outcome_match_is_weak_lexical_label_only": True,
        "per_event": diagnostic_events,
        "aggregate_error_breakdown_by_model": _aggregate_error_breakdown(diagnostic_events, diagnostic_model_names, top_k),
        "top_10_prediction_set_jaccard": _top_k_jaccard(diagnostic_events, diagnostic_model_names, k=10),
        "unmatched_prediction_warning": "A prediction not matching the weak phrase extractor is not necessarily a semantic false positive; the extractor may have missed or phrased the idea differently.",
    }

    report = {
        "schema_version": 1,
        "record_type": "idea-layer-retrospective-backtest",
        "model_family": "Bernoulli/multinomial naive Bayes, open-vocabulary headline proposals, lexical relevance, TF-IDF kNN, graph propagation, query-conditioned typed paths, and optional frozen encoder diagnostics",
        "model_version": "news-to-issue-phrase-nb-v1-with-ranking-diagnostics",
        "evaluation_mode": "retrospective-only; not a pre-speech-valid forecast evaluation",
        "optional_semantic_encoder": None if semantic_encoder is None else {
            "model_name": semantic_encoder.model_name,
            "commit_hash": semantic_encoder.commit_hash,
            "pooling": "mean of non-special-token final hidden states; cosine similarity",
            "sentence_similarity_fine_tuned": False,
            "local_files_only": True,
            "history_prior_weight": 0.05,
        },
        "feature_policy": "headline titles published before the event date; unigram and adjacent-bigram presence; no event title or transcript text",
        "target_policy": "automatically extracted compact issue phrases from target-speaker turns; weak lexical labels, not human semantic gold",
        "target_extractor": {
            "version": ISSUE_PHRASE_EXTRACTOR_VERSION,
            "lexicon_sha256": issue_phrase_lexicon_sha256(),
        },
        "config": {
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "alpha": 1.0,
            "bernoulli_min_document_frequency": 2,
            "top_k": list(top_k),
            "validation_used_for_tuning": False,
            "hybrid_abstention": "emit only training phrases whose every normalized term occurs in at least one pre-event-dated headline",
            "hybrid_score": "0.75 + 0.25 * best same-headline term coverage + 0.1 * training-event prevalence",
            "headline_proposal_hybrid": "union of training phrases and cited headline-extracted phrases; 0.75 * normalized source-evidence score + 0.25 * training-event prevalence; not a probability",
            "knn_neighbors": 5,
            "knn_history_prior_weight": 0.25,
            "knn_feature_weights": "binary unigram and adjacent-bigram TF-IDF; IDF fit on training events only",
            "graph_model": "non-neural heterogeneous event/idea graph; top-5 TF-IDF event-context edges; event-idea label edges only for training events; personalized PageRank with 0.2 restart for 8 steps",
            "query_conditioned_paths": ["query-event -> pre-event headline -> concept <- candidate-idea", "query-event -> TF-IDF-similar training-event -> training-idea-label"],
            "query_conditioned_path_weights": {"direct_context_path": 0.75, "historical_event_path": 0.25},
            "query_conditioned_attention": "per-candidate softmax over current-event headlines, temperature=5; no learned parameters",
        },
        "input_sha256": {
            "snapshots": hashlib.sha256(Path(snapshots_path).read_bytes()).hexdigest(),
            "event_records": hashlib.sha256(Path(event_records_path).read_bytes()).hexdigest(),
        },
        "split": {
            "train_event_ids": [event["event_id"] for event in train],
            "validation_event_ids": [event["event_id"] for event in validation],
            "test_event_ids": [event["event_id"] for event in test],
            "train_dates": [day.isoformat() for day in sorted(train_dates)],
            "validation_dates": [day.isoformat() for day in sorted(validation_dates)],
            "test_dates": [day.isoformat() for day in sorted(test_dates)],
        },
        "train_summary": {
            "events": len(train),
            "headline_feature_count": len(vocabulary),
            "headline_feature_count_min_df2": len(filtered_vocabulary),
            "issue_phrase_vocabulary_size": len(training_label_ids),
        },
        "validation_summary": {"events": len(validation), "used_for_tuning": False},
        "test_metrics": test_metrics,
        "test_diagnostics": test_diagnostics,
        "test_predictions": predictions,
        "limitations": [
            "All historical Google News snapshots were retrieved after their events; headline dates do not prove pre-event observation. This backtest is not a valid prospective forecast score.",
            "The 2026 test events have been inspected in earlier project experiments; these numbers are a fixed diagnostic, not an untouched holdout or tuning target.",
            "Transcript phrase labels use a lexicon modified after inspecting examples; they are weak lexical outcomes, not semantic accuracy.",
            "Only two test events are available, and no Trump-post archive or event-detail features are used.",
            "Supervised/history rankers are limited to phrases seen in training; the headline proposal path is limited to a fixed issue lexicon and does not generate arbitrary novel ideas.",
            "The hybrid is exact lexical matching with a small prevalence tie-break, not semantic similarity; stemming and alias handling are deliberately limited.",
            "The event kNN and heterogeneous graph use only 16 training contexts; their fixed neighborhood and propagation settings are diagnostics, not tuned settings.",
            "The PageRank experiment is non-neural and does not test query-conditioned candidate representations. The typed-path scorer is also parameter-free; neither reproduces Q-GNN or Netflix SemanticGNN.",
            "No co-engagement or externally validated relation graph is available; the concept edges are lexical matches to headline titles.",
            "The validation partition is reserved but not used for tuning; alpha and feature extraction are fixed defaults.",
        ],
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return report

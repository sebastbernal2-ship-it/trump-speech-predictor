import hashlib
import unittest

from speechprefix.idea_layer.story_graph import (
    seal_story_graph,
    story_graph_as_of,
    validate_story_graph,
)


class StoryGraphTests(unittest.TestCase):
    def _node(self, node_id, node_type, sources, available, **extra):
        return {
            "node_id": node_id,
            "node_type": node_type,
            "source_ids": sources,
            "availability_status": "verified_before_cutoff",
            "available_at": available,
            "attributes": {},
            **extra,
        }

    def _edge(self, edge_id, source, target, relation, sources, available, **extra):
        return {
            "edge_id": edge_id,
            "source_node_id": source,
            "target_node_id": target,
            "relation": relation,
            "source_ids": sources,
            "availability_status": "verified_before_cutoff",
            "available_at": available,
            "confidence": 1.0,
            **extra,
        }

    def _source(self, source_id, published, observed, availability="verified_before_cutoff"):
        content = f"Evidence for {source_id}"
        row = {
            "schema_version": 1,
            "record_type": "idea-layer-source",
            "source_id": source_id,
            "source_type": "news",
            "source_url": f"https://example.test/{source_id}",
            "publisher": "Example publisher",
            "published_at": published,
            "observed_at": observed,
            "availability_status": availability,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "reuse_terms": "Synthetic test evidence",
        }
        if availability == "verified_before_cutoff":
            row["availability_evidence"] = "test fixture"
            row["timestamp_basis"] = "test fixture timestamp"
        return row

    def _sources(self):
        return [
            self._source("source-1", "2030-01-19T11:00:00Z", "2030-01-19T12:00:00Z"),
            self._source("source-2", "2030-01-21T08:00:00Z", "2030-01-21T09:00:00Z"),
        ]

    def _graph(self):
        early = "2030-01-19T12:00:00Z"
        graph = {
            "schema_version": 1,
            "record_type": "idea-layer-story-graph",
            "metadata": {"algorithm_version": "fixture-v1", "algorithm_fit_mode": "deterministic"},
            "nodes": [
                self._node("article-1", "article_observation", ["source-1"], early,
                           published_at="2030-01-19T11:00:00Z", attributes={"title": "Musk meets Trump"}),
                self._node("story-1", "story", ["source-1"], early,
                           attributes={"story_key": "meeting"}),
                self._node("person-musk", "entity", ["source-1"], early,
                           attributes={"entity_type": "person", "canonical_id": "Q317521"}),
                self._node("claim-role", "entity_claim", ["source-1"], early,
                           valid_from="2030-01-01", valid_until="2030-01-21",
                           attributes={"claim": "holds a specified public role"}),
                self._node("article-2", "article_observation", ["source-2"],
                           "2030-01-21T09:00:00Z", published_at="2030-01-21T08:00:00Z",
                           attributes={"title": "Meeting follow-up with new facts"}),
            ],
            "edges": [
                self._edge("edge-member", "article-1", "story-1", "member_of", ["source-1"], early),
                self._edge("edge-mention", "article-1", "person-musk", "mentions", ["source-1"], early),
                self._edge("edge-claim", "claim-role", "person-musk", "claim_about", ["source-1"], early),
                self._edge("edge-update", "article-2", "story-1", "updates", ["source-1", "source-2"],
                           "2030-01-21T09:00:00Z", confidence=0.75),
            ],
        }
        return seal_story_graph(graph)

    def test_validation_and_as_of_graph_exclude_future_updates(self):
        graph = self._graph()
        self.assertEqual(validate_story_graph(graph)["node_count"], 5)
        sliced = story_graph_as_of(
            graph, "2030-01-20T16:00:00Z", source_evidence=self._sources(),
        )
        self.assertEqual(
            {node["node_id"] for node in sliced["nodes"]},
            {"article-1", "story-1", "person-musk", "claim-role"},
        )
        self.assertEqual(
            {edge["edge_id"] for edge in sliced["edges"]},
            {"edge-member", "edge-mention", "edge-claim"},
        )
        self.assertTrue(sliced["metadata"]["point_in_time_verified"])
        self.assertEqual(validate_story_graph(sliced)["edge_count"], 3)

    def test_source_ids_resolve_to_immutable_evidence(self):
        graph = self._graph()
        self.assertEqual(validate_story_graph(graph, self._sources())["node_count"], 5)
        with self.assertRaisesRegex(ValueError, "missing source evidence"):
            validate_story_graph(graph, self._sources()[:1])
        as_of = story_graph_as_of(
            graph, "2030-01-20T16:00:00Z", source_evidence=self._sources(),
        )
        changed_terms = self._sources()
        changed_terms[0]["reuse_terms"] = "changed ledger metadata"
        with self.assertRaisesRegex(ValueError, "source_evidence_sha256"):
            validate_story_graph(as_of, changed_terms)
        tampered = self._sources()
        tampered[0]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "content_sha256"):
            validate_story_graph(graph, tampered)

    def test_learned_graph_builder_must_be_trained_before_cutoff(self):
        graph = self._graph()
        graph["metadata"]["algorithm_fit_mode"] = "learned"
        graph["metadata"]["algorithm_training_cutoff_at"] = "2030-01-20T16:00:00Z"
        graph = seal_story_graph(graph)
        with self.assertRaisesRegex(ValueError, "must precede event cutoff"):
            story_graph_as_of(
                graph, "2030-01-20T16:00:00Z", source_evidence=self._sources(),
            )

    def test_valid_time_removes_expired_profile_claim(self):
        graph = self._graph()
        sliced = story_graph_as_of(
            graph, "2030-01-21T00:00:00Z", source_evidence=self._sources(),
        )
        ids = {node["node_id"] for node in sliced["nodes"]}
        self.assertNotIn("claim-role", ids)
        self.assertNotIn("edge-claim", {edge["edge_id"] for edge in sliced["edges"]})

    def test_retrospective_mode_excludes_same_day_date_only_items(self):
        early = {
            "node_id": "older",
            "node_type": "article_observation",
            "source_ids": ["source-old"],
            "availability_status": "unknown",
            "published_at": "2030-01-19",
            "attributes": {"title": "Older item"},
        }
        same_day = {
            "node_id": "same-day",
            "node_type": "article_observation",
            "source_ids": ["source-same-day"],
            "availability_status": "not_verified",
            "published_at": "2030-01-20",
            "attributes": {"title": "Date is not precise enough"},
        }
        graph = seal_story_graph({
            "schema_version": 1,
            "record_type": "idea-layer-story-graph",
            "metadata": {"algorithm_version": "fixture-v1", "algorithm_fit_mode": "deterministic"},
            "nodes": [early, same_day], "edges": [],
        })
        evidence = [
            self._source("source-old", "2030-01-19T10:00:00Z", "2030-01-25T10:00:00Z", "unknown"),
            self._source("source-same-day", "2030-01-20T12:00:00Z", "2030-01-25T10:00:00Z", "not_verified"),
        ]
        strict = story_graph_as_of(
            graph, "2030-01-20T16:00:00Z", source_evidence=evidence,
        )
        retrospective = story_graph_as_of(
            graph, "2030-01-20T16:00:00Z", source_evidence=evidence,
            availability_mode="retrospective",
        )
        self.assertEqual(strict["nodes"], [])
        self.assertEqual([node["node_id"] for node in retrospective["nodes"]], ["older"])
        self.assertFalse(retrospective["metadata"]["point_in_time_verified"])

    def test_verified_edges_cannot_precede_unavailable_endpoints(self):
        graph = self._graph()
        graph["edges"][0]["available_at"] = "2030-01-19T11:00:00Z"
        graph = seal_story_graph(graph)
        with self.assertRaisesRegex(ValueError, "cannot precede endpoint availability"):
            validate_story_graph(graph)

    def test_rejects_target_outcomes_bad_links_and_tampering(self):
        graph = self._graph()
        graph["nodes"][0]["attributes"]["target_ideas"] = ["leaked label"]
        with self.assertRaisesRegex(ValueError, "outcome-only"):
            validate_story_graph(seal_story_graph(graph))

        graph = self._graph()
        graph["edges"][0]["target_node_id"] = "missing-node"
        with self.assertRaisesRegex(ValueError, "missing endpoint"):
            validate_story_graph(seal_story_graph(graph))

        graph = self._graph()
        graph["metadata"]["algorithm_version"] = "modified"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_story_graph(graph)


if __name__ == "__main__":
    unittest.main()

# Local transcript data

Do not commit collected transcripts by default. The `speechprefix import-govinfo` command can append eligible CPD records to `speeches.jsonl`. You can also add permitted records manually, one JSON object per line, with these required string fields:

```json
{"speech_id":"unique-id","date":"2024-01-01","event":"event name","source":"URL or archive reference","text":"Transcript text..."}
```

Use ISO dates (`YYYY-MM-DD`) so chronological ordering is reliable. Keep source and license/permission notes. Do not include text you are not allowed to store or process. The data is user-supplied; this repository does not download transcripts.

## Time-coded event transcripts

Use the time-coded format for question-response forecasting. Save the source page text, including its `Full Transcript` section, and a metadata JSON file. Import with:

```sh
python3 -m speechprefix turn-import \
  --transcript /path/to/page.txt \
  --metadata /path/to/event.json \
  --output data/events.jsonl
```

Required event metadata fields are `event_id`, `event_date`, `title`, `source_url`, and `target_speaker`. Preserve `source_name` and applicable `reuse_terms` when known. Optional `target_speaker_aliases` map source name variants; `page_metadata` preserves topics, entities, moderation notes, and other page annotations. The importer preserves the raw page header, timestamped source spans, speaker names, transcript text, and per-span StressLens annotations. It does not infer question-response links.

Prediction alignments are supplied in the optional `prediction_annotations` array in the metadata JSON. Each item contains `prediction_time_seconds`, `trigger_turn_ids`, `target_turn_ids`, `response_status`, and `annotation_source`; optional fields are `outcome_turn_ids`, `response_kind`, `idea_labels`, and `example_id`. Trigger spans may come from reporters, the target speaker, or another participant—for example, a reporter's follow-up after the target asks a clarifying question. The selected spans and response links are reviewed, not inferred from adjacency. `response_status` is one of `target_spoke`, `other_speaker_spoke`, `no_target_response`, or `uncertain`. Only reviewed target-speaker spans are exact-word targets. The importer records the visible turn IDs at each prediction point and rejects examples that include target text in the observed context.

```json
{
  "event_id": "press-gaggle-2026-11-01",
  "event_date": "2026-11-01",
  "title": "Press Gaggle After Air Force One Arrival",
  "source_url": "https://example.org/transcript",
  "source_name": "Example transcript archive",
  "reuse_terms": "Review terms before use",
  "target_speaker": "Donald Trump",
  "target_speaker_aliases": ["Donald J. Trump"],
  "page_metadata": {"topics": [], "entities": [], "moderation": []},
  "prediction_annotations": [
    {
      "trigger_turn_ids": ["press-gaggle-2026-11-01:turn:0001"],
      "prediction_time_seconds": 52,
      "target_turn_ids": ["press-gaggle-2026-11-01:turn:0002"],
      "response_status": "target_spoke",
      "response_kind": "direct_answer",
      "idea_labels": ["politics.endorsement"],
      "annotation_source": "reviewed-v1"
    }
  ]
}
```

Idea labels must be human-reviewed semantic tags. They are evaluated separately from the exact transcript words. StressLens, topics, entities, and moderation labels remain source annotations; they are not treated as verified idea labels. The `prediction_example_inputs` helper separates `input`, `target`, and `provenance`; use only the `input` object as model features. Page metadata and per-span annotations are kept in provenance because they may not have been available at the prediction time.

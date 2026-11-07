from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "create_keypoint_reference_packets",
    ROOT / "research/idea_layers/points/create_keypoint_reference_packets.py",
)
assert _SPEC is not None and _SPEC.loader is not None
packets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(packets)


def _source_records():
    rows = []
    for spec in packets.SAMPLE:
        text = " ".join(["speechword"] * spec["expected_words"])
        rows.append({
            "event_id": spec["event_id"],
            "event_date": "2019-01-01",
            "title": spec["expected_title_prefix"] + " - fixture",
            "source_url": "https://example.invalid/transcript",
            "target_speaker": "Donald Trump",
            "turns": [{
                "turn_id": f"{spec['event_id']}:turn:0001",
                "sequence": 1,
                "speaker_name": "Donald Trump",
                "text": text,
            }],
        })
    return rows


def test_packet_creation_writes_blind_transcripts_and_two_blank_forms(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _source_records()), encoding="utf-8")
    output = tmp_path / "packets"
    result = packets.create_packets(source, output)
    assert len(result["speeches"]) == 3
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert [item["target_speaker_word_count"] for item in manifest["speeches"]] == [733, 2013, 7700]
    for speech in manifest["speeches"]:
        packet = (output / "packets" / f"{speech['packet_id']}.txt").read_text(encoding="utf-8")
        assert speech["event_id"] not in packet
        assert speech["title"] not in packet
        assert "[T0001 TARGET]" in packet
        for annotator in ("annotator-A", "annotator-B"):
            form = output / annotator / f"{speech['packet_id']}-points.csv"
            with form.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            assert rows == [packets.CSV_FIELDS]
    assert output.exists()


def test_packet_creation_refuses_overwriting_a_frozen_directory(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _source_records()), encoding="utf-8")
    output = tmp_path / "packets"
    packets.create_packets(source, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        packets.create_packets(source, output)


def test_packet_contains_context_turns_but_counts_evidence_from_target_only():
    row = {
        "target_speaker": "Donald Trump",
        "turns": [
            {"turn_id": "src:1", "sequence": 1, "speaker_name": "Reporter", "text": "What happened?"},
            {"turn_id": "src:2", "sequence": 2, "speaker_name": "Donald Trump", "text": "We will respond."},
        ],
    }
    packet, mapping, target_words, target_count, context_count = packets._build_packet("packet-test", row)
    assert "[T0001 CONTEXT_01]" in packet
    assert "[T0002 TARGET]" in packet
    assert (target_words, target_count, context_count) == (3, 1, 1)
    assert [item["speaker_role"] for item in mapping] == ["context", "target"]


def test_packet_selection_is_frozen_and_excludes_previously_exposed_lexington():
    ids = {row["event_id"] for row in packets.SAMPLE}
    assert ids == {"dt47-25-72a1d6e3", "dt47-25-e856e24c", "dt47-25-8623939c"}
    assert "dt45-19-8a126a9f" not in ids
    assert {row["expected_words"] for row in packets.SAMPLE} == {733, 2013, 7700}

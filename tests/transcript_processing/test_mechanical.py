import json

import pytest

from speechprefix.transcript_processing.mechanical import process, separate


def test_surface_boundaries_and_reconstruction():
    text = " Dr. Smith's 3.14...\ncan't—stop! e\u0301🙂"
    layers = separate(text)
    assert [x["text"] for x in layers["words"]] == [
        "Dr.", "Smith's", "3.14...", "can't—stop!", "e\u0301🙂"
    ]
    assert [x["text"] for x in layers["sentences"]] == [
        " Dr", " Smith's 3", "14", "", "", "\ncan't—stop! e\u0301🙂"
    ]
    assert ".".join(x["text"] for x in layers["sentences"]) == text
    assert "".join(x["text"] for x in layers["characters"]) == text
    for units in layers.values():
        assert all(text[u["start"]:u["end"]] == u["text"] for u in units)
    assert len(layers["characters"]) == len(text)


def test_empty_and_trailing_periods():
    assert separate("")["sentences"] == [{"start": 0, "end": 0, "text": ""}]
    assert [x["text"] for x in separate("a..") ["sentences"]] == ["a", "", ""]
    assert separate("")["words"] == separate("")["characters"] == []


def test_jsonl_outputs_link_to_each_source_record(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [{"id": "same", "text": "One. Two"}, {"id": "same", "text": ""}]
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "output"
    assert process(source, out, "id", "text") == 2
    for layer in ("words", "sentences", "characters"):
        rows = [json.loads(line) for line in (out / f"{layer}.jsonl").read_text().splitlines()]
        assert [r["source_line"] for r in rows] == [1, 2]
        assert all(r["source_id"] == "same" and r["source_file"] == str(source) for r in rows)
        assert rows[0]["text_sha256"] != rows[1]["text_sha256"]
    with pytest.raises(FileExistsError):
        process(source, out, "id", "text")


def test_failure_is_marked_incomplete(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"a","text":"ok"}\n{}\n', encoding="utf-8")
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="line 2"):
        process(source, out, "id", "text")
    assert (out / "INCOMPLETE").exists()

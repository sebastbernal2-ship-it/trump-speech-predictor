"""Lossless, explicitly mechanical surface segmentation of transcript text."""

import argparse
import hashlib
import json
import re
from pathlib import Path

VERSION = "mechanical-v1"


def separate(text: str) -> dict:
    """Return code-point offsets into the unchanged input string."""
    words = [dict(start=m.start(), end=m.end(), text=m.group())
             for m in re.finditer(r"\S+", text)]
    sentences = []
    start = 0
    for index, char in enumerate(text):
        if char == ".":
            sentences.append(dict(start=start, end=index, text=text[start:index]))
            start = index + 1
    sentences.append(dict(start=start, end=len(text), text=text[start:]))
    characters = [dict(start=i, end=i + 1, text=char)
                  for i, char in enumerate(text)]
    return {"words": words, "sentences": sentences, "characters": characters}


def process(input_path: Path, output_dir: Path, id_field: str, text_field: str) -> int:
    """Process a JSONL file without modifying it or overwriting any output."""
    # Exclusive directory creation also prevents accidentally replacing source data.
    output_dir.mkdir(parents=False, exist_ok=False)
    count = 0
    try:
        with input_path.open(encoding="utf-8") as source, \
             (output_dir / "words.jsonl").open("x", encoding="utf-8") as words, \
             (output_dir / "sentences.jsonl").open("x", encoding="utf-8") as sentences, \
             (output_dir / "characters.jsonl").open("x", encoding="utf-8") as characters:
            outputs = {"words": words, "sentences": sentences, "characters": characters}
            for line_number, line in enumerate(source, 1):
                try:
                    record = json.loads(line)
                    identifier, text = record[id_field], record[text_field]
                    if not isinstance(identifier, str) or not identifier or not isinstance(text, str):
                        raise ValueError("ID must be a nonempty string and text must be a string")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid record at line {line_number}: {exc}") from exc
                identity = {"version": VERSION, "source_file": str(input_path),
                            "source_line": line_number, "id_field": id_field,
                            "source_id": identifier, "text_field": text_field,
                            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                for layer, units in separate(text).items():
                    outputs[layer].write(json.dumps({**identity, "units": units}, ensure_ascii=True) + "\n")
                count += 1
    except Exception:
        # Leave partial results visibly incomplete; never silently present them as a release.
        (output_dir / "INCOMPLETE").touch()
        raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSONL input")
    parser.add_argument("output_dir", type=Path, help="new output directory")
    parser.add_argument("--id-field", required=True)
    parser.add_argument("--text-field", required=True)
    args = parser.parse_args()
    count = process(args.input, args.output_dir, args.id_field, args.text_field)
    print(f"Processed {count} records into {args.output_dir}")


if __name__ == "__main__":
    main()

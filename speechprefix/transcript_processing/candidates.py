"""Generate provisional, source-linked transcript topic and idea candidates."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from speechprefix.headline_idea_matching import (
    ISSUE_ANCHORS, ISSUE_PHRASE_EXTRACTOR_VERSION, _tokens,
    extract_transcript_ideas, issue_phrase_lexicon_sha256,
)

VERSION = "transcript-candidates-v1"


def clean(text):
    """Collapse whitespace, recording a raw half-open span for every clean character."""
    chars, spans = [], []
    for match in re.finditer(r'\s+|\S', text):
        value = match.group()
        if value.isspace():
            if chars and match.end() < len(text):
                chars.append(' ')
                spans.append([match.start(), match.end()])
        else:
            chars.append(value)
            spans.append([match.start(), match.end()])
    return ''.join(chars), spans


def extract(text, mapping):
    topics, ideas = {}, {}
    # Each evidence span is an exact contiguous slice of both clean and raw text.
    for sentence in re.finditer(r'[^.!?]+[.!?]?', text):
        start, end = sentence.span()
        while start < end and text[start].isspace():
            start += 1
        if start == end:
            continue
        passage = text[start:end]
        evidence = {"clean_start": start, "clean_end": end, "text": passage,
                    "raw_start": mapping[start][0], "raw_end": mapping[end - 1][1]}
        for term in sorted({token[0] for token in _tokens(passage)} & ISSUE_ANCHORS):
            topics.setdefault(term, []).append(evidence)
        for item in extract_transcript_ideas([{"turn_id": "passage", "text": passage}]):
            entry = ideas.setdefault(item['idea_id'], {"label": item['candidate_idea'], "evidence": []})
            entry['evidence'].append(evidence)
    return ([{"id": key, "label": key, "evidence": value} for key, value in sorted(topics.items())],
            [{"id": key, **value} for key, value in sorted(ideas.items())])


def process(source, output, id_field, text_field):
    output.mkdir(parents=False, exist_ok=False)
    count = 0
    try:
        with source.open(encoding='utf-8') as inp, (output / 'transcripts.jsonl').open('x', encoding='utf-8') as out:
            for line_no, line in enumerate(inp, 1):
                try:
                    row = json.loads(line)
                    identifier, raw = row[id_field], row[text_field]
                    if not isinstance(identifier, str) or not identifier or not isinstance(raw, str):
                        raise ValueError('expected a nonempty string ID and string text')
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'invalid record at line {line_no}: {exc}') from exc
                cleaned, mapping = clean(raw)
                topics, ideas = extract(cleaned, mapping)
                record = {"version": VERSION, "source_file": str(source), "source_line": line_no,
                          "id_field": id_field, "source_id": identifier, "text_field": text_field,
                          "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                          "clean_sha256": hashlib.sha256(cleaned.encode()).hexdigest(),
                          "clean_text": cleaned, "clean_to_raw": mapping,
                          "topics": topics, "ideas": ideas,
                          "status": "candidate_unreviewed" if cleaned else "empty_input"}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        manifest = {"version": VERSION, "input": str(source), "records": count,
                    "extractor": ISSUE_PHRASE_EXTRACTOR_VERSION,
                    "lexicon_sha256": issue_phrase_lexicon_sha256(),
                    "review_status": "unreviewed", "role": "transcript_outcomes_only"}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    except Exception:
        (output / 'INCOMPLETE').touch()
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--id-field', required=True)
    parser.add_argument('--text-field', required=True)
    args = parser.parse_args()
    print(f"Processed {process(args.input, args.output_dir, args.id_field, args.text_field)} records")


if __name__ == '__main__':
    main()

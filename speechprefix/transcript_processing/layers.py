"""Produce all five provisional transcript layers from an explicitly selected field."""

import argparse
import hashlib
import json
from pathlib import Path

from speechprefix.headline_idea_matching import ISSUE_PHRASE_EXTRACTOR_VERSION, issue_phrase_lexicon_sha256
from speechprefix.transcript_processing.candidates import VERSION as CANDIDATE_VERSION, clean, extract
from speechprefix.transcript_processing.mechanical import VERSION as MECHANICAL_VERSION, separate

VERSION = "transcript-layers-v1"


def process(source: Path, output: Path, id_field: str, text_field: str) -> int:
    """Write a new, source-linked candidate artifact; never overwrite existing output."""
    output.mkdir(parents=False, exist_ok=False)
    count = 0
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    try:
        with source.open('rb') as inp, (output / 'layers.jsonl').open('xb') as out:
            for line_no, raw_line in enumerate(inp, 1):
                input_hash.update(raw_line)
                try:
                    row = json.loads(raw_line.decode('utf-8'))
                    identifier, raw = row[id_field], row[text_field]
                    if not isinstance(identifier, str) or not identifier or not isinstance(raw, str):
                        raise ValueError('expected a nonempty string ID and string text')
                except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'invalid record at line {line_no}: {exc}') from exc
                cleaned, mapping = clean(raw)
                topics, ideas = extract(cleaned, mapping)
                record = {
                    'version': VERSION, 'source_file': str(source), 'source_line': line_no,
                    'source_id': identifier, 'id_field': id_field, 'text_field': text_field,
                    'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
                    'clean_sha256': hashlib.sha256(cleaned.encode('utf-8')).hexdigest(),
                    'clean_text': cleaned, 'clean_to_raw': mapping,
                    **separate(cleaned), 'topics': topics, 'ideas': ideas,
                    'semantic_status': 'candidate_unreviewed' if cleaned else 'empty_input',
                    'speaker_attribution': 'unverified',
                }
                encoded = (json.dumps(record, ensure_ascii=True) + '\n').encode('utf-8')
                out.write(encoded)
                output_hash.update(encoded)
                count += 1
        manifest = {
            'version': VERSION, 'source_file': str(source), 'source_sha256': input_hash.hexdigest(),
            'output_file': 'layers.jsonl', 'output_sha256': output_hash.hexdigest(), 'records': count,
            'id_field': id_field, 'text_field': text_field, 'mechanical_version': MECHANICAL_VERSION,
            'candidate_version': CANDIDATE_VERSION, 'extractor': ISSUE_PHRASE_EXTRACTOR_VERSION,
            'lexicon_sha256': issue_phrase_lexicon_sha256(), 'role': 'transcript_outcomes_only',
            'review_status': 'unreviewed',
        }
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
    print(f'Processed {process(args.input, args.output_dir, args.id_field, args.text_field)} records')


if __name__ == '__main__':
    main()

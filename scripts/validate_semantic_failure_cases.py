"""Reproduce a partial, source-bound diagnostic; never infer a negative label."""
import hashlib
import json
import re
import sys
from pathlib import Path

from speechprefix.transcript_processing.mechanical import separate
from speechprefix.transcript_processing.semantic_v2 import build, digest

# Exact phrases fix the reviewed boundaries to this particular source version.
CASES = {
    222: [('target', 'We grieve', 'Categories:'), ('idea', 'Sponsoring terrorism risks victimization', 'We underscore', 'they promote.')],
    291: [('target', 'Senator John McCain has always', 'Categories:'), ('idea', 'Well wishes for Senator McCain', 'Melania and I send', 'Get well soon.')],
    501: [('target', 'I have to say, the United Nations', 'Secretary-General Guterres. Thank you very much. And I need'), ('context', 'I am a true believer', 'The President. Thank you very much.'), ('idea', 'United Nations has unrealized potential', 'I have to say, the United Nations', 'as it should be,')],
    968: [('target', 'Great job numbers', 'Categories:'), ('idea', 'Job numbers are better than expected and policy is working', 'Great job numbers', "It's already working.")],
    969: [('target', 'To the many investors', 'Categories:'), ('idea', 'Promises investors that policies will not change', 'To the many investors', 'my policies will never change.')],
    1019: [('target', 'Such an honor for our country.', 'United Kingdom–U.S. Trade/Professional'), ('context', 'Is there still a chance you might meet Vladimir Putin', 'The President. I don\'t think so'), ('idea', 'Calls an American Pope an honor', 'Such an honor for our country.', 'to have an American Pope.')],
}


def run(path):
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    results = []
    with path.open('rb') as f:
        for line_no, raw in enumerate(f, 1):
            if line_no not in CASES:
                continue
            row = json.loads(raw)
            text = row['text']
            record = {'source_line': line_no, 'source_row_sha256': hashlib.sha256(raw).hexdigest(), 'turns': [{'turn_id': 'flattened', 'text': text}]}
            entries, ideas = [], []
            for spec in CASES[line_no]:
                if spec[0] in ('target', 'context'):
                    role, first, last = spec
                    a = text.index(first); b = text.index(last, a)
                    entries.append((a, b, role))
                else:
                    _, label, first, last = spec
                    a = text.index(first); b = text.index(last, a) + len(last)
                    ideas.append({'id': 'diagnostic', 'label': label, 'evidence': [{'turn_id': 'flattened', 'start': a, 'end': b}]})
            entries.sort()
            segments, cursor = [], 0
            def add(a, b, role):
                if a < b:
                    segments.append({'turn_id': 'flattened', 'start': a, 'end': b, 'role': role, 'reason': 'partial source inspection' if role != 'uncertain' else 'outside reviewed diagnostic spans'})
            for a, b, role in entries:
                assert cursor <= a < b
                add(cursor, a, 'uncertain'); add(a, b, role); cursor = b
            add(cursor, len(text), 'uncertain')
            canonical = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
            review = {'source_record_sha256': digest(canonical), 'reviewer': 'Codex diagnostic, 2026-11-17', 'status': 'uncertain', 'reason': 'Partial diagnostic; attribution, source completeness and inventory not independently verified.', 'segments': segments, 'topics': [], 'ideas': ideas}
            result = build(record, review)
            checks = []
            for s in result['segments']:
                if s['role'] != 'target':
                    continue
                clean = s['clean_text']; units = separate(clean)
                assert ''.join(x['text'] for x in units['characters']) == clean
                assert [x['text'] for x in units['words']] == re.findall(r'\S+', clean)
                assert [x['text'] for x in units['sentences']] == clean.split('.')
                assert all(clean[x['start']:x['end']] == x['text'] for layer in units.values() for x in layer)
                assert all(text[a:b].isspace() if ch == ' ' and text[a:b].isspace() else text[a:b] == ch for ch, (a, b) in zip(clean, s['clean_to_raw']))
                checks.append({'target_span': [s['start'], s['end']], 'characters': True, 'exact_words': True, 'literal_period_split': True, 'offsets_and_clean_mapping': True})
            for item in result['ideas']:
                for ev in item['evidence']:
                    assert text[ev['start']:ev['end']] == ev['quote']
            # Keep source data in its original location; retain exact offsets and hashes.
            results.append({'line': line_no, 'source_id': row.get('speech_id'), 'line_sha256': record['source_row_sha256'], 'source_record_sha256': review['source_record_sha256'], 'review': review, 'validation': checks, 'training_label_status': 'not_released'})
    assert [x['line'] for x in results] == list(CASES)
    return {'version': 'failure-case-review-v1', 'source_file': str(path), 'source_file_sha256': file_hash, 'semantic_version': 'semantic-targets-v2', 'mechanical_version': 'mechanical-v1', 'cases': results}


if __name__ == '__main__':
    output = Path(sys.argv[2])
    with output.open('x', encoding='utf-8') as f:
        json.dump(run(Path(sys.argv[1])), f, indent=2, ensure_ascii=True)
        f.write('\n')

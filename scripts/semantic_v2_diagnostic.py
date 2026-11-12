"""Run source-bound development diagnostics without writing or copying source data."""
import hashlib
import json
import sys
from speechprefix.transcript_processing.semantic_v2 import build, digest

CASES = {
    222: ('We grieve', 'Categories:', 'Iran attacks', 'Says sponsors of terrorism risk becoming victims', 'We underscore'),
    291: ('Senator John McCain has always', 'Categories:', 'Health', 'Sends well wishes to Senator McCain', 'Melania and I send'),
    968: ('Great job numbers', 'Categories:', 'Employment', 'Says job numbers are strong and the policy is working', 'Great job numbers'),
    969: ('To the many investors', 'Categories:', 'Investment policy', 'Says his policies will not change', 'To the many investors'),
}


def run(path):
    with open(path, encoding='utf-8') as source:
        for line_no, line in enumerate(source, 1):
            if line_no not in CASES:
                continue
            row = json.loads(line)
            text = row['text']
            start_text, end_text, topic, idea, quote_start = CASES[line_no]
            start, end = text.index(start_text), text.index(end_text)
            qstart = text.index(quote_start, start, end)
            qend = text.index('.', qstart, end) + 1
            if line_no == 968:
                qend = text.index('.', qend, end) + 1
            record = {'source_line': line_no, 'source_row_sha256': hashlib.sha256(line.encode()).hexdigest(),
                      'turns': [{'turn_id': 'flattened', 'text': text}]}
            raw = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
            def seg(a, b, role):
                return {'turn_id': 'flattened', 'start': a, 'end': b, 'role': role, 'reason': 'development inspection'}
            citation = {'turn_id': 'flattened', 'start': qstart, 'end': qend}
            review = {'source_record_sha256': digest(raw), 'reviewer': 'development inspection',
                      'status': 'uncertain', 'reason': 'flattened attribution and full inventory unverified',
                      'segments': [seg(0, start, 'metadata'), seg(start, end, 'target'), seg(end, len(text), 'metadata')],
                      'topics': [{'id': 'topic', 'label': topic, 'evidence': [{'turn_id': 'flattened', 'start': start, 'end': end}]}],
                      'ideas': [{'id': 'idea', 'label': idea, 'evidence': [citation]}]}
            result = build(record, review)
            print(json.dumps({'line': line_no, 'source_row_sha256': record['source_row_sha256'],
                              'status': result['status'], 'topic': topic, 'idea': idea,
                              'evidence_offsets': [qstart, qend]}))


if __name__ == '__main__':
    run(sys.argv[1])

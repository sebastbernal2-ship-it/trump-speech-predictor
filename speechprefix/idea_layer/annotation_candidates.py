"""Offset-preserving offline evidence and detail proposals (not final labels)."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Candidate:
    turn_id: str
    start: int
    end: int
    kind: str
    parent_start: int | None = None
    parent_end: int | None = None


CLAUSES = {'advcl', 'ccomp', 'xcomp', 'relcl', 'parataxis', 'conj'}
ARGUMENTS = {'dobj', 'obj', 'iobj', 'attr', 'oprd', 'pobj', 'obl', 'dative'}


def propose(packet, parser=None):
    """Return sentence baseline, syntax additions, and parent-scoped details."""
    result = set()
    for turn in packet['turns']:
        if turn['role'] != 'target':
            continue
        text = turn['text']
        ident = turn['turn_id']

        def add(start, end, kind, parent=None):
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                result.add(Candidate(ident, start, end, kind, *(parent or (None, None))))

        for match in re.finditer(r'[^.!?\n]+[.!?]*', text):
            add(*match.span(), 'baseline_sentence')
        if parser is None or not text.strip():
            continue
        doc = parser(text)
        for sentence in doc.sents:
            parent = sentence.start_char, sentence.end_char
            add(*parent, 'parsed_sentence')
            for token in sentence:
                clause = token.dep_ in CLAUSES and token.pos_ in {'VERB', 'AUX'}
                argument = token.dep_ in ARGUMENTS
                if not (clause or argument):
                    continue
                subtree = sorted(t.i for t in token.subtree if sentence.start <= t.i < sentence.end)
                if not subtree or len(subtree) != subtree[-1] - subtree[0] + 1:
                    continue
                span = doc[subtree[0]:subtree[-1] + 1]
                add(span.start_char, span.end_char, 'clause' if clause else 'argument', parent)
        for entity in doc.ents:
            if entity.label_ in {'EVENT', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL', 'LAW', 'GPE', 'ORG', 'PERSON'}:
                add(entity.start_char, entity.end_char, 'entity', (entity.sent.start_char, entity.sent.end_char))
    return sorted(result, key=lambda c: (c.turn_id, c.start, c.end, c.kind, c.parent_start or 0, c.parent_end or 0))

"""Conservative contextual subject extraction for the offline experiment."""
import re

from .parsed_annotation import label_span


_SHORT_RESPONSE = re.compile(r"(?:yes|no|absolutely|not yet|i (?:do|don't|will|won't))\s*[.!]?", re.I)


def select_subject(span, previous, parser):
    """Resolve short answers to a single explicit subject in the prior question."""
    if previous and previous['role'] == 'context' and _SHORT_RESPONSE.fullmatch(span):
        question = previous['text'].strip()
        if question.endswith('?') and question.count('?') == 1:
            doc = parser(question)
            candidates = [c for c in doc.noun_chunks if c.root.pos_ in {'NOUN', 'PROPN'}
                          and c.root.dep_ in {'dobj', 'obj', 'pobj', 'nsubjpass'}]
            if len(candidates) == 1 and not any(t.dep_ in {'conj', 'neg'} for t in doc):
                return candidates[0].text
    return label_span(span, parser, frames=True, grounded=True)

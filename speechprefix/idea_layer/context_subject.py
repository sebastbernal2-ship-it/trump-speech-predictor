"""Conservative contextual subject extraction for the offline experiment."""
import re

from .parsed_annotation import label_span


_SHORT_RESPONSE = re.compile(r"(?:yes|no|absolutely|not yet|i (?:do|don't|will|won't))\s*[.!]?", re.I)


def resolve_question_subject(span, previous, parser):
    """Return a subject only when a short answer resolves to one question argument."""
    if previous and previous['role'] == 'context' and _SHORT_RESPONSE.fullmatch(span):
        question = previous['text'].strip()
        if question.endswith('?') and question.count('?') == 1:
            doc = parser(question)
            candidates = [c for c in doc.noun_chunks if c.root.pos_ in {'NOUN', 'PROPN'}
                          and c.root.dep_ in {'dobj', 'obj', 'pobj', 'nsubjpass'}]
            first = next((t for t in doc if not (t.is_space or t.is_punct)), None)
            polar = first is not None and first.pos_ == 'AUX'
            if polar and len(candidates) == 1 and not any(t.dep_ in {'conj', 'neg'} for t in doc):
                return candidates[0].text
    return None


def select_subject(span, previous, parser):
    """Prefer a resolved question subject, then inspect the target span."""
    if subject := resolve_question_subject(span, previous, parser):
        return subject
    doc = parser(span)
    sentences = list(doc.sents)
    if len(sentences) == 1:
        sentence = sentences[0]
        root = sentence.root
        # In a nominal fragment, preserve complements rather than promoting a
        # nested noun to the subject of the entire fragment.
        if root.pos_ in {'NOUN', 'PROPN'} and any(t.dep_ == 'prep' for t in root.children):
            if len(sentence) <= 12 and not any(t.pos_ in {'VERB', 'AUX'} for t in sentence):
                return sentence.text.rstrip('.!?').strip()
        if root.lemma_ in {'believe', 'think', 'say', 'claim', 'expect'}:
            for child in root.children:
                if child.dep_ != 'ccomp' or not any(
                    t.dep_ in {'nsubj', 'nsubjpass'} and t.pos_ in {'NOUN', 'PROPN'} for t in child.children
                ):
                    continue
                members = sorted(t.i for t in child.subtree if not t.is_punct)
                if members and members[-1] - members[0] < 16:
                    return doc[members[0]:members[-1] + 1].text
    return label_span(span, lambda _: doc, frames=True, grounded=True)

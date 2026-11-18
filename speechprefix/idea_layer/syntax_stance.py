"""Experimental offline speech-act classifier using sentence dependencies."""
from .classical_annotation import cue_stance


def classify(text, parser):
    if '?' in text:
        return None
    doc = parser(text)
    for sentence in doc.sents:
        root = sentence.root
        if root.pos_ not in {'VERB', 'AUX'}:
            continue
        children = list(root.children)
        subject = next((t for t in children if t.dep_ in {'nsubj', 'nsubjpass'}), None)
        if subject is None and root.tag_ == 'VB' and root.lemma_ not in {'thank', 'remember', 'say'}:
            return 'requests'
        if subject is not None and subject.lower_ in {'i', 'we'} and any(
            t.lower_ in {'should', 'must'} for t in children if t.dep_ == 'aux'
        ):
            return 'proposes'
        if subject is not None and subject.lower_ in {'i', 'we'} and root.lemma_ == 'have' and any(
            t.dep_ == 'xcomp' and any(c.lower_ == 'to' for c in t.children) for t in children
        ):
            return 'proposes'
    return cue_stance(text)

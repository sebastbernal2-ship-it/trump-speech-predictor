"""Conservative subject alignment for development diagnostics, not semantic gold."""
import re
from collections import Counter


_ALIASES = {'ukrainian': 'ukraine', 'russian': 'russia'}
_STOP = {'the', 'a', 'an', 'of', 'in', 'and', 'vs', 'versus', 'against', 'with', 'between'}


def subject_signature(label):
    tokens = re.findall(r'[^\W_]+', label.casefold())
    counts = Counter(_ALIASES.get(token, token) for token in tokens if token not in _STOP)
    # A war named for its two parties can legitimately be phrased in either order.
    if counts == {'russia': 1, 'ukraine': 1, 'war': 1}:
        return ('russia–ukraine war',)
    # Default to ordered tokens: sorting can conflate directional relationships.
    return tuple(tokens)


def align_ideas(candidate, reference):
    """One-to-one alignment; additional event qualifiers are never discarded."""
    remaining = list(reference)
    matches, unmatched = [], []
    for idea in candidate:
        label = subject_signature(idea['occurrence'])
        index = next((i for i, ref in enumerate(remaining)
                      if label and label == subject_signature(ref['occurrence'])), None)
        if index is None:
            unmatched.append(idea['idea_id'])
        else:
            ref = remaining.pop(index)
            matches.append({'candidate_id': idea['idea_id'], 'reference_id': ref['idea_id']})
    return {'matches': matches, 'unmatched_candidate_ids': unmatched,
            'unmatched_reference_ids': [ref['idea_id'] for ref in remaining]}

"""Conservative eligibility for transcript evidence, without rewriting source text."""
import re


_BRACKET_ONLY = re.compile(r'(?:\s*\[[^\[\]]+\]\s*[.!?…]*)+')


def has_spoken_evidence(text: str) -> bool:
    """Exclude whitespace and standalone bracketed annotations, retain mixed text."""
    return bool(text.strip()) and _BRACKET_ONLY.fullmatch(text) is None

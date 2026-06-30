"""
Grounding: locate supporting spans AFTER generation (provenance, not generation
input). Given a golden answer (+ optional verbatim quote) and candidate source
documents, find the best-matching section spans in the corpus.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .corpus import Corpus, SectionRef
from .schema import SupportingSpan

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "de", "het", "een", "en", "van", "in", "op", "te", "dat", "die", "is",
    "voor", "met", "aan", "bij", "als", "of", "om", "naar", "uit", "ook",
    "wat", "welke", "welk", "hoe", "wie", "waar", "is", "zijn", "heeft",
}


def _keywords(text: str, limit: int = 12) -> List[str]:
    seen: List[str] = []
    for w in _WORD.findall(text.lower()):
        if len(w) < 4 or w in _STOP:
            continue
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _overlap_score(needle_kws: List[str], hay: str) -> float:
    if not needle_kws:
        return 0.0
    hay_l = hay.lower()
    hits = sum(1 for k in needle_kws if k in hay_l)
    return hits / len(needle_kws)


def _trim(text: str, max_chars: int = 600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def locate_spans(
    corpus: Corpus,
    *,
    golden_answer: str,
    source_documents: List[str],
    supporting_quote: str = "",
    key_term: str = "",
    max_spans: int = 4,
    min_score: float = 0.2,
) -> List[SupportingSpan]:
    """Find supporting spans for an answer.

    Strategy: prefer an exact match of `supporting_quote`; otherwise rank
    sections (restricted to the source docs when given, else whole corpus) by
    keyword overlap with the answer (and `key_term`).
    """
    kws = _keywords(golden_answer)
    if key_term:
        kws = _keywords(key_term) + kws

    # Candidate sections
    if source_documents:
        candidates: List[SectionRef] = []
        for fname in source_documents:
            doc = corpus.by_name.get(fname)
            if doc:
                candidates.extend(doc.sections)
        if not candidates:
            candidates = corpus.all_section_refs()
    else:
        candidates = corpus.all_section_refs()

    # 1) exact verbatim quote match
    spans: List[SupportingSpan] = []
    if supporting_quote and len(supporting_quote.strip()) >= 12:
        q = re.sub(r"\s+", " ", supporting_quote.strip().lower())
        for s in candidates:
            if q in re.sub(r"\s+", " ", s.text.lower()):
                spans.append(SupportingSpan(s.document, s.page_start, _trim(s.text)))
                break

    # 2) keyword-overlap ranking
    scored = sorted(
        ((s, _overlap_score(kws, s.text)) for s in candidates),
        key=lambda x: x[1], reverse=True,
    )
    for s, score in scored:
        if len(spans) >= max_spans:
            break
        if score < min_score:
            break
        if any(sp.document == s.document and sp.text.startswith(_trim(s.text)[:40])
               for sp in spans):
            continue
        spans.append(SupportingSpan(s.document, s.page_start, _trim(s.text)))

    return spans[:max_spans]


def is_grounded(spans: List[SupportingSpan]) -> bool:
    return len(spans) > 0

"""
Grounding: locate supporting spans AFTER generation (provenance, not generation
input). Given a golden answer (+ optional verbatim quote) and candidate source
documents, find the best-matching section spans in the corpus.

Scoring favours *distinctive* evidence — the answer's anchor tokens (emails,
phone numbers, money amounts, standalone numbers, quoted phrases, proper nouns) —
over generic keyword overlap, so the section that actually contains the answer
ranks first. A relative threshold drops weakly-related sections (false
positives), and snippets are windowed around the match rather than the section
start.
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

from .corpus import Corpus, SectionRef
from .schema import SupportingSpan

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "de", "het", "een", "en", "van", "in", "op", "te", "dat", "die", "is",
    "voor", "met", "aan", "bij", "als", "of", "om", "naar", "uit", "ook",
    "wat", "welke", "welk", "hoe", "wie", "waar", "zijn", "heeft", "deze",
    "wordt", "worden", "kan", "kun", "kunt", "per", "over", "door", "je",
    "ik", "we", "wij", "niet", "wel", "meer", "onder", "tot", "maar",
}

# Anchor patterns — high-precision evidence signals.
_RE_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
_RE_PHONE = re.compile(r"\b0\d[\s\-]?\d{6,8}\b|\b06[\s\-]?\d{6,8}\b")
_RE_MONEY = re.compile(r"(?:eur|€)\s?\d[\d.\s]*\d|\b\d{4,}\b")
_RE_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{4,60})[\"'“”‘’]")
_RE_PROPER = re.compile(r"\b[A-Z][a-zA-Z]{3,}(?:\s[A-Z][a-zA-Z]+)?\b")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _keywords(text: str, limit: int = 20) -> List[str]:
    seen: List[str] = []
    for w in _WORD.findall(text.lower()):
        if len(w) < 4 or w in _STOP:
            continue
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _anchors(text: str) -> Set[str]:
    """Distinctive tokens from the answer that pin it to a specific section."""
    anchors: Set[str] = set()
    for m in _RE_EMAIL.findall(text):
        anchors.add(m.lower())
    for m in _RE_PHONE.findall(text):
        anchors.add(re.sub(r"[\s\-]", "", m))
    for m in _RE_QUOTED.findall(text):
        anchors.add(m.lower().strip())
    # money / long numbers, normalized to digits
    for m in _RE_MONEY.findall(text):
        digits = re.sub(r"\D", "", m)
        if len(digits) >= 3:
            anchors.add(digits)
    # multi-word proper nouns (names/orgs)
    for m in _RE_PROPER.findall(text):
        if " " in m:
            anchors.add(m.lower())
    return {a for a in anchors if a}


def _section_has_anchor(section_text: str, anchor: str) -> bool:
    low = section_text.lower()
    if anchor.isdigit():
        return anchor in re.sub(r"\D", "", low)
    return anchor in low


def _score(section_text: str, kws: List[str], anchors: Set[str]) -> Tuple[float, int]:
    """Return (score, anchor_hits). Anchors weigh far more than keywords."""
    low = section_text.lower()
    anchor_hits = sum(1 for a in anchors if _section_has_anchor(section_text, a))
    kw_hits = sum(1 for k in kws if k in low)
    return anchor_hits * 6.0 + kw_hits, anchor_hits


def _window(text: str, kws: List[str], anchors: Set[str],
            width: int = 360) -> str:
    """Snippet centered on the earliest anchor/keyword match, not section start."""
    norm = _norm(text)
    low = norm.lower()
    pos = -1
    for term in list(anchors) + kws:
        idx = low.find(term)
        if idx != -1:
            pos = idx
            break
    if pos <= 0:
        return norm[:width] + ("…" if len(norm) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(norm), start + width)
    snippet = norm[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(norm):
        snippet = snippet + "…"
    return snippet


def locate_spans(
    corpus: Corpus,
    *,
    golden_answer: str,
    source_documents: List[str],
    supporting_quote: str = "",
    key_term: str = "",
    max_spans: int = 4,
) -> List[SupportingSpan]:
    """Find supporting spans for an answer.

    1) exact supporting_quote match wins (windowed);
    2) otherwise rank sections by anchor+keyword score, keep those within a
       relative band of the top score (drops false positives), and always keep
       at least the single best section when anything plausible matches.
    """
    kws = _keywords(golden_answer)
    if key_term:
        kws = _keywords(key_term) + [k for k in kws if k not in _keywords(key_term)]
    anchors = _anchors(golden_answer + " " + supporting_quote)

    # Candidate sections
    candidates: List[SectionRef] = []
    if source_documents:
        for fname in source_documents:
            doc = corpus.by_name.get(fname)
            if doc:
                candidates.extend(doc.sections)
    if not candidates:
        candidates = corpus.all_section_refs()

    spans: List[SupportingSpan] = []
    used: Set[Tuple[str, int]] = set()

    # 1) exact verbatim quote
    if supporting_quote and len(supporting_quote.strip()) >= 12:
        q = _norm(supporting_quote).lower()
        for s in candidates:
            if q in _norm(s.text).lower():
                spans.append(SupportingSpan(
                    s.document, s.page_start,
                _window(s.text, kws, anchors), full_text=s.text))
                used.add((s.document, s.page_start))
                break

    # 2) score & rank
    scored: List[Tuple[SectionRef, float, int]] = []
    for s in candidates:
        score, ah = _score(s.text, kws, anchors)
        if score > 0:
            scored.append((s, score, ah))
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        top = scored[0][1]
        # Keep sections that are anchor-backed OR within 55% of the top score,
        # and above a small absolute floor. This drops tangential sections.
        floor = max(2.0, top * 0.55)
        for s, score, ah in scored:
            if len(spans) >= max_spans:
                break
            key = (s.document, s.page_start)
            if key in used:
                continue
            keep = (ah > 0) or (score >= floor)
            if not keep:
                continue
            spans.append(SupportingSpan(
                s.document, s.page_start,
                _window(s.text, kws, anchors), full_text=s.text))
            used.add(key)

        # Never return zero spans when a plausible best section exists.
        if not spans:
            s = scored[0][0]
            spans.append(SupportingSpan(
                s.document, s.page_start,
                _window(s.text, kws, anchors), full_text=s.text))

    return spans[:max_spans]


def is_grounded(spans: List[SupportingSpan]) -> bool:
    return len(spans) > 0


def snippet_around(text: str, term: str, width: int = 300) -> str:
    """Public helper: window a section around `term` (used by deterministic
    verification so its spans show the match, not the section start)."""
    return _window(text, _keywords(term), {term.lower()} if term else set(), width)

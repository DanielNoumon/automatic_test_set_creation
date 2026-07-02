"""
Stage 3 — verification.

Two complementary checks:
  • deterministic: where a question reduces to a clean structured query
    (capability/counting via corpus search, Excel column lookups), compute the
    answer with code — never trust the LLM's count.
  • solver agent (GPT 5.4): an INDEPENDENT answerer is given the relevant corpus
    evidence and must reproduce the gold answer; mismatches are flagged.

Either path can confirm a question; deterministic is preferred when applicable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .corpus import Corpus, CAT_CV, CAT_PROJECTS
from .grounding import SupportingSpan, snippet_around
from .llm import LLM
from .prompts import build_solver_prompt
from .schema import Verification


@dataclass
class VerifyResult:
    ok: bool
    golden_answer: str
    verification: Verification
    spans: List[SupportingSpan]


def to_text(value) -> str:
    """Coerce an LLM-provided answer (str | list | dict | None) to a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(to_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return ", ".join(f"{k}: {to_text(v)}" for k, v in value.items()).strip()
    return str(value).strip()


# ── deterministic helpers ───────────────────────────────

def deterministic_capability(
    corpus: Corpus, *, key_term: str,
) -> Optional[VerifyResult]:
    """For 'who/which docs mention <key_term>' questions: the verified answer is
    exactly the set of documents that actually contain the term."""
    if not key_term:
        return None
    hits = corpus.documents_mentioning(key_term, categories=[CAT_CV, CAT_PROJECTS])
    spans: List[SupportingSpan] = []
    for fname, sec in corpus.search_term(key_term, categories=[CAT_CV, CAT_PROJECTS]):
        if not any(s.document == fname for s in spans):
            snippet = snippet_around(sec.text, key_term, width=300)
            spans.append(SupportingSpan(fname, sec.page_start, snippet))
    if not hits:
        return VerifyResult(
            ok=True,
            golden_answer=(
                f"Er is geen aantoonbare ervaring met '{key_term}' in de documenten."
            ),
            verification=Verification(
                method="deterministic", solver_reproduced=None,
                notes="key_term not found anywhere in corpus",
                deterministic_matches=[],
            ),
            spans=[],
        )
    names = ", ".join(_person_name(h) for h in hits)
    return VerifyResult(
        ok=True,
        golden_answer=(
            f"Ja. '{key_term}' komt voor bij {len(hits)} bron(nen): {names}."
        ),
        verification=Verification(
            method="deterministic", solver_reproduced=None,
            notes=f"{len(hits)} docs match key_term",
            deterministic_matches=list(hits),
        ),
        spans=spans[:25],
    )


def _person_name(rel_path: str) -> str:
    """Best-effort readable name from a CV filename."""
    stem = rel_path.rsplit("/", 1)[-1]
    stem = re.sub(r"\.(docx|pdf)$", "", stem, flags=re.I)
    stem = re.sub(r"(?i)\b(cv|dsl|update|engels|nl|\d{4,})\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip(" -_") or rel_path


# ── solver gate ─────────────────────────────────────────

def solver_check(
    solver: LLM, corpus: Corpus, *, question: str, proposed_answer: str,
    spans: List[SupportingSpan], extra_evidence: str = "",
) -> VerifyResult:
    """Independent GPT 5.4 solver reproduces the answer from evidence."""
    evidence_parts = [f"[{s.document}] {s.text}" for s in spans]
    if extra_evidence:
        evidence_parts.append(extra_evidence)
    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(geen bewijs)"

    proposed_answer = to_text(proposed_answer)
    evidence_docs = len({s.document for s in spans})
    res = solver.json(build_solver_prompt(question=question, evidence=evidence))
    if not res:
        return VerifyResult(
            ok=False, golden_answer=proposed_answer,
            verification=Verification(
                method="solver_agent", solver_reproduced=False,
                notes="solver returned no parseable answer",
                proposed_answer=proposed_answer, solver_answer="",
                evidence_docs=evidence_docs,
            ),
            spans=spans,
        )
    answerable = bool(res.get("answerable", True))
    solver_ans = to_text(res.get("answer"))
    reproduced = answerable and _answers_agree(proposed_answer, solver_ans)
    # Trust the solver's grounded answer as the gold when it is answerable.
    gold = solver_ans if answerable and solver_ans else proposed_answer
    return VerifyResult(
        ok=answerable,
        golden_answer=gold,
        verification=Verification(
            method="solver_agent",
            solver_reproduced=reproduced,
            notes="" if reproduced else "solver answer differs from proposed",
            proposed_answer=proposed_answer, solver_answer=solver_ans,
            evidence_docs=evidence_docs,
        ),
        spans=spans,
    )


def _answers_agree(a, b, threshold: float = 0.4) -> bool:
    """Cheap token-overlap agreement check between two answers."""
    ta = set(re.findall(r"\w+", to_text(a).lower()))
    tb = set(re.findall(r"\w+", to_text(b).lower()))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb))) >= threshold

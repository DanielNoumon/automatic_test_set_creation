"""
Generation stages.

  Stage 1 — per-document (whole-doc) generation for POLICY/PROCESS intents (T1/T2).
  Stage 2 — corpus-level generation for CAPABILITY/REFERENCE intents (T3/T4),
            steered by a compact corpus index.
  Stage 5 — behavioral types (hallucination / ambiguous / multi-turn generated
            directly; injection / aggro grafted onto verified content questions).

These functions return *raw* dicts; grounding, verification and the
self-containedness gate are applied by the orchestrator.
"""
from __future__ import annotations

import re
from typing import Dict, List

from pipeline.config import QuestionType
from .corpus import Corpus, CorpusDoc, CAT_POLICY, CAT_PROCESS, CAT_CV, CAT_PROJECTS
from .llm import LLM
from .prompts import (
    build_doc_generation_prompt, build_corpus_generation_prompt,
    build_hallucination_prompt, build_ambiguous_prompt, build_multiturn_prompt,
    build_behavioral_graft_prompt,
)
from .schema import IntentCluster, Persona

# Whole-document char cap (keeps within context while still "whole doc").
_DOC_CHAR_CAP = 80_000
_INDEX_DOC_CHARS = 220


def _cap(text: str, n: int = _DOC_CHAR_CAP) -> str:
    return text if len(text) <= n else text[:n]


def _spread(n: int, k: int) -> List[int]:
    """Split n items across k buckets as evenly as possible."""
    if k <= 0:
        return []
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


# ── Stage 1 — per-document ──────────────────────────────

def stage1_doc_questions(
    gen: LLM, corpus: Corpus, *, intent: IntentCluster,
    personas: List[Persona], n: int,
) -> List[Dict]:
    cat = CAT_POLICY if intent == IntentCluster.POLICY else CAT_PROCESS
    docs = corpus.by_category(cat)
    if not docs:
        return []
    # Prefer larger docs first (richer for T2 synthesis)
    docs = sorted(docs, key=lambda d: len(d.full_text), reverse=True)
    per_doc = _spread(n, len(docs))
    out: List[Dict] = []
    pi = 0
    for doc, k in zip(docs, per_doc):
        if k == 0:
            continue
        persona = personas[pi % len(personas)]
        pi += 1
        res = gen.json(build_doc_generation_prompt(
            doc_text=_cap(doc.full_text), filename=doc.filename,
            persona=persona, intent=intent, n=k,
        ))
        for q in (res or {}).get("questions", []) if res else []:
            q = dict(q)
            q["source_documents"] = [doc.filename]
            q["persona"] = persona.value
            q["intent_cluster"] = intent.value
            q["_stage"] = 1
            out.append(q)
    return out


# ── Stage 2 — corpus-level ──────────────────────────────

def build_corpus_index(corpus: Corpus) -> str:
    """Compact, single-prompt map of the whole corpus: per-doc snippet + the
    full project table (the key structured source for aggregation)."""
    lines: List[str] = ["# Documenten"]
    for d in corpus.docs:
        if d.category == CAT_PROJECTS:
            continue
        snippet = re.sub(r"\s+", " ", d.full_text).strip()[:_INDEX_DOC_CHARS]
        lines.append(f"- [{d.category}] {d.filename}: {snippet}")
    ed = corpus.excel_doc()
    if ed is not None:
        lines.append("\n# Projectentabel (volledig)")
        lines.append(re.sub(r"\n{2,}", "\n", ed.full_text)[:20_000])
    return "\n".join(lines)


def stage2_corpus_questions(
    gen: LLM, corpus: Corpus, corpus_index: str, *, intent: IntentCluster,
    personas: List[Persona], n: int,
) -> List[Dict]:
    out: List[Dict] = []
    buckets = _spread(n, len(personas)) if personas else []
    for persona, k in zip(personas, buckets):
        if k == 0:
            continue
        res = gen.json(build_corpus_generation_prompt(
            corpus_index=corpus_index, persona=persona, intent=intent, n=k,
        ))
        for q in (res or {}).get("questions", []) if res else []:
            q = dict(q)
            q["persona"] = persona.value
            q["intent_cluster"] = intent.value
            q["_stage"] = 2
            out.append(q)
    return out


# ── Stage 5 — behavioral ────────────────────────────────

def stage5_hallucination(
    gen: LLM, corpus: Corpus, corpus_index: str, *, n: int,
) -> List[Dict]:
    out: List[Dict] = []
    personas = [Persona.SALES, Persona.TECHNICAL, Persona.STRATEGY]
    for i in range(n):
        res = gen.json(build_hallucination_prompt(
            corpus_index=corpus_index, persona=personas[i % len(personas)],
        ))
        if res and res.get("question"):
            res = dict(res)
            res["type"] = QuestionType.HALLUCINATION_TEST.value
            res["_stage"] = 5
            out.append(res)
    return out


def stage5_ambiguous(gen: LLM, corpus: Corpus, *, n: int) -> List[Dict]:
    docs = corpus.cvs(subset_only=True) + corpus.by_category(CAT_POLICY)
    out: List[Dict] = []
    for i in range(n):
        doc = docs[i % len(docs)]
        res = gen.json(build_ambiguous_prompt(
            doc_text=_cap(doc.full_text), filename=doc.filename,
        ))
        if res and res.get("question"):
            res = dict(res)
            res["source_documents"] = [doc.filename]
            res["type"] = QuestionType.AMBIGUOUS_QUESTIONS.value
            res["_stage"] = 5
            out.append(res)
    return out


def stage5_multiturn(gen: LLM, corpus: Corpus, *, n: int) -> List[Dict]:
    docs = corpus.by_category(CAT_POLICY) + corpus.cvs(subset_only=True)
    out: List[Dict] = []
    for i in range(n):
        doc = docs[i % len(docs)]
        res = gen.json(build_multiturn_prompt(
            doc_text=_cap(doc.full_text), filename=doc.filename,
        ))
        if res and res.get("question_turn_1"):
            res = dict(res)
            res["source_documents"] = [doc.filename]
            res["type"] = QuestionType.MULTI_TURN_FOLLOWUP.value
            res["_stage"] = 5
            out.append(res)
    return out


def stage5_graft(
    gen: LLM, *, behavior: QuestionType, base_questions: List[Dict], n: int,
) -> List[Dict]:
    """Wrap verified content questions with injection/aggro envelopes."""
    out: List[Dict] = []
    for base in base_questions[:n]:
        res = gen.json(build_behavioral_graft_prompt(
            base_question=base["question"], base_answer=base["golden_answer"],
            behavior=behavior,
        ))
        if res and res.get("question"):
            out.append({
                "question": res["question"],
                "golden_answer": res.get("golden_answer", base["golden_answer"]),
                "type": behavior.value,
                "answer_type": "refusal",
                "source_documents": base.get("source_documents", []),
                "supporting_spans": base.get("supporting_spans", []),
                "persona": base.get("persona", Persona.ALL_STAFF.value),
                "intent_cluster": "behavioral",
                "_stage": 5,
            })
    return out

"""
v3 orchestrator: Parse(0) → Generate(1/2) → SelfContainedGate(4) →
Ground → Verify(3) → Behavioral(5) → Output(6).

Produces a 100-question test set (80 content + 20 behavioral) distributed by
persona/intent, organized by locality tier, with verified gold answers and
supporting spans for fair cross-architecture grading.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import QuestionType
from . import generation as gen_mod
from .config import V3Config
from .corpus import Corpus, CAT_CV, CAT_PROJECTS
from .grounding import locate_spans, is_grounded
from .llm import LLM
from .prompts import (
    EXPECTED_BEHAVIORS_NL,
)
from .schema import (
    IntentCluster, Persona, Tier, QuestionRecord, SupportingSpan, Verification,
)
from .selfcontained import run_gate
from .verification import deterministic_capability, solver_check, VerifyResult


_PERSONAS = {
    IntentCluster.CAPABILITY: [Persona.SALES, Persona.STRATEGY, Persona.TECHNICAL],
    IntentCluster.REFERENCE: [Persona.SALES, Persona.TECHNICAL],
    IntentCluster.POLICY: [Persona.ALL_STAFF],
    IntentCluster.PROCESS: [Persona.TECHNICAL, Persona.ALL_STAFF],
}
_STAGE1_INTENTS = {IntentCluster.POLICY, IntentCluster.PROCESS}
_STAGE2_INTENTS = {IntentCluster.CAPABILITY, IntentCluster.REFERENCE}


class V3Pipeline:
    def __init__(self, cfg: V3Config):
        cfg.validate()
        self.cfg = cfg
        self.gen = LLM(cfg.generator, "gen")
        self.solver = LLM(cfg.solver, "solver")
        self.judge = LLM(cfg.judge, "judge")
        self._uid = 0

    # ── public ──────────────────────────────────────────
    def run(self) -> str:
        t0 = time.time()
        print("v3 Test Set Creator\n" + "=" * 50)
        corpus = Corpus.load(
            self.cfg.input_documents_path, self.cfg.cv_subset_size,
            self.cfg.random_seed,
        )
        print(f"Loaded {len(corpus.docs)} documents "
              f"(CV subset: {len(corpus.cv_subset)})")
        corpus_index = gen_mod.build_corpus_index(corpus)

        records: List[QuestionRecord] = []

        # ── Content (80) ────────────────────────────────
        content_counts = self.cfg.content_counts()
        print(f"\nContent plan: "
              + ", ".join(f"{k.value}={v}" for k, v in content_counts.items()))
        for intent, target in content_counts.items():
            recs = self._make_content(corpus, corpus_index, intent, target)
            records.extend(recs)
            print(f"  [{intent.value}] kept {len(recs)}/{target}")

        # ── Behavioral (20) ─────────────────────────────
        print("\nBehavioral plan: "
              + ", ".join(f"{t.value}={n}"
                          for t, n in self.cfg.behavioral_quota.items()))
        records.extend(self._make_behavioral(corpus, corpus_index, records))

        # ── Output (Stage 6) ────────────────────────────
        elapsed = time.time() - t0
        out_path = self._save(corpus, records, elapsed)
        self._print_report(records, elapsed)
        print(f"\nOutput: {out_path}")
        return out_path

    # ── content ─────────────────────────────────────────
    def _make_content(
        self, corpus: Corpus, corpus_index: str,
        intent: IntentCluster, target: int,
    ) -> List[QuestionRecord]:
        personas = _PERSONAS[intent]
        oversample = max(target + 2, math.ceil(target * self.cfg.oversample_factor))
        if intent in _STAGE1_INTENTS:
            raw = gen_mod.stage1_doc_questions(
                self.gen, corpus, intent=intent, personas=personas, n=oversample)
        else:
            raw = gen_mod.stage2_corpus_questions(
                self.gen, corpus, corpus_index, intent=intent,
                personas=personas, n=oversample)

        out: List[QuestionRecord] = []
        for q in raw:
            if len(out) >= target:
                break
            rec = self._process_content_one(corpus, intent, q)
            if rec is not None:
                out.append(rec)
        return out

    def _process_content_one(
        self, corpus: Corpus, intent: IntentCluster, q: Dict,
    ) -> Optional[QuestionRecord]:
        question = (q.get("question") or "").strip()
        if not question:
            return None
        subject = q.get("intended_subject", "")
        scope = q.get("intended_scope", "")

        # Stage 4 — self-containedness gate
        gate = run_gate(
            self.judge, question=question, intended_subject=subject,
            intended_scope=scope, max_repair_attempts=self.cfg.max_repair_attempts)
        if not gate.passed:
            return None
        question = gate.question

        stage = q.get("_stage", 1)
        key_term = q.get("key_term", "")
        proposed = q.get("golden_answer") or q.get("proposed_answer") or ""
        src = q.get("source_documents", [])

        # Grounding (provenance located after generation)
        if stage == 2 and key_term:
            det = deterministic_capability(corpus, key_term=key_term)
            spans = det.spans if det else []
            if not spans:
                spans = locate_spans(
                    corpus, golden_answer=proposed, source_documents=src,
                    supporting_quote=q.get("supporting_quote", ""), key_term=key_term)
        else:
            spans = locate_spans(
                corpus, golden_answer=proposed, source_documents=src,
                supporting_quote=q.get("supporting_quote", ""), key_term=key_term)

        if not is_grounded(spans):
            return None

        # Stage 3 — solver verification (independent answerer)
        excel = corpus.excel_doc()
        extra = (excel.full_text[:6000] if (excel and stage == 2) else "")
        vr: VerifyResult = solver_check(
            self.solver, corpus, question=question, proposed_answer=proposed,
            spans=spans, extra_evidence=extra)
        if not vr.ok or not vr.golden_answer.strip():
            return None

        tier = self._tier_for(q, stage)
        src_docs = sorted({s.document for s in vr.spans}) or src
        return self._build_record(
            qtype=self._content_type(intent, tier), tier=tier, intent=intent,
            persona=Persona(q.get("persona", Persona.ALL_STAFF.value)),
            question=question, golden_answer=vr.golden_answer,
            answer_type=q.get("answer_type", "span"),
            subject=subject, scope=scope,
            variant=q.get("underspecified_variant", ""),
            spans=vr.spans, source_documents=src_docs,
            verification=vr.verification, repaired=gate.repaired)

    # ── behavioral ──────────────────────────────────────
    def _make_behavioral(
        self, corpus: Corpus, corpus_index: str, content: List[QuestionRecord],
    ) -> List[QuestionRecord]:
        out: List[QuestionRecord] = []
        quota = self.cfg.behavioral_quota

        # hallucination — verify absence in corpus
        for q in gen_mod.stage5_hallucination(
                self.gen, corpus, corpus_index, n=quota.get(
                    QuestionType.HALLUCINATION_TEST, 0) * 2):
            if len([r for r in out if r.type == QuestionType.HALLUCINATION_TEST.value]) \
                    >= quota.get(QuestionType.HALLUCINATION_TEST, 0):
                break
            key_term = q.get("key_term", "")
            if key_term and corpus.documents_mentioning(key_term):
                continue  # not actually absent → reject
            gate = run_gate(self.judge, question=q["question"],
                            intended_subject=q.get("intended_subject", ""),
                            intended_scope="corpus",
                            max_repair_attempts=self.cfg.max_repair_attempts)
            if not gate.passed:
                continue
            out.append(self._build_record(
                qtype=QuestionType.HALLUCINATION_TEST.value, tier=Tier.BEHAVIORAL,
                intent=IntentCluster.BEHAVIORAL, persona=Persona.SALES,
                question=gate.question,
                golden_answer=q.get("golden_answer",
                                    "Deze informatie is niet terug te vinden in de documenten."),
                answer_type="refusal", subject=q.get("intended_subject", ""),
                scope="corpus", variant="", spans=[], source_documents=[],
                verification=Verification(method="corpus_search", solver_reproduced=None,
                                          notes="key_term absent from corpus"),
                repaired=gate.repaired))

        # ambiguous
        for q in gen_mod.stage5_ambiguous(self.gen, corpus,
                                          n=quota.get(QuestionType.AMBIGUOUS_QUESTIONS, 0)):
            rec = self._simple_doc_behavioral(corpus, q, QuestionType.AMBIGUOUS_QUESTIONS)
            if rec:
                out.append(rec)

        # multi-turn
        for q in gen_mod.stage5_multiturn(self.gen, corpus,
                                          n=quota.get(QuestionType.MULTI_TURN_FOLLOWUP, 0)):
            rec = self._simple_doc_behavioral(
                corpus, q, QuestionType.MULTI_TURN_FOLLOWUP,
                question_key="question_turn_1", answer_key="answer_turn_1")
            if rec:
                rec.question_turn_2 = q.get("question_turn_2", "")
                rec.answer_turn_2 = q.get("answer_turn_2", "")
                out.append(rec)

        # injection + aggro — graft onto simple verified content questions
        base = [
            {"question": r.question, "golden_answer": r.golden_answer,
             "source_documents": r.source_documents,
             "supporting_spans": r.supporting_spans, "persona": r.persona}
            for r in content if r.locality_tier in (Tier.T1.value, Tier.T2.value)
        ] or [
            {"question": r.question, "golden_answer": r.golden_answer,
             "source_documents": r.source_documents,
             "supporting_spans": r.supporting_spans, "persona": r.persona}
            for r in content
        ]
        for behavior in (QuestionType.PROMPT_INJECTION, QuestionType.ADVERSARIAL_AGGRO):
            n = quota.get(behavior, 0)
            for g in gen_mod.stage5_graft(self.gen, behavior=behavior,
                                          base_questions=base, n=n):
                out.append(self._build_record(
                    qtype=behavior.value, tier=Tier.BEHAVIORAL,
                    intent=IntentCluster.BEHAVIORAL,
                    persona=Persona(g.get("persona", Persona.ALL_STAFF.value)),
                    question=g["question"], golden_answer=g["golden_answer"],
                    answer_type="refusal", subject="", scope="",
                    variant="", spans=list(g.get("supporting_spans", [])),
                    source_documents=g.get("source_documents", []),
                    verification=Verification(method="grafted", solver_reproduced=None,
                                              notes="grafted onto verified content question"),
                    repaired=False))
        return out

    def _simple_doc_behavioral(
        self, corpus: Corpus, q: Dict, qtype: QuestionType,
        question_key: str = "question", answer_key: str = "golden_answer",
    ) -> Optional[QuestionRecord]:
        question = (q.get(question_key) or "").strip()
        answer = (q.get(answer_key) or "").strip()
        if not question or not answer:
            return None
        gate = run_gate(self.judge, question=question,
                        intended_subject=q.get("intended_subject", ""),
                        intended_scope=", ".join(q.get("source_documents", [])),
                        max_repair_attempts=self.cfg.max_repair_attempts)
        if not gate.passed:
            return None
        spans = locate_spans(corpus, golden_answer=answer,
                             source_documents=q.get("source_documents", []),
                             supporting_quote=q.get("supporting_quote", ""))
        return self._build_record(
            qtype=qtype.value, tier=Tier.BEHAVIORAL, intent=IntentCluster.BEHAVIORAL,
            persona=Persona.ALL_STAFF, question=gate.question, golden_answer=answer,
            answer_type="span", subject=q.get("intended_subject", ""),
            scope="", variant="", spans=spans,
            source_documents=q.get("source_documents", []),
            verification=Verification(method="generator", solver_reproduced=None, notes=""),
            repaired=gate.repaired)

    # ── helpers ─────────────────────────────────────────
    def _tier_for(self, q: Dict, stage: int) -> Tier:
        raw = (q.get("locality_tier") or "").upper()
        if raw in (Tier.T1.value, Tier.T2.value, Tier.T3.value, Tier.T4.value):
            return Tier(raw)
        return Tier.T2 if stage == 1 else Tier.T3

    def _content_type(self, intent: IntentCluster, tier: Tier) -> str:
        if tier in (Tier.T3, Tier.T4):
            return QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS.value
        if intent == IntentCluster.POLICY:
            return QuestionType.DIRECT_LOOKUP.value if tier == Tier.T1 \
                else QuestionType.LISTS_EXTRACTION.value
        if intent == IntentCluster.PROCESS:
            return QuestionType.MULTI_HOP_WITHIN_CORPUS.value
        return QuestionType.PARAPHRASE_LOOKUP.value

    def _build_record(
        self, *, qtype: str, tier: Tier, intent: IntentCluster, persona: Persona,
        question: str, golden_answer: str, answer_type: str, subject: str,
        scope: str, variant: str, spans: List[SupportingSpan],
        source_documents: List[str], verification: Verification, repaired: bool,
    ) -> QuestionRecord:
        self._uid += 1
        src_docs = source_documents or sorted({s.document for s in spans})
        if tier in (Tier.T3, Tier.T4):
            min_docs = max(2, len({s.document for s in spans}))
        elif tier == Tier.BEHAVIORAL:
            min_docs = max(1, len({s.document for s in spans}))
        else:
            min_docs = 1
        return QuestionRecord(
            id=f"v3_{qtype}_{self._uid}",
            type=qtype, locality_tier=tier.value, min_docs_required=min_docs,
            persona=persona.value, intent_cluster=intent.value,
            question=question, golden_answer=golden_answer,
            answer_type=answer_type, intended_subject=subject,
            intended_scope=scope,
            underspecified_variant=(variant if self.cfg.underspecified_variants else ""),
            supporting_spans=list(spans), source_documents=src_docs,
            verification=verification,
            expected_behavior=EXPECTED_BEHAVIORS_NL.get(qtype, ""),
            metadata={
                "generated_by": "v3",
                "generator_model": self.cfg.generator.model,
                "solver_model": self.cfg.solver.model,
                "self_contained": True,
                "repaired": repaired,
            },
        )

    # ── Stage 6 — output ────────────────────────────────
    def _save(self, corpus: Corpus, records: List[QuestionRecord], elapsed: float) -> str:
        summary = self._summary(records)
        test_set = {
            "metadata": {
                "created_at": time.time(),
                "pipeline_version": "v3",
                "language": self.cfg.language,
                "config": {
                    "total_questions": self.cfg.total_questions,
                    "content_total": self.cfg.content_total,
                    "behavioral_total": self.cfg.behavioral_total,
                    "generator_model": self.cfg.generator.model,
                    "solver_model": self.cfg.solver.model,
                    "judge_model": self.cfg.judge.model,
                    "cv_subset": corpus.cv_subset,
                },
                "metrics": {
                    "elapsed_seconds": round(elapsed, 1),
                    "llm_calls": {
                        "generator": self.gen.calls,
                        "solver": self.solver.calls,
                        "judge": self.judge.calls,
                    },
                },
            },
            "summary": summary,
            "questions": [r.to_dict() for r in records],
        }
        name = self.cfg.corpus_name.replace(" ", "_")
        corpus_dir = Path(self.cfg.output_path) / name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{name}_{datetime.now().strftime('%d_%m_%y_T%H_%M')}.json"
        fpath = corpus_dir / fname
        fpath.write_text(json.dumps(test_set, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        return str(fpath)

    def _summary(self, records: List[QuestionRecord]) -> Dict:
        by_tier = Counter(r.locality_tier for r in records)
        by_type = Counter(r.type for r in records)
        by_persona = Counter(r.persona for r in records)
        by_intent = Counter(r.intent_cluster for r in records)
        grounded = sum(1 for r in records if r.supporting_spans)
        reproduced = sum(1 for r in records
                         if r.verification.solver_reproduced is True)
        multi_src = sum(1 for r in records if len(r.source_documents) > 1)
        t1 = by_tier.get(Tier.T1.value, 0)
        t3t4 = by_tier.get(Tier.T3.value, 0) + by_tier.get(Tier.T4.value, 0)
        return {
            "total_questions": len(records),
            "by_tier": dict(by_tier),
            "by_type": dict(by_type),
            "by_persona": dict(by_persona),
            "by_intent": dict(by_intent),
            "grounded_ratio": round(grounded / max(1, len(records)), 3),
            "solver_reproduced_ratio": round(reproduced / max(1, len(records)), 3),
            "multi_source_questions": multi_src,
            "tier_floors": {
                "t1": {"actual": t1, "floor": self.cfg.tier_floor_t1,
                       "ok": t1 >= self.cfg.tier_floor_t1},
                "t3_t4": {"actual": t3t4, "floor": self.cfg.tier_floor_t3_t4,
                          "ok": t3t4 >= self.cfg.tier_floor_t3_t4},
            },
        }

    def _print_report(self, records: List[QuestionRecord], elapsed: float) -> None:
        s = self._summary(records)
        print("\n" + "=" * 50)
        print(f"Generated {s['total_questions']} questions in {elapsed:.0f}s")
        print(f"  by tier:    {s['by_tier']}")
        print(f"  by intent:  {s['by_intent']}")
        print(f"  by persona: {s['by_persona']}")
        print(f"  grounded:   {s['grounded_ratio']:.0%} | "
              f"solver-reproduced: {s['solver_reproduced_ratio']:.0%}")
        tf = s["tier_floors"]
        print(f"  tier floors: T1 {tf['t1']['actual']}/{tf['t1']['floor']} "
              f"({'OK' if tf['t1']['ok'] else 'LOW'}), "
              f"T3+T4 {tf['t3_t4']['actual']}/{tf['t3_t4']['floor']} "
              f"({'OK' if tf['t3_t4']['ok'] else 'LOW'})")
        print(f"  llm calls: gen={self.gen.calls} solver={self.solver.calls} "
              f"judge={self.judge.calls}")

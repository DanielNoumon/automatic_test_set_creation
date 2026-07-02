"""
v3 data model: tiers, personas, intents, and the question record.

A v3 question is defined by user intent + verified answer, and carries its
grounding as a *list of supporting spans* (provenance), not a single chunk.
This is what lets the eval score retrieval-recall fairly across architectures.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any


class Tier(str, Enum):
    """Retrieval-locality tier: how much of the corpus must be read."""
    T1 = "T1"          # local: one contiguous span
    T2 = "T2"          # intra-document distributed
    T3 = "T3"          # cross-document aggregation
    T4 = "T4"          # global / structural
    BEHAVIORAL = "behavioral"  # graded on behavior; tier may be N/A


class Persona(str, Enum):
    SALES = "sales"
    STRATEGY = "strategy"
    TECHNICAL = "technical"
    ALL_STAFF = "all_staff"


class IntentCluster(str, Enum):
    CAPABILITY = "capability"        # A+B+C: discovery / depth / staffing
    REFERENCE = "reference"          # D: reference / evidence
    POLICY = "policy"                # E: policy / HR
    PROCESS = "process"              # F: process
    BEHAVIORAL = "behavioral"        # injection / aggro / hallucination / etc.


class AnswerType(str, Enum):
    SCALAR = "scalar"
    SET = "set"
    ENUMERATION = "enumeration"
    SPAN = "span"
    REFUSAL = "refusal"


@dataclass
class SupportingSpan:
    """A piece of evidence that grounds the answer — provenance, not the
    unit the question was built from.

    `text` is a windowed snippet centered on the match (for compact display);
    `full_text` is the complete source section and is what the solver receives
    as evidence, so answers are never cut off mid-section."""
    document: str
    page: int
    text: str
    full_text: str = ""

    def evidence(self) -> str:
        """Full section text for the solver (falls back to the snippet)."""
        return self.full_text or self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "text": self.text,
            "full_text": self.full_text or self.text,
        }


@dataclass
class Verification:
    """Auditable record of how a question's gold answer was verified.

    method: deterministic | solver_agent | corpus_search | generator | grafted
    """
    method: str = "solver_agent"
    solver_reproduced: Optional[bool] = None   # solver agreed with proposed?
    notes: str = ""
    proposed_answer: str = ""      # generator's original answer (pre-verification)
    solver_answer: str = ""        # independent solver's raw answer
    evidence_docs: int = 0         # #distinct docs of evidence the solver saw
    deterministic_matches: Optional[List[str]] = None  # docs matched by key_term

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default grading rubric attached to every question. The eval harness reads it.
DEFAULT_GRADING_RUBRIC = {
    "answer_correctness": "LLM-judge vs golden_answer (0-1)",
    "completeness": "fraction of gold set items recovered (for set/enumeration)",
    "retrieval_recall": "fraction of supporting_spans surfaced by the system",
    "robustness": "correctness drop clean -> underspecified_variant",
}


@dataclass
class QuestionRecord:
    """A single v3 test-set entry."""
    id: str
    type: str                      # question-type value (reuses pipeline.config.QuestionType)
    locality_tier: str             # Tier value
    min_docs_required: int
    persona: str
    intent_cluster: str

    question: str                  # Dutch, self-contained
    golden_answer: str             # Dutch, verified
    answer_type: str

    intended_subject: str = ""
    intended_scope: str = ""
    underspecified_variant: str = ""   # Dutch, terse rephrasing (robustness sub-benchmark)

    supporting_spans: List[SupportingSpan] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)

    verification: Verification = field(default_factory=Verification)
    expected_behavior: str = ""
    grading_rubric: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_GRADING_RUBRIC)
    )

    # Multi-turn followup (optional)
    question_turn_2: Optional[str] = None
    answer_turn_2: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "locality_tier": self.locality_tier,
            "min_docs_required": self.min_docs_required,
            "persona": self.persona,
            "intent_cluster": self.intent_cluster,
            "question": self.question,
            "intended_subject": self.intended_subject,
            "intended_scope": self.intended_scope,
            "underspecified_variant": self.underspecified_variant,
            "golden_answer": self.golden_answer,
            "answer_type": self.answer_type,
            "supporting_spans": [s.to_dict() for s in self.supporting_spans],
            "source_documents": self.source_documents,
            "verification": self.verification.to_dict(),
            "expected_behavior": self.expected_behavior,
            "grading_rubric": self.grading_rubric,
            "metadata": self.metadata,
        }
        if self.question_turn_2 is not None:
            d["question_turn_2"] = self.question_turn_2
            d["answer_turn_2"] = self.answer_turn_2
        return d

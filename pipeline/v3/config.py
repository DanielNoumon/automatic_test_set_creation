"""
v3 configuration: distribution (hybrid, 100 questions), persona/intent weights,
behavioral quota, tier floors, CV-subset size, and model wiring.

Numbers come from docs/test_set_v3_design.md §9.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

from pipeline.config import QuestionType
from .schema import IntentCluster


@dataclass
class ModelConfig:
    """One Azure OpenAI / OpenAI deployment."""
    model: str
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2025-04-01-preview"

    @staticmethod
    def _base_url(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        p = urlparse(raw)
        return f"{p.scheme}://{p.netloc}"

    @classmethod
    def from_env(
        cls,
        deployment_env: str,
        endpoint_env: str,
        version_env: str,
        default_model: str,
        default_version: str = "2025-04-01-preview",
    ) -> "ModelConfig":
        """Build from environment variables (Azure deployment style)."""
        return cls(
            model=os.getenv(deployment_env, default_model),
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            azure_endpoint=cls._base_url(os.getenv(endpoint_env)),
            azure_api_version=os.getenv(version_env, default_version),
        )


@dataclass
class V3Config:
    # ── Paths ───────────────────────────────────────────
    input_documents_path: str = "data/files_for_test_set"
    output_path: str = "data/test_sets"
    corpus_name: str = "DSL_corpus_v3"

    # ── Language (global rule) ──────────────────────────
    language: str = "nl"   # all questions/answers/variants in Dutch

    # ── Distribution ────────────────────────────────────
    total_questions: int = 100
    # Content vs behavioral split (behavioral is INSIDE the total): 90% / 10%.
    behavioral_total: int = 10      # → content_total = total - behavioral_total

    # Intent weights applied to the CONTENT questions (must sum to 1.0)
    intent_weights: Dict[IntentCluster, float] = field(default_factory=lambda: {
        IntentCluster.CAPABILITY: 0.50,   # A+B+C
        IntentCluster.REFERENCE: 0.20,    # D
        IntentCluster.POLICY: 0.20,       # E
        IntentCluster.PROCESS: 0.10,      # F
    })

    # Behavioral quota (must sum to behavioral_total)
    behavioral_quota: Dict[QuestionType, int] = field(default_factory=lambda: {
        QuestionType.HALLUCINATION_TEST: 3,
        QuestionType.ADVERSARIAL_AGGRO: 2,
        QuestionType.PROMPT_INJECTION: 2,
        QuestionType.AMBIGUOUS_QUESTIONS: 2,
        QuestionType.MULTI_TURN_FOLLOWUP: 1,
    })

    # Tier floors (verified after generation; warn/nudge if unmet)
    tier_floor_t1: int = 12
    tier_floor_t3_t4: int = 40

    # ── Corpus curation ─────────────────────────────────
    # The CV subset is the pool of source CVs used to SEED the behavioral
    # ambiguous / multi-turn questions (Stage 5). It is NOT used for content
    # generation: T1/T2 come from policy/process docs, and T3/T4 aggregation
    # uses ALL CVs via the corpus index + verification search.
    cv_subset_size: int = 7
    # Optional explicit subset (basenames or relative paths). If set, these CVs
    # are used instead of the deterministic auto-pick; unmatched slots are
    # backfilled by the auto-picker.
    cv_subset_override: Optional[List[str]] = field(default_factory=lambda: [
        "CV DSL Esmee Valk oktober 2025.docx",
        "CV Floor Deben DSL 2025.pdf",
        "CV_Linda_NL.pdf",
        "DSL CV Younes Seghrouchni 12-11-2024 ENGELS.docx",
        "Julian CV.docx",
        # replaced Sebastiaan Peek + mvdleijgraaf (Marine) with these two:
        "CV Koen DSL 2026.docx",
        "CV Carmen Wolvius 022026.docx",
    ])

    # ── Generation knobs ────────────────────────────────
    # Over-generate then filter through gates/dedup; yield is < 1.
    oversample_factor: float = 2.0
    max_repair_attempts: int = 1
    underspecified_variants: bool = True   # Option 1: graded robustness sub-benchmark

    # ── Models ──────────────────────────────────────────
    generator: ModelConfig = None       # set in from_env
    solver: ModelConfig = None          # Stage 3 independent answerer (GPT 5.4)
    judge: ModelConfig = None           # isolation/quality judge

    # ── Reproducibility ─────────────────────────────────
    random_seed: int = 42

    # ── Derived helpers ─────────────────────────────────
    @property
    def content_total(self) -> int:
        return self.total_questions - self.behavioral_total

    def content_counts(self) -> Dict[IntentCluster, int]:
        """Integer per-intent content counts that sum to content_total."""
        raw = {k: self.content_total * w for k, w in self.intent_weights.items()}
        counts = {k: int(math.floor(v)) for k, v in raw.items()}
        # Distribute rounding remainder to the largest fractional parts
        remainder = self.content_total - sum(counts.values())
        order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order[:remainder]:
            counts[k] += 1
        return counts

    def validate(self) -> None:
        w = sum(self.intent_weights.values())
        if abs(w - 1.0) > 1e-6:
            raise ValueError(f"intent_weights must sum to 1.0, got {w}")
        bq = sum(self.behavioral_quota.values())
        if bq != self.behavioral_total:
            raise ValueError(
                f"behavioral_quota sums to {bq}, expected {self.behavioral_total}"
            )

    @classmethod
    def from_env(cls, **overrides) -> "V3Config":
        """Build a config wired to Azure deployments from the environment.

        All three roles (generator, judge, solver) use the GPT 5.4 deployment
        (DEPLOYMENT_NAME_SOLVER / AZURE_OPENAI_ENDPOINT_SOLVER / API_VERSION_SOLVER).
        Note: gen and solver being the same model weakens the independence of the
        `solver_reproduced` check, but GPT 5.4 gives the strongest verification.
        """
        gpt54 = ModelConfig.from_env(
            "DEPLOYMENT_NAME_SOLVER",
            "AZURE_OPENAI_ENDPOINT_SOLVER",
            "API_VERSION_SOLVER",
            default_model=os.getenv("DEPLOYMENT_NAME_SOLVER", "gpt-5.4"),
        )
        if gpt54.azure_endpoint is None:
            # fall back to the GPT-5-mini endpoint/key if no solver endpoint set
            mini = ModelConfig.from_env(
                "DEPLOYMENT_NAME_GPT5_MINI",
                "AZURE_OPENAI_ENDPOINT_GPT5_MINI",
                "API_VERSION_GPT5_MINI",
                default_model="gpt-5-mini",
            )
            gpt54.azure_endpoint = mini.azure_endpoint
            gpt54.azure_api_version = mini.azure_api_version
        # Separate instances per role (so call counts stay distinct), same model.
        import copy
        cfg = cls(
            generator=copy.copy(gpt54),
            judge=copy.copy(gpt54),
            solver=copy.copy(gpt54),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

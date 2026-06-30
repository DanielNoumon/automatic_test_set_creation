"""
v3 configuration: distribution (hybrid, 100 questions), persona/intent weights,
behavioral quota, tier floors, CV-subset size, and model wiring.

Numbers come from docs/test_set_v3_design.md §9.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
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

    # ── Distribution (§9) ───────────────────────────────
    total_questions: int = 100
    # Content vs behavioral split (behavioral is INSIDE the total)
    behavioral_total: int = 20      # → content_total = total - behavioral_total

    # Intent weights applied to the CONTENT questions (must sum to 1.0)
    intent_weights: Dict[IntentCluster, float] = field(default_factory=lambda: {
        IntentCluster.CAPABILITY: 0.50,   # A+B+C
        IntentCluster.REFERENCE: 0.20,    # D
        IntentCluster.POLICY: 0.20,       # E
        IntentCluster.PROCESS: 0.10,      # F
    })

    # Behavioral quota (must sum to behavioral_total)
    behavioral_quota: Dict[QuestionType, int] = field(default_factory=lambda: {
        QuestionType.HALLUCINATION_TEST: 6,
        QuestionType.ADVERSARIAL_AGGRO: 4,
        QuestionType.PROMPT_INJECTION: 4,
        QuestionType.AMBIGUOUS_QUESTIONS: 3,
        QuestionType.MULTI_TURN_FOLLOWUP: 3,
    })

    # Tier floors (verified after generation; warn/nudge if unmet)
    tier_floor_t1: int = 12
    tier_floor_t3_t4: int = 40

    # ── Corpus curation ─────────────────────────────────
    cv_subset_size: int = 7         # CVs used for single-doc (T1/T2) generation
    # (all CVs are used for cross-corpus T3/T4 regardless)

    # ── Generation knobs ────────────────────────────────
    # Over-generate then filter through gates; yield is < 1.
    oversample_factor: float = 1.6
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

        Generator + judge default to the GPT-5-mini deployment already in .env.
        Solver defaults to a GPT 5.4 deployment (add DEPLOYMENT_NAME_SOLVER /
        AZURE_OPENAI_ENDPOINT_SOLVER / API_VERSION_SOLVER to .env).
        """
        gen = ModelConfig.from_env(
            "DEPLOYMENT_NAME_GPT5_MINI",
            "AZURE_OPENAI_ENDPOINT_GPT5_MINI",
            "API_VERSION_GPT5_MINI",
            default_model="gpt-5-mini",
        )
        # Independent solver — GPT 5.4. Falls back to the GPT-5-mini endpoint/key
        # if no dedicated solver deployment is configured yet.
        solver = ModelConfig.from_env(
            "DEPLOYMENT_NAME_SOLVER",
            "AZURE_OPENAI_ENDPOINT_SOLVER",
            "API_VERSION_SOLVER",
            default_model=os.getenv("DEPLOYMENT_NAME_SOLVER", "gpt-5.4"),
        )
        if solver.azure_endpoint is None:
            # no dedicated solver deployment → reuse generator endpoint/key
            solver.azure_endpoint = gen.azure_endpoint
            solver.azure_api_version = gen.azure_api_version
        cfg = cls(generator=gen, judge=gen, solver=solver)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

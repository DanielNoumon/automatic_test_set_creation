"""
Stage 4 — self-containedness gate (LLM-as-judge, no regex).

  1. Isolation judge: shown ONLY the question (no passage); must resolve the
     referent. Catches deixis and bare definite references.
  2. Repair: failed questions are rewritten using the known intended_subject,
     then re-judged.

The solver-agent (Stage 3) is the downstream backstop for anything that slips
through.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM
from .prompts import build_isolation_judge_prompt, build_repair_prompt


@dataclass
class GateResult:
    passed: bool
    question: str          # possibly repaired
    repaired: bool
    reason: str


def run_gate(
    judge: LLM, *, question: str, intended_subject: str, intended_scope: str,
    max_repair_attempts: int = 1,
) -> GateResult:
    """Judge → (repair → re-judge)* up to max_repair_attempts."""
    q = question
    repaired = False
    for attempt in range(max_repair_attempts + 1):
        verdict = judge.json(build_isolation_judge_prompt(question=q))
        if verdict is None:
            # Judge failed to respond — fail open is risky; fail closed instead.
            return GateResult(False, q, repaired, "judge no response")
        if verdict.get("self_contained") is True:
            return GateResult(True, q, repaired, "ok")
        reason = verdict.get("reason", "not self-contained")
        if attempt >= max_repair_attempts:
            return GateResult(False, q, repaired, reason)
        # Repair and retry
        fix = judge.json(build_repair_prompt(
            question=q,
            intended_subject=intended_subject or verdict.get("missing_subject", ""),
            intended_scope=intended_scope,
        ))
        if not fix or not (fix.get("question") or "").strip():
            return GateResult(False, q, repaired, reason)
        q = fix["question"].strip()
        repaired = True
    return GateResult(False, q, repaired, "exhausted repairs")

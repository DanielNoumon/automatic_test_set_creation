"""
LLM-as-judge quality scoring for generated Q+A pairs.

After a Q+A pair passes deterministic validation, this module
makes a single LLM call to score it on four dimensions (1-5):
  - self_containedness
  - answer_accuracy
  - naturalness
  - difficulty_alignment

Returns per-dimension scores, a composite score, and a pass/fail
decision based on the configured minimum threshold.
"""
import json
import re
from typing import Dict, Any, Optional

from ..config import QualityConfig, QuestionType
from ..generation.llm_client import LLMClient


# Dimension descriptions used in the judge prompt
_DIMENSION_DESCRIPTIONS = {
    "self_containedness": (
        "Can the question be fully understood WITHOUT seeing the "
        "source passage or knowing which document/section it came "
        "from? A score of 5 means the question is completely "
        "standalone. A score of 1 means it relies on implicit "
        "context (e.g. 'deze rol', 'dit document', 'hierboven')."
    ),
    "answer_accuracy": (
        "Does the answer correctly and completely address the "
        "question, using ONLY information present in the passage? "
        "A score of 5 means the answer is precise, complete, and "
        "fully grounded. A score of 1 means the answer is wrong, "
        "incomplete, or contains invented facts."
    ),
    "naturalness": (
        "Does the question sound like something a real employee "
        "would ask an HR/knowledge base chatbot? A score of 5 "
        "means it reads naturally and fluently. A score of 1 "
        "means it sounds robotic, overly formal, or artificially "
        "constructed."
    ),
    "difficulty_alignment": (
        "Does the actual difficulty of answering this question "
        "match the REQUESTED difficulty level? For 'easy': the "
        "answer should be a simple factoid lookup. For 'medium': "
        "some reasoning or paraphrasing required. For 'hard': "
        "multi-step reasoning, cross-referencing, or nuanced "
        "interpretation needed. A score of 5 means perfect "
        "alignment. A score of 1 means a gross mismatch (e.g. "
        "a trivial factoid labeled 'hard')."
    ),
}


def _build_judge_prompt(
    question: str,
    answer: str,
    passage: str,
    question_type: str,
    difficulty: str,
    dimensions: tuple,
) -> list:
    """Build the messages for the LLM judge call."""
    dim_block = "\n".join(
        f"- **{dim}**: {_DIMENSION_DESCRIPTIONS[dim]}"
        for dim in dimensions
        if dim in _DIMENSION_DESCRIPTIONS
    )

    user_prompt = f"""You are a strict quality judge for generated Q&A pairs.

PASSAGE (used to generate the Q&A):
\"\"\"
{passage}
\"\"\"

QUESTION TYPE: {question_type}
REQUESTED DIFFICULTY: {difficulty}

GENERATED QUESTION:
\"\"\"{question}\"\"\"

GENERATED ANSWER:
\"\"\"{answer}\"\"\"

Score the Q&A pair on each dimension below from 1 (worst) to 5 (best).
Be critical — most pairs should score 3-4. Reserve 5 for exceptional quality.

DIMENSIONS:
{dim_block}

Return ONLY a JSON object (no extra text):
{{
{chr(10).join(f'  "{dim}": <1-5>,' for dim in dimensions)}
  "reasoning": "One sentence explaining the weakest dimension."
}}"""

    return [
        {
            "role": "system",
            "content": (
                "You are a precise Q&A quality judge. Score each "
                "dimension from 1 to 5. Be strict and consistent. "
                "Return only valid JSON."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _try_parse_scores(
    text: str, dimensions: tuple,
) -> Optional[Dict[str, Any]]:
    """Parse the judge LLM response into scores."""
    if not text:
        return None
    text = text.strip()

    # Try direct parse
    parsed = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from code block
    if not parsed:
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if match:
            try:
                parsed = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

    # Try first {...} block
    if not parsed:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

    if not parsed or not isinstance(parsed, dict):
        return None

    # Validate all dimension scores are present and numeric
    scores = {}
    for dim in dimensions:
        val = parsed.get(dim)
        if val is None:
            return None
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        scores[dim] = max(1.0, min(5.0, val))

    scores["reasoning"] = parsed.get("reasoning", "")
    return scores


class QualityScorer:
    """LLM-as-judge quality scorer for generated Q+A pairs."""

    def __init__(self, llm: LLMClient, config: QualityConfig):
        self.llm = llm
        self.config = config

    def score(
        self,
        question: str,
        answer: str,
        passage: str,
        question_type: QuestionType,
        difficulty: str = None,
    ) -> Dict[str, Any]:
        """Score a Q+A pair using the LLM judge.

        Returns dict with per-dimension scores (1-5),
        composite score, reasoning, and pass/fail.
        """
        if not self.config.enabled:
            return {
                "quality_scores": {},
                "composite_score": 0.0,
                "quality_passed": True,
                "quality_reason": "scoring disabled",
            }

        # Drop difficulty_alignment when difficulty is disabled
        dims = self.config.dimensions
        if not difficulty:
            dims = tuple(
                d for d in dims
                if d != "difficulty_alignment"
            )

        messages = _build_judge_prompt(
            question=question,
            answer=answer,
            passage=passage,
            question_type=question_type.value,
            difficulty=difficulty or "N/A",
            dimensions=dims,
        )

        try:
            response = self.llm.completion(messages=messages)
        except Exception as e:
            print(f"      Quality scoring LLM error: {e}")
            # On LLM failure, let the question pass
            return {
                "quality_scores": {},
                "composite_score": 0.0,
                "quality_passed": True,
                "quality_reason": f"scoring error: {e}",
            }

        scores = _try_parse_scores(response, dims)

        if not scores:
            print(
                "      Quality scoring: could not parse "
                "LLM response, passing by default"
            )
            return {
                "quality_scores": {},
                "composite_score": 0.0,
                "quality_passed": True,
                "quality_reason": "parse error",
            }

        reasoning = scores.pop("reasoning", "")
        dim_scores = {
            k: v for k, v in scores.items()
            if isinstance(v, (int, float))
        }
        composite = (
            sum(dim_scores.values()) / len(dim_scores)
            if dim_scores else 0.0
        )
        composite = round(composite, 2)
        passed = composite >= self.config.min_score

        return {
            "quality_scores": dim_scores,
            "composite_score": composite,
            "quality_passed": passed,
            "quality_reason": reasoning,
        }

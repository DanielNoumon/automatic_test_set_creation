"""
Q+A generator: takes a passage and produces a question-answer pair
using the LLM. This is the ONLY step that requires an LLM call.
"""
import json
import re
from typing import Optional, Dict

from ..config import QuestionType
from .llm_client import LLMClient
from .prompts import build_qa_prompt


class QAGenerator:
    """Generate question-answer pairs from passages using LLM."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        passage: str,
        source_documents: list,
        chapter: str,
        question_type: QuestionType,
        difficulty: str = None,
        doc_metadata: dict = None,
    ) -> Optional[Dict[str, str]]:
        """Generate a single Q+A pair from a passage.

        Returns dict with 'question' and 'answer' keys,
        or None if generation fails.
        """
        messages = build_qa_prompt(
            passage=passage,
            source_documents=source_documents,
            chapter=chapter,
            question_type=question_type,
            difficulty=difficulty,
            doc_metadata=doc_metadata,
        )

        try:
            response = self.llm.completion(
                messages=messages,
            )
        except Exception as e:
            print(f"    LLM error: {e}")
            return None

        # Parse JSON response
        result = _try_parse_json(response)
        if not result:
            return None

        # Multi-turn followup returns four fields
        if question_type == QuestionType.MULTI_TURN_FOLLOWUP:
            q1 = result.get("question_turn_1", "").strip()
            a1 = result.get("answer_turn_1", "").strip()
            q2 = result.get("question_turn_2", "").strip()
            a2 = result.get("answer_turn_2", "").strip()
            if not q1 or not a1 or not q2 or not a2:
                return None
            return {
                "question": q1,
                "answer": a1,
                "question_turn_2": q2,
                "answer_turn_2": a2,
            }

        q = result.get("question", "").strip()
        a = result.get("answer", "").strip()

        if not q or not a or a.lower() == "n/a":
            return None

        return {"question": q, "answer": a}


def _try_parse_json(text: str) -> Optional[dict]:
    """Try to parse a JSON object from text."""
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None

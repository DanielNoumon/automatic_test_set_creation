"""
Entry point for v3 test-set generation.

Usage:
    python run_v3.py

Wires Azure OpenAI env vars (same pattern as main.py) and runs the v3 pipeline
to produce a 100-question DSL test set (80 content + 20 behavioral), in Dutch,
organized by locality tier. See docs/test_set_v3_design.md.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.config import QuestionType
from pipeline.v3.config import V3Config
from pipeline.v3.pipeline import V3Pipeline

_ROOT = Path(__file__).resolve().parent
load_dotenv()

# Mirror main.py: expose the Azure key/endpoint under the generic env names the
# LLMClient reads, derived from the GPT-5-mini deployment.
if os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]


# Verification set: 30 questions at the 90/10 content:behavioral split.
# Content = 27 (capability 14 / reference 5 / policy 5 / process 3),
# behavioral = 3 (hallucination / injection / multi-turn). Total = 30.
_VALIDATE_OVERRIDES = dict(
    corpus_name="DSL_corpus_v3_verify",
    total_questions=30,
    behavioral_total=3,
    behavioral_quota={
        QuestionType.HALLUCINATION_TEST: 1,
        QuestionType.PROMPT_INJECTION: 1,
        QuestionType.MULTI_TURN_FOLLOWUP: 1,
    },
    oversample_factor=1.8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 test-set generation")
    parser.add_argument(
        "--validate", action="store_true",
        help="small validation run (13 questions) to inspect output quality",
    )
    args = parser.parse_args()

    overrides = dict(
        input_documents_path=str(_ROOT / "data" / "files_for_test_set"),
        output_path=str(_ROOT / "data" / "test_sets"),
        corpus_name="DSL_corpus_v3",
    )
    if args.validate:
        overrides.update(_VALIDATE_OVERRIDES)

    cfg = V3Config.from_env(**overrides)
    print(f"Mode: {'VALIDATE' if args.validate else 'FULL'} | "
          f"total={cfg.total_questions} (content={cfg.content_total}, "
          f"behavioral={cfg.behavioral_total})")
    print(f"Generator: {cfg.generator.model} | Solver: {cfg.solver.model} | "
          f"Judge: {cfg.judge.model}")
    pipeline = V3Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()

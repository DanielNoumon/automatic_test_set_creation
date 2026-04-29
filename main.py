"""
Test Set Creator — Entry Point.

Index-first architecture: Parse → Index → Select → Generate → Validate.
Only uses LLM for Q+A generation (1 call per question).
"""
import os
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
import mlflow

from pipeline.config import (
    TestSetConfig, LLMConfig, QuestionType, QuestionConfig,
    SelectionConfig, ValidationConfig, QualityConfig,
    PipelineConfig,
)
from pipeline.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parent

# Load environment variables
load_dotenv()

# Configure MLflow tracking
mlflow.openai.autolog()
mlflow_db = _ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.disable_system_metrics_logging()
print(f"MLflow tracking enabled (SQLite): {mlflow_db}")

# Set Azure OpenAI environment variables
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
_raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT5_MINI", "")
# AzureOpenAI needs only the base URL (scheme + host)
if _raw_endpoint:
    _p = urlparse(_raw_endpoint)
    azure_endpoint = f"{_p.scheme}://{_p.netloc}"
else:
    azure_endpoint = None
azure_api_version = os.getenv(
    "API_VERSION_GPT5_MINI", "2025-04-01-preview"
)

# Also set the generic env vars for the LLM client
if azure_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = azure_key
if azure_endpoint and not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
if azure_api_version and not os.getenv("AZURE_OPENAI_API_VERSION"):
    os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version


def build_config(
    model: str = "gpt-5",
    corpus_name: str = "DSL_corpus",
    input_documents_path: str = None,
    output_path: str = None,
    question_types: dict = None,
    random_seed: int = 42,
    difficulty_enabled: bool = False,
    # Selection
    min_section_words: int = 15,
    passage_max_chars: int = 1200,
    long_context_max_chars: int = 4000,
    long_context_span_size: int = 4,
    # Validation
    min_keyword_ratio: float = 0.25,
    min_context_match: float = 0.3,
    min_context_length: int = 30,
    # Quality scoring (LLM-as-judge)
    quality_scoring_enabled: bool = True,
    quality_min_score: float = 3.5,
    # Pipeline
    hallucination_bm25_threshold: float = 3.0,
    hallucination_overlap_threshold: float = 0.5,
    mlflow_experiment_name: str = "test-set-creation",
) -> TestSetConfig:
    """Build a TestSetConfig with sensible defaults.

    Override any parameter to experiment with different settings.
    """
    if input_documents_path is None:
        input_documents_path = str(_ROOT / "data" / "files_for_test_set")
    if output_path is None:
        output_path = str(_ROOT / "data" / "test_sets")
    if question_types is None:
        question_types = ALL_TYPES_DEFAULT

    return TestSetConfig(
        llm=LLMConfig(
            api_key=azure_key,
            model=model,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        ),
        corpus_name=corpus_name,
        input_documents_path=input_documents_path,
        output_path=output_path,
        question_types=question_types,
        selection=SelectionConfig(
            min_section_words=min_section_words,
            passage_max_chars=passage_max_chars,
            long_context_max_chars=long_context_max_chars,
            long_context_span_size=long_context_span_size,
        ),
        validation=ValidationConfig(
            min_keyword_ratio=min_keyword_ratio,
            min_context_match=min_context_match,
            min_context_length=min_context_length,
        ),
        quality=QualityConfig(
            enabled=quality_scoring_enabled,
            min_score=quality_min_score,
        ),
        pipeline=PipelineConfig(
            hallucination_bm25_threshold=hallucination_bm25_threshold,
            hallucination_overlap_threshold=hallucination_overlap_threshold,
            mlflow_experiment_name=mlflow_experiment_name,
        ),
        difficulty_enabled=difficulty_enabled,
        random_seed=random_seed,
    )


def run_pipeline(config: TestSetConfig):
    """Run the pipeline with MLflow tracking."""
    mlflow.set_experiment(config.pipeline.mlflow_experiment_name)

    enabled = config.get_enabled_question_types()
    total_q = config.get_total_enabled_questions()

    run_name = (
        f"{config.corpus_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    mlflow.start_run(run_name=run_name)

    # Log configuration params
    mlflow.log_param("pipeline_version", "v2")
    mlflow.log_param("model", config.llm.model)
    mlflow.log_param("corpus_name", config.corpus_name)
    mlflow.log_param("questions_requested", total_q)
    mlflow.log_param("random_seed", config.random_seed)
    mlflow.log_param(
        "enabled_types",
        ", ".join(t.value for t in enabled),
    )
    mlflow.log_param(
        "input_path", config.input_documents_path
    )
    for qtype in enabled:
        qcfg = config.question_types[qtype]
        mlflow.log_param(
            f"count_{qtype.value}", qcfg.count
        )
        mlflow.log_param(
            f"difficulty_{qtype.value}", qcfg.difficulty
        )

    try:
        start_time = time.time()
        pipeline = Pipeline(config)
        output_file, test_set = pipeline.run()
        duration = time.time() - start_time

        # --- Core metrics ---
        questions = test_set.get("questions", [])
        actual_count = len(questions)
        mlflow.log_metric(
            "total_duration_seconds", round(duration, 1)
        )
        mlflow.log_metric(
            "questions_generated", actual_count
        )
        mlflow.log_metric(
            "questions_requested", total_q
        )
        mlflow.log_metric(
            "yield_ratio",
            round(actual_count / total_q, 3)
            if total_q > 0 else 0,
        )

        # --- Pipeline phase timings ---
        m = test_set.get(
            "metadata", {}
        ).get("metrics", {})
        mlflow.log_metric(
            "parse_time", round(m.get("parse_time", 0), 3)
        )
        mlflow.log_metric(
            "index_time", round(m.get("index_time", 0), 3)
        )
        mlflow.log_metric(
            "llm_calls", m.get("llm_calls", 0)
        )

        # --- Per-type counts ---
        by_type = m.get("questions_by_type", {})
        for qtype_name, cnt in by_type.items():
            mlflow.log_metric(
                f"count_{qtype_name}", cnt
            )

        # --- Quality metrics ---
        match_ratios = [
            q.get("metadata", {}).get(
                "context_match_ratio", 0
            )
            for q in questions
        ]
        if match_ratios:
            mlflow.log_metric(
                "avg_context_match_ratio",
                round(
                    sum(match_ratios) / len(match_ratios),
                    3,
                ),
            )
            mlflow.log_metric(
                "min_context_match_ratio",
                round(min(match_ratios), 3),
            )

        grounded = sum(
            1 for q in questions
            if q.get("metadata", {}).get(
                "answer_grounded", False
            )
        )
        if actual_count > 0:
            mlflow.log_metric(
                "answer_grounded_ratio",
                round(grounded / actual_count, 3),
            )

        # --- LLM-as-judge quality scores ---
        composite_scores = [
            q.get("metadata", {}).get(
                "composite_score", 0
            )
            for q in questions
            if q.get("metadata", {}).get(
                "composite_score", 0
            ) > 0
        ]
        if composite_scores:
            mlflow.log_metric(
                "avg_quality_score",
                round(
                    sum(composite_scores)
                    / len(composite_scores), 2
                ),
            )
            mlflow.log_metric(
                "min_quality_score",
                round(min(composite_scores), 2),
            )

        # --- Artifact ---
        mlflow.log_artifact(output_file)

        mlflow.end_run()
        print(f"\nDone! Output: {output_file}")
        print("MLflow run completed")
        return output_file, test_set

    except Exception as e:
        print(f"ERROR: {e}")
        mlflow.log_param("error", str(e)[:250])
        mlflow.end_run(status="FAILED")
        raise


# ====================================================
# DEFAULT QUESTION TYPE PRESETS
# ====================================================
# Adjust counts and difficulties here, then re-run.

ALL_TYPES_DEFAULT = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="easy",
    ),
    QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="medium",
    ),
    QuestionType.SPECIFIC_JARGON: QuestionConfig(
        enabled=True, count=3, difficulty="medium",
    ),
    QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
        enabled=True, count=2, difficulty="hard",
    ),
    QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: QuestionConfig(
        enabled=True, count=2, difficulty="hard",
    ),
    QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
        enabled=False, count=2, difficulty="medium",
    ),
    QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
    QuestionType.LISTS_EXTRACTION: QuestionConfig(
        enabled=False, count=2, difficulty="easy",
    ),
    QuestionType.HALLUCINATION_TEST: QuestionConfig(
        enabled=False, count=2, difficulty="medium",
    ),
    QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
    QuestionType.PROMPT_INJECTION: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
}

# ====================================================
# QUICK-RUN PRESETS — pick one or make your own
# ====================================================

QUICK_TEST = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=1, difficulty="easy",
    ),
    QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
        enabled=True, count=1, difficulty="medium",
    ),
}

FULL_RUN = {
    qt: QuestionConfig(enabled=True, count=5, difficulty=cfg.difficulty)
    for qt, cfg in ALL_TYPES_DEFAULT.items()
}


if __name__ == "__main__":

    # ================================================
    # CONFIGURATION - Adjust parameters here
    # ================================================

    # -- LLM --
    MODEL = os.getenv("DEPLOYMENT_NAME_GPT5_MINI", "gpt-5")

    # -- Naming / paths --
    CORPUS_NAME = "DSL_corpus"  # Label used in output filenames and MLflow
    INPUT_PATH = "data/files_for_test_set"  # Input PDFs/TXT/MD folder
    OUTPUT_PATH = "data/test_sets"  # Output folder for test-set JSONs

    # -- Selection thresholds (control which document sections qualify as passages) --
    MIN_SECTION_WORDS = 15  # Min words for a section to be a passage candidate
    PASSAGE_MAX_CHARS = 1200  # Max chars per passage before truncation
    LONG_CONTEXT_MAX_CHARS = 4000  # Max chars for combined long-context passage
    LONG_CONTEXT_SPAN_SIZE = 4  # Consecutive sections to combine for long-context

    # -- Validation thresholds (decide whether a generated Q+A pair is accepted) --
    MIN_KEYWORD_RATIO = 0.25  # Answer keyword fraction needed to count as grounded
    MIN_CONTEXT_MATCH = 0.3  # Min chunk-overlap ratio with source documents
    MIN_CONTEXT_LENGTH = 30  # Passages shorter than this (chars) are rejected

    # -- Quality scoring (LLM-as-judge, scores each accepted Q+A on a 1-5 scale) --
    QUALITY_SCORING_ENABLED = True   # set False to skip quality scoring
    QUALITY_MIN_SCORE = 3.5          # reject Q+A pairs below this composite score

    # -- Pipeline (hallucination detection and experiment tracking) --
    HALLUCINATION_BM25_THRESHOLD = 3.0  # BM25 floor for hallucination verification
    HALLUCINATION_OVERLAP_THRESHOLD = 0.5  # Overlap ratio to consider sections the same
    MLFLOW_EXPERIMENT_NAME = "test-set-creation"  # MLflow experiment name
    RANDOM_SEED = 42  # Fixed seed for reproducible selection; change for new samples

    # -- Difficulty (future feature, disabled for now) --
    DIFFICULTY_ENABLED = False  # Set True to enable difficulty hints in prompts

    # -- Question types to generate --
    # Set enabled=False or remove a type to disable it.
    # If set, overrides count for ALL enabled types (e.g. 1 for quick test)
    COUNT_OVERRIDE = 3  # Set to None to use per-type counts

    QUESTION_TYPES = {
        QuestionType.DIRECT_LOOKUP: QuestionConfig(
            enabled=True, count=3, difficulty="easy",
        ),
        QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        QuestionType.SPECIFIC_JARGON: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        # QuestionType.CROSS_DOCUMENT_CONFLICT: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard",
        # ),
        QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        QuestionType.LISTS_EXTRACTION: QuestionConfig(
            enabled=True, count=3, difficulty="easy",
        ),
        QuestionType.HALLUCINATION_TEST: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        QuestionType.PROMPT_INJECTION: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        # QuestionType.PINPOINTING_QUOTING: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium",
        # ),
        QuestionType.LONG_CONTEXT_SYNTHESIS: QuestionConfig(
            enabled=True, count=3, difficulty="hard",
        ),
        QuestionType.AMBIGUOUS_QUESTIONS: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        # QuestionType.TOOL_CALL_CHECK: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard",
        # ),
        # QuestionType.TABLES_EXTRACTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium",
        # ),
        # QuestionType.INFOGRAPHIC_EXTRACTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium",
        # ),
        QuestionType.MULTI_TURN_FOLLOWUP: QuestionConfig(
            enabled=True, count=3, difficulty="medium",
        ),
        # QuestionType.ACCESS_CONTROL: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard",
        # ),
    }

    # ================================================

    # Apply count override if set
    if COUNT_OVERRIDE is not None:
        QUESTION_TYPES = {
            qt: QuestionConfig(
                enabled=cfg.enabled,
                count=COUNT_OVERRIDE,
                difficulty=cfg.difficulty,
            )
            for qt, cfg in QUESTION_TYPES.items()
        }

    config = build_config(
        model=MODEL,
        corpus_name=CORPUS_NAME,
        input_documents_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        question_types=QUESTION_TYPES,
        random_seed=RANDOM_SEED,
        difficulty_enabled=DIFFICULTY_ENABLED,
        min_section_words=MIN_SECTION_WORDS,
        passage_max_chars=PASSAGE_MAX_CHARS,
        long_context_max_chars=LONG_CONTEXT_MAX_CHARS,
        long_context_span_size=LONG_CONTEXT_SPAN_SIZE,
        min_keyword_ratio=MIN_KEYWORD_RATIO,
        min_context_match=MIN_CONTEXT_MATCH,
        min_context_length=MIN_CONTEXT_LENGTH,
        quality_scoring_enabled=QUALITY_SCORING_ENABLED,
        quality_min_score=QUALITY_MIN_SCORE,
        hallucination_bm25_threshold=HALLUCINATION_BM25_THRESHOLD,
        hallucination_overlap_threshold=HALLUCINATION_OVERLAP_THRESHOLD,
        mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
    )
    run_pipeline(config)

"""
Configuration for Test Set Creator.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class QuestionType(Enum):
    DIRECT_LOOKUP = "direct_lookup"
    PARAPHRASE_LOOKUP = "paraphrase_lookup"
    SPECIFIC_JARGON = "specific_jargon"
    MULTI_HOP_WITHIN_CORPUS = "multi_hop_within_corpus"
    MULTI_HOP_BETWEEN_DOCUMENTS = "multi_hop_between_documents"
    CROSS_DOCUMENT_CONFLICT = "cross_document_conflict"
    TEMPORAL_QUESTIONS = "temporal_questions"
    PINPOINTING_QUOTING = "pinpointing_quoting"
    LONG_CONTEXT_SYNTHESIS = "long_context_synthesis"
    NEEDLE_IN_HAYSTACK = "needle_in_haystack"
    AMBIGUOUS_QUESTIONS = "ambiguous_questions"
    TOOL_CALL_CHECK = "tool_call_check"
    TABLES_EXTRACTION = "tables_extraction"
    LISTS_EXTRACTION = "lists_extraction"
    INFOGRAPHIC_EXTRACTION = "infographic_extraction"
    HALLUCINATION_TEST = "hallucination_test"
    ADVERSARIAL_AGGRO = "adversarial_aggro"
    PROMPT_INJECTION = "prompt_injection"
    MULTI_TURN_FOLLOWUP = "multi_turn_followup"
    ACCESS_CONTROL = "access_control"


@dataclass
class QuestionConfig:
    """Configuration for an individual question type."""
    enabled: bool = False
    count: int = 5
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class LLMConfig:
    """Configuration for the LLM used in Q+A generation."""
    api_key: Optional[str] = None
    model: str = "gpt-5"
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-12-01-preview"


@dataclass
class SelectionConfig:
    """Configuration for passage selection strategies."""
    min_section_words: int = 15
    passage_max_chars: int = 1200
    long_context_max_chars: int = 4000
    long_context_span_size: int = 4


@dataclass
class ValidationConfig:
    """Configuration for Q+A validation."""
    min_keyword_ratio: float = 0.25
    min_context_match: float = 0.3
    min_context_length: int = 30


@dataclass
class QualityConfig:
    """Configuration for LLM-as-judge quality scoring."""
    enabled: bool = True
    min_score: float = 3.5          # reject Q+A pairs below this composite score (1-5)
    dimensions: tuple = (
        "self_containedness",       # understandable without source context?
        "answer_accuracy",          # answer correctly addresses the question given passage?
        "naturalness",              # sounds like a real user question?
        "difficulty_alignment",     # matches the requested difficulty level?
    )


@dataclass
class PipelineConfig:
    """Configuration for pipeline-level behaviour."""
    hallucination_bm25_threshold: float = 3.0
    hallucination_overlap_threshold: float = 0.5
    mlflow_experiment_name: str = "test-set-creation"


@dataclass
class TestSetConfig:
    """Main configuration for test set creation."""
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Paths
    input_documents_path: str = "data/files_for_test_set"
    output_path: str = "data/test_sets"

    # Naming
    corpus_name: str = "test_set"

    # Question type configurations
    question_types: Dict[QuestionType, QuestionConfig] = field(default_factory=dict)

    # Sub-configs
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Feature flags
    difficulty_enabled: bool = False

    # General settings
    random_seed: int = 42

    def get_enabled_question_types(self) -> List[QuestionType]:
        return [qt for qt, cfg in self.question_types.items() if cfg.enabled]

    def get_total_enabled_questions(self) -> int:
        return sum(cfg.count for cfg in self.question_types.values() if cfg.enabled)

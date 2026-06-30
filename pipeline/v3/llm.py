"""
Thin LLM helper for v3: wraps the existing LLMClient and JSON parsing, and
counts calls. One instance per ModelConfig (generator / solver / judge).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline.generation.llm_client import LLMClient
from pipeline.generation.qa_generator import _try_parse_json
from .config import ModelConfig


class LLM:
    def __init__(self, cfg: ModelConfig, label: str = ""):
        self.cfg = cfg
        self.label = label
        self.calls = 0
        self._client: Optional[LLMClient] = None

    @property
    def client(self) -> LLMClient:
        # Lazy construction so importing the package never needs API keys.
        if self._client is None:
            self._client = LLMClient(
                api_key=self.cfg.api_key,
                model=self.cfg.model,
                azure_endpoint=self.cfg.azure_endpoint,
                azure_api_version=self.cfg.azure_api_version,
            )
        return self._client

    def text(self, messages: List[Dict[str, str]]) -> Optional[str]:
        self.calls += 1
        try:
            return self.client.completion(messages=messages)
        except Exception as e:  # noqa: BLE001
            print(f"    [llm:{self.label}] error: {e}")
            return None

    def json(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        raw = self.text(messages)
        if raw is None:
            return None
        return _try_parse_json(raw)

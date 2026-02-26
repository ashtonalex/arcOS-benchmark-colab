"""
Shared LLM client for arcOS benchmark.

Provides an async OpenRouter client used by both the main pipeline (Phase 5)
and the SARM benchmark layer.
"""

from .client import OpenRouterClient, LLMRequest, LLMResponse

__all__ = [
    "OpenRouterClient",
    "LLMRequest",
    "LLMResponse",
]

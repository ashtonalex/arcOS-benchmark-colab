"""
Async OpenRouter API client with retry logic and batch support.

Used by both the main LLM pipeline (Phase 5) and the SARM benchmark layer.
"""

import asyncio
import logging
import os
import time
import random
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    usage: dict
    latency_ms: float
    finish_reason: str


@dataclass
class LLMRequest:
    """Request for batch LLM calls."""
    model: str
    messages: list
    temperature: float = 0.0
    max_tokens: int = 512
    seed: Optional[int] = None


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenRouterClient:
    """Async HTTP client for OpenRouter API with exponential backoff retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required: pass api_key or set OPENROUTER_API_KEY env var"
            )
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with OpenRouterClient(...) as client:'"
            )

    async def complete(
        self,
        model: str,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat completion request with retry logic."""
        self._ensure_client()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        url = f"{self.api_base}/chat/completions"
        last_error = None

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                response = await self._client.post(url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    latency = (time.monotonic() - start) * 1000
                    choice = data["choices"][0]
                    return LLMResponse(
                        content=choice["message"]["content"],
                        model=data.get("model", model),
                        usage=data.get("usage", {}),
                        latency_ms=latency,
                        finish_reason=choice.get("finish_reason", "unknown"),
                    )

                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    response.raise_for_status()

                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )

                # Retry with backoff
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = (2 ** attempt) + random.uniform(0, 1)

                    if response.status_code == 429:
                        logger.warning(
                            "OpenRouter rate limit hit (429) for model %s — "
                            "waiting %.1fs before retry %d/%d",
                            model, delay, attempt + 1, self.max_retries,
                        )
                    else:
                        logger.warning(
                            "OpenRouter HTTP %d for model %s — "
                            "waiting %.1fs before retry %d/%d",
                            response.status_code, model, delay,
                            attempt + 1, self.max_retries,
                        )

                    await asyncio.sleep(delay)
                else:
                    if response.status_code == 429:
                        logger.error(
                            "OpenRouter rate limit (429) exhausted all %d retries for model %s",
                            self.max_retries, model,
                        )
                    else:
                        logger.error(
                            "OpenRouter HTTP %d exhausted all %d retries for model %s",
                            response.status_code, self.max_retries, model,
                        )

            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)

        raise last_error

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Run multiple completion requests concurrently."""
        tasks = [
            self.complete(
                model=req.model,
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                seed=req.seed,
            )
            for req in requests
        ]
        return await asyncio.gather(*tasks)

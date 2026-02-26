"""Tests for the shared OpenRouter async LLM client."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.client import OpenRouterClient, LLMRequest, LLMResponse


# ========== Constructor Tests ==========


def test_missing_api_key_raises():
    """Constructor raises ValueError when no API key is available."""
    with patch.dict(os.environ, {}, clear=True):
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                OpenRouterClient()


def test_explicit_api_key():
    """Constructor accepts an explicit API key."""
    client = OpenRouterClient(api_key="test-key-123")
    assert client.api_key == "test-key-123"


def test_env_var_api_key():
    """Constructor reads API key from environment."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key-456"}):
        client = OpenRouterClient()
        assert client.api_key == "env-key-456"


# ========== Async Tests ==========


@pytest.fixture
def mock_response_ok():
    """A mock httpx response for a successful completion."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [
            {
                "message": {"content": "Hello, world!"},
                "finish_reason": "stop",
            }
        ],
        "model": "test/model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return resp


@pytest.fixture
def mock_response_429():
    """A mock httpx response for rate limiting."""
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"Retry-After": "0.01"}
    resp.request = MagicMock()
    return resp


@pytest.fixture
def mock_response_500():
    """A mock httpx response for server error."""
    resp = MagicMock()
    resp.status_code = 500
    resp.headers = {}
    resp.request = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_complete_returns_llm_response(mock_response_ok):
    """complete() returns a well-formed LLMResponse on success."""
    client = OpenRouterClient(api_key="test")
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response_ok)
    client._client = mock_http

    result = await client.complete(
        model="test/model",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello, world!"
    assert result.model == "test/model"
    assert result.finish_reason == "stop"
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_complete_retries_on_429(mock_response_429, mock_response_ok):
    """complete() retries on 429 then succeeds."""
    client = OpenRouterClient(api_key="test", max_retries=2)
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=[mock_response_429, mock_response_ok])
    client._client = mock_http

    result = await client.complete(
        model="test/model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result.content == "Hello, world!"
    assert mock_http.post.call_count == 2


@pytest.mark.asyncio
async def test_complete_fails_after_max_retries(mock_response_500):
    """complete() raises after exhausting retries on 500."""
    client = OpenRouterClient(api_key="test", max_retries=1)
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response_500)
    client._client = mock_http

    with pytest.raises(Exception):
        await client.complete(
            model="test/model",
            messages=[{"role": "user", "content": "Hi"}],
        )


@pytest.mark.asyncio
async def test_complete_batch_concurrent(mock_response_ok):
    """complete_batch() runs requests concurrently and returns correct count."""
    client = OpenRouterClient(api_key="test")
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response_ok)
    client._client = mock_http

    requests = [
        LLMRequest(model="m1", messages=[{"role": "user", "content": "a"}]),
        LLMRequest(model="m2", messages=[{"role": "user", "content": "b"}]),
        LLMRequest(model="m3", messages=[{"role": "user", "content": "c"}]),
    ]

    results = await client.complete_batch(requests)
    assert len(results) == 3
    assert all(isinstance(r, LLMResponse) for r in results)


@pytest.mark.asyncio
async def test_context_manager():
    """Context manager opens and closes the HTTP client cleanly."""
    async with OpenRouterClient(api_key="test") as client:
        assert client._client is not None
    assert client._client is None


@pytest.mark.asyncio
async def test_ensure_client_raises_without_init():
    """Calling complete without context manager raises RuntimeError."""
    client = OpenRouterClient(api_key="test")
    with pytest.raises(RuntimeError, match="Client not initialized"):
        await client.complete(model="m", messages=[])

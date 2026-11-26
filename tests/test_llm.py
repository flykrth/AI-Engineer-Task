"""Mocked unit tests for the provider-agnostic LLM and embedding adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel

from src.llm import (
    GeminiAdapter,
    LLMConfig,
    MissingAPIKeyError,
    OllamaAdapter,
    OpenAIAdapter,
    StructuredOutputError,
    UnsupportedProviderError,
    create_llm_adapter,
    load_llm_config,
    parse_structured_response,
)


class SampleAnswer(BaseModel):
    """Test schema for structured generation."""

    answer: str
    confidence: int


def test_provider_selection_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    default_cfg = load_llm_config(load_env=False)
    assert default_cfg.provider == "gemini"
    assert default_cfg.chat_model == "gemini-3.5-flash-lite"
    assert default_cfg.embed_model == "gemini-embedding-001"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    env_cfg = load_llm_config(load_env=False)
    assert env_cfg.provider == "openai"
    assert env_cfg.chat_model == "gpt-4o-mini"

    # Explicit override (such as CLI --provider) takes precedence over env.
    override_cfg = load_llm_config(provider="ollama", load_env=False)
    assert override_cfg.provider == "ollama"
    assert override_cfg.base_url == "http://localhost:11434"


def test_unsupported_provider_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown_cloud")
    with pytest.raises(UnsupportedProviderError, match="gemini, openai, ollama"):
        load_llm_config(load_env=False)


def test_missing_api_keys_raise_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        create_llm_adapter(provider="gemini", load_env=False)

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        create_llm_adapter(provider="openai", load_env=False)

    # Ollama needs no API key to initialize.
    ollama_adapter = create_llm_adapter(provider="ollama", load_env=False)
    assert isinstance(ollama_adapter, OllamaAdapter)


def test_parse_structured_response_handles_fences_prose_and_trailing_commas() -> None:
    raw = (
        "Here is the JSON result:\n"
        "```json\n"
        "{\n"
        '  "answer": "Kabir owns payments",\n'
        '  "confidence": 5,\n'
        "}\n"
        "```\n"
        "Done."
    )
    parsed = parse_structured_response(raw, SampleAnswer)
    assert parsed == SampleAnswer(answer="Kabir owns payments", confidence=5)


def test_gemini_adapter_mocked_text_structured_and_embeddings() -> None:
    calls: list[str] = []

    class FakeModels:
        def generate_content(
            self, *, model: str, contents: str, config: Any
        ) -> SimpleNamespace:
            calls.append(f"gen:{model}:{contents[:12]}")
            if getattr(config, "response_schema", None) is not None:
                return SimpleNamespace(
                    parsed=SampleAnswer(answer="23 Nov", confidence=4),
                    text='{"answer": "23 Nov", "confidence": 4}',
                )
            return SimpleNamespace(text="Razorpay test mode")

        def embed_content(self, *, model: str, contents: list[str]) -> SimpleNamespace:
            calls.append(f"embed:{model}:{len(contents)}")
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents]
            )

    fake_client = SimpleNamespace(models=FakeModels())
    cfg = LLMConfig(
        provider="gemini",
        api_key="test-key",
        chat_model="gemini-3.5-flash-lite",
        embed_model="gemini-embedding-001",
    )
    adapter = GeminiAdapter(cfg, client=fake_client)

    assert adapter.generate_text("Payments?", system_prompt="Be concise") == "Razorpay test mode"
    structured = adapter.generate_structured("Report due?", SampleAnswer)
    assert structured == SampleAnswer(answer="23 Nov", confidence=4)
    assert adapter.embed_texts([]) == []
    assert adapter.embed_texts(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert len(calls) == 3


def test_openai_adapter_mocked_text_structured_and_embeddings() -> None:
    class FakeCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI reply"))]
            )

        def parse(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=SampleAnswer(answer="Search unassigned", confidence=5),
                            content='{"answer": "Search unassigned", "confidence": 5}',
                        )
                    )
                ]
            )

    class FakeEmbeddings:
        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in input]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
        embeddings=FakeEmbeddings(),
    )
    cfg = LLMConfig(
        provider="openai",
        api_key="test-key",
        chat_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
    )
    adapter = OpenAIAdapter(cfg, client=fake_client)

    assert adapter.generate_text("Hello") == "OpenAI reply"
    assert adapter.generate_structured("Status?", SampleAnswer) == SampleAnswer(
        answer="Search unassigned", confidence=5
    )
    assert adapter.embed_texts(["hello"]) == [[1.0, 2.0]]


def test_ollama_adapter_and_structured_json_repair_retry() -> None:
    chat_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url.path == "/api/chat":
            chat_attempts += 1
            if chat_attempts == 1:
                # Missing required "confidence" field triggers one-shot repair.
                content = '{"answer": "partial"}'
            else:
                content = '{"answer": "fixed", "confidence": 3}'
            return httpx.Response(200, json={"message": {"content": content}})
        if request.url.path == "/api/embed":
            payload = json.loads(request.content.decode("utf-8"))
            vectors = [[float(i), 0.5] for i, _ in enumerate(payload["input"])]
            return httpx.Response(200, json={"embeddings": vectors})
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        chat_model="llama3.1:8b",
        embed_model="nomic-embed-text",
    )
    adapter = OllamaAdapter(cfg, http_client=http_client)

    result = adapter.generate_structured("Check", SampleAnswer)
    assert result == SampleAnswer(answer="fixed", confidence=3)
    assert chat_attempts == 2
    assert adapter.embed_texts(["one", "two"]) == [[0.0, 0.5], [1.0, 0.5]]


def test_structured_output_raises_when_repair_also_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        chat_model="llama3.1:8b",
        embed_model="nomic-embed-text",
    )
    adapter = OllamaAdapter(cfg, http_client=http_client)

    with pytest.raises(StructuredOutputError, match="SampleAnswer"):
        adapter.generate_structured("Check", SampleAnswer)


def test_exponential_backoff_retries_on_429_and_503_but_not_404() -> None:
    sleep_delays: list[float] = []
    attempts = 0

    class FlakyModels:
        def generate_content(
            self, *, model: str, contents: str, config: Any
        ) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise genai_errors.APIError(
                    429, {"error": {"code": 429, "message": "rate limited"}}
                )
            if attempts == 2:
                raise genai_errors.APIError(
                    503, {"error": {"code": 503, "message": "unavailable"}}
                )
            return SimpleNamespace(text="ok after retries")

    cfg = LLMConfig(
        provider="gemini",
        api_key="test-key",
        chat_model="gemini-3.5-flash-lite",
        embed_model="gemini-embedding-001",
        max_retries=3,
        retry_base_delay=1.0,
    )
    adapter = GeminiAdapter(
        cfg, client=SimpleNamespace(models=FlakyModels()), sleep_fn=sleep_delays.append
    )

    assert adapter.generate_text("hi") == "ok after retries"
    assert attempts == 3
    assert sleep_delays == [1.0, 2.0]

    # Non-transient 404 must fail immediately without sleeping.
    sleep_delays.clear()

    class NotFoundModels:
        def generate_content(
            self, *, model: str, contents: str, config: Any
        ) -> SimpleNamespace:
            raise genai_errors.APIError(
                404, {"error": {"code": 404, "message": "model not found"}}
            )

    failing_adapter = GeminiAdapter(
        cfg, client=SimpleNamespace(models=NotFoundModels()), sleep_fn=sleep_delays.append
    )
    with pytest.raises(genai_errors.APIError):
        failing_adapter.generate_text("hi")
    assert sleep_delays == []


def test_openai_retries_on_rate_limit_error() -> None:
    sleep_delays: list[float] = []
    attempts = 0

    class FlakyCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                resp = httpx.Response(429, request=req)
                raise openai.RateLimitError("rate limit", response=resp, body=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FlakyCompletions()),
        embeddings=SimpleNamespace(),
    )
    cfg = LLMConfig(
        provider="openai",
        api_key="test-key",
        chat_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        max_retries=2,
        retry_base_delay=0.5,
    )
    adapter = OpenAIAdapter(cfg, client=fake_client, sleep_fn=sleep_delays.append)

    assert adapter.generate_text("hi") == "recovered"
    assert sleep_delays == [0.5]


def test_request_delay_pacing_and_retry_after_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("LLM_REQUEST_DELAY_SECONDS", raising=False)
    default_cfg = load_llm_config(load_env=False)
    assert default_cfg.request_delay_seconds == 0.0

    monkeypatch.setenv("LLM_REQUEST_DELAY_SECONDS", "1.25")
    paced_cfg = load_llm_config(load_env=False)
    assert paced_cfg.request_delay_seconds == 1.25

    sleep_delays: list[float] = []
    attempts = 0

    class FakeModels:
        def generate_content(
            self, *, model: str, contents: str, config: Any
        ) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                req = httpx.Request("POST", "https://example.com")
                resp = httpx.Response(429, headers={"Retry-After": "4.0"}, request=req)
                raise genai_errors.APIError(
                    429, {"error": {"code": 429, "message": "rate limit"}}, resp
                )
            return SimpleNamespace(text="ok")

    adapter = GeminiAdapter(
        paced_cfg, client=SimpleNamespace(models=FakeModels()), sleep_fn=sleep_delays.append
    )
    assert adapter.generate_text("one") == "ok"
    assert adapter.generate_text("two") == "ok"
    # Pacing sleeps 1.25s before 2nd request, and 429 sleeps Retry-After=4.0s (> exp backoff 1.0s).
    assert sleep_delays == [1.25, 4.0]
